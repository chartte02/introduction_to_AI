#!/usr/bin/env python3
"""解释方式对比脚本（说服力实验）。

对同一段推文，生成三种不同形式的解释：
  1. 仅注意力关键词 — 只看 BERT 认为重要的词
  2. 仅 LLM 解释    — 不看注意力，纯 LLM 分析
  3. LLM + 注意力引导 — 告诉 LLM 模型关注了哪些词，让它结合分析

输出对比 CSV，可用于后续人工评判三种解释方式的说服力。

用法：
    # 单条对比
    python scripts/compare_explain_modes.py --text "BREAKING: Ferguson police..."

    # 批量抽样
    python scripts/compare_explain_modes.py --sample 10 --output outputs/explain_modes.csv
"""

import argparse
import csv
import io
import os
import random
import sys
import time
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from model.preprocessing import clean_text
from model.model import RumorClassifier
from model.attention_viz import (
    extract_attention,
    extract_token_importance,
    get_top_keywords,
)
from model.llm_client import LLMClient
from model.prompts import build_explanation_prompt, SYSTEM_PROMPT

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# ── 注意力引导的增强版 system prompt ──
GUIDED_SYSTEM_PROMPT = """你是一个专业的谣言检测分析师。你的任务是根据一条推文和分类模型的检测结果，分析并解释该分类结果的依据。

特别提示：BERT 分类模型在处理这条推文时，注意力集中在以下关键词上（按重要性排序）。
请你**结合这些关键词**进行分析，说明这些词为什么可能暗示了谣言或非谣言特征。

请从以下角度（选择适用的）进行分析：
1. **信息来源与引用**
2. **措辞与情绪**
3. **事实可验证性**
4. **逻辑一致性**
5. **语言特征**

要求：
- 用 2-4 句话简要分析
- 明确提到模型关注的关键词，说明其与判断的关联
- 使用中文回答
- 不要说"作为AI模型"之类的自我介绍"""


def build_guided_prompt(
    text: str,
    label: int,
    label_name: str,
    confidence: float,
    keywords: list,
) -> list:
    """构建注意力引导的解释 prompt。"""
    keywords_str = "、".join([f"「{w}」(权重 {s:.3f})" for w, s in keywords[:5]])

    user_prompt = f"""以下是一条社交媒体推文，已被谣言检测模型判定为「{label_name}」（置信度: {confidence:.2%}）。

BERT 模型注意力集中的关键词：{keywords_str}

请结合这些关键词，分析推文为什么被判定为{label_name}。

推文内容：
---
{text}
---"""

    return [
        {"role": "system", "content": GUIDED_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def generate_mode1_attention_only(
    tokenizer, input_ids, importance, top_k: int = 8
) -> str:
    """模式 1：仅注意力关键词。"""
    keywords = get_top_keywords(tokenizer, input_ids, importance, top_k=top_k)
    items = [f"{i+1}. 「{w}」(重要性 {s:.3f})" for i, (w, s) in enumerate(keywords)]
    return "模型关注的关键词（按重要性排序）：\n" + "\n".join(items)


def generate_mode2_llm_only(
    text: str, label: int, label_name: str, confidence: float,
    client: LLMClient, llm_model: str,
) -> str:
    """模式 2：纯 LLM 解释（不告诉它注意力）。"""
    messages = build_explanation_prompt(text, label, label_name, confidence)
    return client.chat(messages, model=llm_model)


def generate_mode3_llm_guided(
    text: str, label: int, label_name: str, confidence: float,
    keywords: list, client: LLMClient, llm_model: str,
) -> str:
    """模式 3：LLM + 注意力引导。"""
    messages = build_guided_prompt(text, label, label_name, confidence, keywords)
    return client.chat(messages, model=llm_model)


def load_val_samples(n: int, seed: int = 42) -> list:
    """从 val.csv 随机抽取 n 条。"""
    random.seed(seed)
    samples = []
    with open(PROJECT_ROOT / "data" / "val.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                samples.append({
                    "id": row[0], "text": row[1],
                    "label": int(row[2]), "event": row[3],
                })
    return random.sample(samples, min(n, len(samples)))


def print_single_result(sample, explanations: dict):
    """打印单条对比结果。"""
    print(f"\n{'='*60}")
    print(f"📝 原文: {sample['text'][:100]}...")
    print(f"🔍 判定: {explanations['label_name']} (置信度: {explanations['confidence']:.2%})")
    print(f"{'='*60}")

    for mode_key, mode_label in [
        ("mode1_attention", "【模式 1】仅注意力关键词"),
        ("mode2_llm", "【模式 2】仅 LLM 解释"),
        ("mode3_guided", "【模式 3】LLM + 注意力引导"),
    ]:
        print(f"\n{mode_label}")
        print(f"{'─'*40}")
        text = explanations.get(mode_key, "")
        # 截断显示
        if len(text) > 250:
            text = text[:250] + "..."
        print(f"  {text}")

    print(f"\n{'='*60}")
    print("💡 提示: 可将输出导入 CSV，进行人工说服力评分")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="解释方式对比（说服力实验）")
    parser.add_argument("--text", type=str, default=None, help="待检测文本")
    parser.add_argument("--sample", type=int, default=None, help="从 val.csv 抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUTS_DIR / "explain_modes.csv"),
                        help="输出 CSV 路径")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--llm-model", type=str, default="deepseek-chat",
                        help="LLM 模型名")
    parser.add_argument("--model-pt", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"))

    args = parser.parse_args()

    print("加载 BERT 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = RumorClassifier.load(args.model_pt, device=device)
    tokenizer = AutoTokenizer.from_pretrained(bert_model.model_name)
    print(f"设备: {device} | LLM: {args.llm_model}")

    if args.sample:
        # ── 批量模式 ──
        samples = load_val_samples(args.sample, args.seed)
        print(f"从 val.csv 抽取 {len(samples)} 条")

        client = LLMClient(api_key=args.api_key, model=args.llm_model)
        results = []

        for i, s in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] id={s['id']}", end="", flush=True)

            try:
                # BERT 分类 + 注意力
                clean = clean_text(s["text"])
                encoded = tokenizer(clean, padding="max_length", truncation=True,
                                    max_length=128, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attn_mask = encoded["attention_mask"].to(device)

                logits, attentions = bert_model(input_ids, attn_mask,
                                                return_attentions=True)
                probs = torch.softmax(logits, dim=1)
                pred = 1 if probs[0, 1].item() > 0.46 else 0
                conf = probs[0, pred].item()
                label_name = "谣言" if pred else "非谣言"

                importance = extract_token_importance(attentions)
                keywords = get_top_keywords(tokenizer, input_ids, importance, top_k=10)

                # 三种模式
                mode1 = generate_mode1_attention_only(tokenizer, input_ids, importance)
                mode2 = generate_mode2_llm_only(
                    s["text"], pred, label_name, conf, client, args.llm_model,
                )
                mode3 = generate_mode3_llm_guided(
                    s["text"], pred, label_name, conf, keywords,
                    client, args.llm_model,
                )

                results.append({
                    "id": s["id"],
                    "text": s["text"],
                    "true_label": s["label"],
                    "pred_label": pred,
                    "label_name": label_name,
                    "confidence": round(conf, 4),
                    "attention_keywords": "; ".join(
                        [f"{w}({s:.3f})" for w, s in keywords[:5]]
                    ),
                    "mode1_attention": mode1,
                    "mode2_llm": mode2,
                    "mode3_guided": mode3,
                })
                print(" ✅")

            except Exception as e:
                print(f" ❌ {e}")

            if (i + 1) % 10 == 0 and (i + 1) < len(samples):
                wait = 8
                print(f"  ⏳ 等待 {wait}s（API 限速）...")
                time.sleep(wait)
            elif i < len(samples) - 1:
                time.sleep(1)

        # 保存 CSV
        with open(args.output, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "id", "text", "true_label", "pred_label", "label_name",
                "confidence", "attention_keywords",
                "mode1_attention", "mode2_llm", "mode3_guided",
            ])
            writer.writeheader()
            writer.writerows(results)

        print(f"\n✅ 结果已保存: {args.output}")
        print(f"共 {len(results)} 条样本，每条包含三种解释方式")
        print(f"可用于人工评判——对每条样本的三种解释打分（1-5 分）：")
        print(f"  - 可理解性（能否看懂）")
        print(f"  - 说服力（是否让你相信判断结果）")
        print(f"  - 信息量（是否提供了有价值的分析）")

    else:
        # ── 单条模式 ──
        text = args.text or "BREAKING: Ferguson police chief says officer shot unarmed teen."
        clean = clean_text(text)
        encoded = tokenizer(clean, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt")
        input_ids = encoded["input_ids"].to(device)
        attn_mask = encoded["attention_mask"].to(device)

        logits, attentions = bert_model(input_ids, attn_mask, return_attentions=True)
        probs = torch.softmax(logits, dim=1)
        pred = 1 if probs[0, 1].item() > 0.46 else 0
        conf = probs[0, pred].item()
        label_name = "谣言" if pred else "非谣言"

        importance = extract_token_importance(attentions)
        keywords = get_top_keywords(tokenizer, input_ids, importance, top_k=10)

        client = LLMClient(api_key=args.api_key, model=args.llm_model)

        print(f"\n生成三种解释方式...")
        mode1 = generate_mode1_attention_only(tokenizer, input_ids, importance)
        print("  模式1 (注意力) ✅")
        mode2 = generate_mode2_llm_only(text, pred, label_name, conf, client, args.llm_model)
        print("  模式2 (LLM)   ✅")
        mode3 = generate_mode3_llm_guided(
            text, pred, label_name, conf, keywords, client, args.llm_model,
        )
        print("  模式3 (引导)  ✅")

        sample = {"id": "manual", "text": text, "label": -1,
                  "pred_label": pred, "event": "-"}
        explanations = {
            "label_name": label_name,
            "confidence": conf,
            "mode1_attention": mode1,
            "mode2_llm": mode2,
            "mode3_guided": mode3,
        }
        print_single_result(sample, explanations)


if __name__ == "__main__":
    main()
