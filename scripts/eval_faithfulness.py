#!/usr/bin/env python3
"""解释忠实度评估脚本。

评估 LLM 生成的判断依据是否真正反映了 BERT 模型的推理过程。
方法：对比 LLM 解释中提到的关键词 与 BERT 注意力权重最高的词汇，
计算重叠率（faithfulness score）。

指标说明：
  - Overlap@5: LLM 解释是否覆盖了 Top-5 注意力关键词
  - Overlap@10: Top-10 覆盖率
  - Precision: LLM 解释中提到的词有多少是高注意力词
  - Faithfulness Score: 综合忠实度（0~1）

用法：
    python scripts/eval_faithfulness.py --sample 20
    python scripts/eval_faithfulness.py --sample 30 --output outputs/faithfulness.csv
"""

import argparse
import csv
import io
import json
import os
import random
import re
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
from model.prompts import build_explanation_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


def load_val_samples(n: int, seed: int = 42) -> list:
    """从 val.csv 随机抽取 n 条样本。"""
    random.seed(seed)
    samples = []
    with open(PROJECT_ROOT / "data" / "val.csv", "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                samples.append({
                    "id": row[0],
                    "text": row[1],
                    "label": int(row[2]),
                    "event": row[3],
                })
    return random.sample(samples, min(n, len(samples)))


def normalize_words(keywords: list) -> set:
    """将关键词列表规范化为小写词集合（去除标点）。"""
    result = set()
    for word, score in keywords:
        clean = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]", "", word.lower())
        if clean and len(clean) >= 2:
            result.add(clean)
    return result


def extract_explanation_keywords(explanation: str, min_len: int = 2) -> set:
    """从 LLM 解释文本中提取英文关键词（去停用词）。"""
    # 常见停用词
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "in", "on", "at", "to", "for", "of", "by", "with", "from",
        "and", "or", "but", "not", "no", "that", "this", "it", "its",
        "has", "have", "had", "do", "does", "did", "will", "would",
        "can", "could", "may", "might", "should", "such", "as", "if",
        "so", "than", "then", "also", "very", "too", "just", "only",
        "about", "into", "over", "after", "before", "between", "through",
        "during", "because", "since", "while", "these", "those", "they",
        "them", "their", "his", "her", "he", "she", "we", "you",
    }

    words = set()
    # 匹配英文单词
    for match in re.finditer(r"[a-zA-Z]{2,}", explanation):
        word = match.group().lower()
        if word not in stopwords:
            words.add(word)
    return words


def compute_faithfulness(
    attention_keywords: list,
    explanation: str,
    top_k: int = 10,
) -> dict:
    """计算单条样本的忠实度指标。

    Args:
        attention_keywords: BERT 注意力关键词 [(word, score), ...]。
        explanation: LLM 生成的判断依据文本。
        top_k: 取 top-k 注意力词比较。

    Returns:
        {"overlap_k": int, "overlap_ratio": float, "total_attn_words": int,
         "total_expl_words": int, "expl_in_attn": int, "precision": float}
    """
    attn_words = normalize_words(attention_keywords[:top_k])
    expl_words = extract_explanation_keywords(explanation)

    overlap = attn_words & expl_words
    expl_in_attn = sum(1 for w in expl_words if w in attn_words)

    return {
        "overlap_count": len(overlap),
        "overlap_ratio": len(overlap) / max(len(attn_words), 1),
        "total_attn_words_topk": len(attn_words),
        "total_expl_words": len(expl_words),
        "expl_in_attn": expl_in_attn,
        "precision": expl_in_attn / max(len(expl_words), 1),
        "matched_words": sorted(overlap),
    }


def compute_comprehensive_score(metrics_list: list) -> dict:
    """计算一批样本的综合忠实度得分。

    综合得分 = (overlap_ratio@10 均值 + precision 均值) / 2
    """
    if not metrics_list:
        return {"comprehensive_faithfulness": 0.0}

    avg_overlap = sum(m["overlap_ratio"] for m in metrics_list) / len(metrics_list)
    avg_precision = sum(m["precision"] for m in metrics_list) / len(metrics_list)
    comprehensive = (avg_overlap + avg_precision) / 2

    return {
        "n_samples": len(metrics_list),
        "mean_overlap_ratio": round(avg_overlap, 4),
        "mean_precision": round(avg_precision, 4),
        "comprehensive_faithfulness": round(comprehensive, 4),
    }


def print_report(aggregate: dict, per_sample: list):
    """打印忠实度评估报告。"""
    print(f"\n{'='*60}")
    print(f"📊 解释忠实度评估报告")
    print(f"{'='*60}")
    print(f"  样本数:     {aggregate['n_samples']}")
    print(f"  平均重叠率: {aggregate['mean_overlap_ratio']:.2%}")
    print(f"                 (LLM 解释覆盖了 top-10 注意力词的 {aggregate['mean_overlap_ratio']:.0%})")
    print(f"  平均精确率: {aggregate['mean_precision']:.2%}")
    print(f"                 (LLM 解释中的词有 {aggregate['mean_precision']:.0%} 是高注意力词)")
    print(f"{'-'*40}")
    print(f"  🎯 综合忠实度: {aggregate['comprehensive_faithfulness']:.2%}")
    print(f"{'='*60}")

    # 解释分级
    score = aggregate["comprehensive_faithfulness"]
    if score >= 0.6:
        grade = "优秀 — LLM 解释与模型推理高度一致"
    elif score >= 0.4:
        grade = "良好 — LLM 解释部分反映了模型判断逻辑"
    elif score >= 0.2:
        grade = "一般 — LLM 解释与模型注意力仅有弱关联"
    else:
        grade = "需改进 — LLM 解释与模型内部推理脱节"

    print(f"  评级: {grade}")
    print(f"{'='*60}")

    # 展示几个案例
    print(f"\n📝 典型案例展示（前 3 条）:")
    for i, s in enumerate(per_sample[:3]):
        print(f"\n  [{i+1}] id={s['id']}")
        print(f"  原文: {s['text'][:80]}...")
        print(f"  注意力关键词: {s.get('top_attn_words', '')}")
        print(f"  覆盖词: {', '.join(s.get('matched_words', []))}")
        print(f"  重叠率: {s.get('overlap_ratio', 0):.2%}")


def main():
    parser = argparse.ArgumentParser(description="解释忠实度评估")
    parser.add_argument("--sample", type=int, default=20, help="抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str,
                        default=str(OUTPUTS_DIR / "faithfulness.json"),
                        help="输出 JSON 路径")
    parser.add_argument("--output-csv", type=str,
                        default=str(OUTPUTS_DIR / "faithfulness.csv"),
                        help="输出 CSV 路径")
    parser.add_argument("--api-key", type=str, default=None,
                        help="SJTU API key")
    parser.add_argument("--llm-model", type=str, default="deepseek-chat",
                        help="LLM 模型名")
    parser.add_argument("--top-k", type=int, default=10,
                        help="取 top-k 注意力词做对比")
    parser.add_argument("--model-pt", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"),
                        help="BERT 模型路径")

    args = parser.parse_args()

    print("加载 BERT 模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = RumorClassifier.load(args.model_pt, device=device)
    tokenizer = AutoTokenizer.from_pretrained(bert_model.model_name)
    print(f"设备: {device} | 模型: {bert_model.model_name}")

    # 加载样本
    samples = load_val_samples(args.sample, args.seed)
    print(f"从 val.csv 抽取 {len(samples)} 条样本")

    # 初始化 LLM 客户端
    print(f"LLM 模型: {args.llm_model}")
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

            logits, attentions = bert_model(
                input_ids, attn_mask, return_attentions=True
            )
            probs = torch.softmax(logits, dim=1)
            pred = 1 if probs[0, 1].item() > 0.46 else 0
            conf = probs[0, pred].item()
            label_name = "谣言" if pred == 1 else "非谣言"

            # 注意力关键词
            importance = extract_token_importance(attentions)
            attn_keywords = get_top_keywords(tokenizer, input_ids, importance, top_k=args.top_k)

            # LLM 解释
            messages = build_explanation_prompt(s["text"], pred, label_name, conf)
            explanation = client.chat(messages, model=args.llm_model)

            # 计算忠实度
            faith = compute_faithfulness(attn_keywords, explanation, top_k=args.top_k)
            record = {
                "id": s["id"],
                "text": s["text"],
                "true_label": s["label"],
                "pred_label": pred,
                "confidence": round(conf, 4),
                "top_attn_words": "; ".join([f"{w}({s:.3f})" for w, s in attn_keywords]),
                "explanation": explanation,
                **faith,
            }
            results.append(record)
            print(f" ✅ 重叠={faith['overlap_ratio']:.2%} 匹配词={faith['matched_words']}")

        except Exception as e:
            print(f" ❌ {e}")

        # 限速
        if (i + 1) % 10 == 0 and (i + 1) < len(samples):
            wait = 8
            print(f"  ⏳ 等待 {wait} 秒（API 限速）...")
            time.sleep(wait)
        elif i < len(samples) - 1:
            time.sleep(1)

    # 汇总
    aggregate = compute_comprehensive_score(results)
    print_report(aggregate, results)

    # 保存 JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "n_samples": args.sample,
                "llm_model": args.llm_model,
                "top_k": args.top_k,
            },
            "aggregate": aggregate,
            "per_sample": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\nJSON 已保存: {args.output}")

    # 保存 CSV
    csv_path = args.output_csv or args.output.replace(".json", ".csv")
    fieldnames = [
        "id", "text", "true_label", "pred_label", "confidence",
        "top_attn_words", "overlap_count", "overlap_ratio",
        "expl_in_attn", "precision", "matched_words",
        "explanation",
    ]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)
    print(f"CSV 已保存: {csv_path}")


if __name__ == "__main__":
    main()
