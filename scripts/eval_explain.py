#!/usr/bin/env python3
"""解释质量评估脚本。

从验证集中抽样 N 条，分别：
  1. 用 BERT 模型做分类
  2. 用 LLM 生成判断依据
  3. 输出评估表格（CSV），供人工评判可读性、相关性、准确性

用法：
    # 默认抽样 50 条，使用 deepseek-chat
    python scripts/eval_explain.py

    # 指定抽样数和 LLM 模型
    python scripts/eval_explain.py --sample 30 --llm-model deepseek-reasoner

    # 仅分类 + 注意力分析（不调 LLM，快速检查）
    python scripts/eval_explain.py --no-llm

    # 输出到指定文件
    python scripts/eval_explain.py --output outputs/explanations.csv
"""

import argparse
import csv
import io
import os
import sys
import random
from pathlib import Path

# Windows 控制台 UTF-8
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("no_proxy", "*")

import torch
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

from model.preprocessing import clean_text
from model.model import RumorClassifier
from model.data import RumorDataset
from model.attention_viz import (
    extract_attention,
    extract_token_importance,
    get_top_keywords,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_val_data(data_path: str) -> list:
    """加载验证集数据。

    Returns:
        [{"id": ..., "text": ..., "label": ..., "event": ...}, ...]
    """
    samples = []
    with open(data_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 4:
                continue
            samples.append({
                "id": row[0],
                "text": row[1],
                "label": int(row[2]),
                "event": row[3],
            })
    return samples


def classify_samples(
    samples: list,
    model: RumorClassifier,
    tokenizer,
    device: str,
    max_length: int = 128,
    with_attention: bool = False,
) -> list:
    """对样本列表执行分类推理。

    Args:
        samples: 样本列表。
        model: 分类模型。
        tokenizer: tokenizer。
        device: 设备。
        max_length: 最大 token 长度。
        with_attention: 是否提取注意力权重。

    Returns:
        增强后的样本列表，增加预测结果字段。
    """
    results = []

    for s in samples:
        clean = clean_text(s["text"])
        encoded = tokenizer(
            clean,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        model.eval()
        with torch.no_grad():
            if with_attention:
                logits, attentions = model(
                    input_ids, attention_mask, return_attentions=True
                )
            else:
                logits = model(input_ids, attention_mask)
                attentions = None

        probs = torch.softmax(logits, dim=1)
        pred = torch.argmax(logits, dim=1).item()
        confidence = probs[0, pred].item()

        item = {
            **s,
            "pred_label": pred,
            "pred_name": "谣言" if pred == 1 else "非谣言",
            "confidence": confidence,
            "correct": pred == s["label"],
        }

        if with_attention and attentions is not None:
            importance = extract_token_importance(attentions)
            top_words = get_top_keywords(
                tokenizer, input_ids, importance, top_k=5
            )
            item["top_keywords"] = "; ".join(
                [f"{w}({s:.3f})" for w, s in top_words]
            )

        results.append(item)

    return results


def generate_explanations(
    samples: list,
    llm_model: str = "deepseek-chat",
    api_key: str = None,
    base_url: str = None,
    max_workers: int = 1,
) -> list:
    """为样本列表生成 LLM 解释。

    Args:
        samples: 已分类的样本列表。
        llm_model: LLM 模型名。
        api_key: API key。
        base_url: API 基础地址。
        max_workers: 并发数（预留，当前为串行）。

    Returns:
        增加 explanation 字段的样本列表。
    """
    from model.llm_client import LLMClient
    from model.prompts import build_explanation_prompt

    client = LLMClient(
        api_key=api_key,
        base_url=base_url,
        model=llm_model,
    )

    results = []
    total = len(samples)

    for i, s in enumerate(samples):
        print(f"  [{i+1}/{total}] 生成解释中... (id={s['id']})", end="", flush=True)

        try:
            messages = build_explanation_prompt(
                s["text"], s["pred_label"], s["pred_name"], s["confidence"]
            )
            explanation = client.chat(messages, model=llm_model)
            s["explanation"] = explanation
            print(" ✅")
        except Exception as e:
            s["explanation"] = f"（生成失败: {e}）"
            print(" ❌")

        results.append(s)

    return results


def save_results(samples: list, output_path: str):
    """将评估结果保存为 CSV。"""
    fieldnames = [
        "id", "text", "label", "event",
        "pred_label", "pred_name", "confidence", "correct",
        "top_keywords", "explanation",
    ]

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            row = {k: s.get(k, "") for k in fieldnames}
            writer.writerow(row)

    print(f"\n结果已保存: {output_path}")


def print_summary(samples: list):
    """打印评估汇总。"""
    total = len(samples)
    correct = sum(1 for s in samples if s.get("correct"))
    accuracy = correct / total if total > 0 else 0

    print(f"\n{'='*50}")
    print(f"📊 评估汇总")
    print(f"{'='*50}")
    print(f"  样本总数:     {total}")
    print(f"  分类正确:     {correct}/{total}")
    print(f"  分类准确率:   {accuracy:.2%}")

    # 按标签统计
    rumor_samples = [s for s in samples if s["label"] == 1]
    non_rumor_samples = [s for s in samples if s["label"] == 0]

    if rumor_samples:
        rumor_correct = sum(1 for s in rumor_samples if s.get("correct"))
        print(f"  谣言召回率:   {rumor_correct}/{len(rumor_samples)} "
              f"({rumor_correct/len(rumor_samples):.2%})")

    if non_rumor_samples:
        non_correct = sum(1 for s in non_rumor_samples if s.get("correct"))
        print(f"  非谣言准确率: {non_correct}/{len(non_rumor_samples)} "
              f"({non_correct/len(non_rumor_samples):.2%})")

    # 解释生成统计
    explanations = [s.get("explanation", "") for s in samples]
    generated = sum(1 for e in explanations if e and not e.startswith("（"))
    failed = sum(1 for e in explanations if e.startswith("（"))
    print(f"  解释生成成功: {generated}/{total}")
    if failed:
        print(f"  解释生成失败: {failed}/{total}")

    print(f"{'='*50}")

    # 显示一些示例
    print(f"\n📝 示例展示（前 3 条）:")
    print("-" * 50)
    for s in samples[:3]:
        print(f"  [{'✅' if s.get('correct') else '❌'}] "
              f"真实={s['label']} → 预测={s['pred_name']} "
              f"(置信度: {s.get('confidence', 0):.2%})")
        if s.get("explanation") and not s["explanation"].startswith("（"):
            print(f"      解释: {s['explanation'][:150]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="解释质量评估")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"),
        help="模型文件路径",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=str(PROJECT_ROOT / "data" / "val.csv"),
        help="评估数据集 CSV",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=50,
        help="抽样数量（默认 50）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "explanations.csv"),
        help="输出文件路径",
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="仅分类，不调 LLM（快速模式）",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-chat",
        help="LLM 模型名",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="SJTU API key（默认从环境变量 SJTU_API_KEY 读取）",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="同时提取注意力关键词",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=128,
        help="最大 token 长度",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 检查模型
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行: python scripts/train.py")
        sys.exit(1)

    # 加载模型
    print("加载模型中...")
    model = RumorClassifier.load(str(model_path), device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    print(f"模型: {model.model_name}")

    # 加载数据
    print(f"加载数据: {args.data}")
    all_samples = load_val_data(args.data)
    print(f"验证集总样本数: {len(all_samples)}")

    # 抽样
    random.seed(args.seed)
    sample_size = min(args.sample, len(all_samples))
    samples = random.sample(all_samples, sample_size)
    # 尽量保持标签平衡
    rumor = [s for s in samples if s["label"] == 1]
    non_rumor = [s for s in samples if s["label"] == 0]
    print(f"抽样 {sample_size} 条（谣言: {len(rumor)}, 非谣言: {len(non_rumor)}）")

    # 分类
    print("\n执行分类推理...")
    samples = classify_samples(
        samples, model, tokenizer, device,
        args.max_length, with_attention=args.attention,
    )

    # LLM 解释
    if not args.no_llm:
        print(f"\n生成 LLM 解释（模型: {args.llm_model}）...")
        print(f"注意: 每分钟限 10 次请求，将自动限速...")
        import time

        from model.llm_client import LLMClient
        from model.prompts import build_explanation_prompt

        client = LLMClient(
            api_key=args.api_key,
            model=args.llm_model,
        )

        total = len(samples)
        for i, s in enumerate(samples):
            print(f"  [{i+1}/{total}] 生成解释... (id={s['id']})", end="", flush=True)
            try:
                messages = build_explanation_prompt(
                    s["text"], s["pred_label"], s["pred_name"], s["confidence"]
                )
                s["explanation"] = client.chat(messages, model=args.llm_model)
                print(" ✅")
            except Exception as e:
                s["explanation"] = f"（生成失败: {e}）"
                print(" ❌")

            # 限速：每分钟最多 10 次请求 → 每次间隔至少 6 秒
            if (i + 1) % 10 == 0 and (i + 1) < total:
                wait = 8
                print(f"  ⏳ 达到速率限制，等待 {wait} 秒...")
                time.sleep(wait)
            elif i < total - 1:
                time.sleep(1)  # 小幅延迟避免突发

    # 保存
    save_results(samples, args.output)

    # 汇总
    print_summary(samples)


if __name__ == "__main__":
    main()
