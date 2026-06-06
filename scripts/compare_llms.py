#!/usr/bin/env python3
"""多模型解释对比脚本。

用同一段文本，调用多个 LLM 模型分别生成判断依据，
横向对比各模型的解释质量、风格差异、耗时和 token 用量。

用法：
    # 对比全部可用模型
    python scripts/compare_llms.py --text "BREAKING: Ferguson police chief says..."

    # 对比指定模型
    python scripts/compare_llms.py --text "..." --models deepseek-chat deepseek-reasoner glm

    # 从 val.csv 随机抽样多条对比
    python scripts/compare_llms.py --sample 5 --output outputs/llm_comparison.csv
"""

import argparse
import csv
import io
import json
import os
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
from model.llm_client import LLMClient
from model.prompts import build_explanation_prompt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# 可用模型列表
AVAILABLE_MODELS = [
    {
        "name": "deepseek-chat",
        "label": "DeepSeek V3.2 (常规)",
        "desc": "通用文本处理，均衡型",
    },
    {
        "name": "deepseek-reasoner",
        "label": "DeepSeek V3.2 (思考)",
        "desc": "深度推理，适合复杂逻辑分析",
    },
    {
        "name": "glm",
        "label": "GLM-5.1",
        "desc": "754B 参数，代码与长程任务",
    },
    {
        "name": "qwen",
        "label": "Qwen3.5-27B",
        "desc": "多模态背景，视觉与文本理解",
    },
    {
        "name": "minimax",
        "label": "MiniMax-M2.7",
        "desc": "230B 参数，智能体任务优化",
    },
]


def load_val_samples(n: int, seed: int = 42) -> list:
    """从 val.csv 中随机抽取 n 条样本。"""
    import random
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


def run_single_comparison(
    text: str,
    label: int,
    label_name: str,
    confidence: float,
    models: list,
    api_key: str = None,
) -> list:
    """用多个 LLM 对同一文本生成解释，返回对比结果列表。"""
    results = []

    for model_info in models:
        model_name = model_info["name"]
        print(f"  [{model_info['label']}] 请求中...", end=" ", flush=True)

        try:
            client = LLMClient(
                api_key=api_key,
                model=model_name,
                temperature=0.3,
            )

            t0 = time.time()
            explanation = client.generate_explanation(
                text=text,
                label=label,
                label_name=label_name,
                confidence=confidence,
                model=model_name,
            )
            elapsed = time.time() - t0

            results.append({
                "model": model_name,
                "label": model_info["label"],
                "description": model_info["desc"],
                "explanation": explanation,
                "latency_sec": round(elapsed, 2),
                "length_chars": len(explanation),
                "success": True,
            })
            print(f"✅ {elapsed:.1f}s, {len(explanation)} 字")

        except Exception as e:
            results.append({
                "model": model_name,
                "label": model_info["label"],
                "description": model_info["desc"],
                "explanation": f"失败: {e}",
                "latency_sec": 0,
                "length_chars": 0,
                "success": False,
            })
            print(f"❌ {e}")

        # 限速：每分钟 10 次请求
        time.sleep(1)

    return results


def print_comparison_table(text: str, results: list):
    """打印对比报告。"""
    print(f"\n{'='*70}")
    print(f"📝 原文: {text[:100]}{'...' if len(text)>100 else ''}")
    print(f"{'='*70}\n")

    for i, r in enumerate(results):
        icon = "✅" if r["success"] else "❌"
        print(f"--- 模型 {i+1}: {r['label']} {icon} ---")
        print(f"  描述: {r['description']}")
        if r["success"]:
            print(f"  延迟: {r['latency_sec']}s | 字数: {r['length_chars']}")
        print(f"  解释: {r['explanation'][:200]}{'...' if len(r['explanation'])>200 else ''}")
        print()

    # 汇总表格
    print(f"{'='*70}")
    print(f"{'模型':<22} {'成功':>3} {'延迟(s)':>8} {'字数':>6}")
    print(f"{'-'*42}")
    for r in results:
        status = "✅" if r["success"] else "❌"
        print(f"{r['label']:<22} {status:>3} {r['latency_sec']:>8.1f} {r['length_chars']:>6}")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description="多模型解释对比")
    parser.add_argument("--text", type=str, default=None, help="待检测文本")
    parser.add_argument("--sample", type=int, default=None, help="从 val.csv 抽样数量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--models", nargs="+",
        default=["deepseek-chat"],
        help="对比的模型名（默认 deepseek-chat，--all 用全部）",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="使用全部 5 个模型对比",
    )
    parser.add_argument("--output", type=str, default=None, help="输出 JSON 路径")
    parser.add_argument("--api-key", type=str, default=None, help="API key")
    parser.add_argument("--model-pt", type=str,
                        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"),
                        help="BERT 模型路径")

    args = parser.parse_args()

    # 选择模型：--all 覆盖 --models
    if args.all:
        models = AVAILABLE_MODELS
    else:
        models = [m for m in AVAILABLE_MODELS if m["name"] in args.models]

    if not models:
        print("错误: 未选中任何模型")
        sys.exit(1)

    print(f"对比模型 ({len(models)}):")
    for m in models:
        print(f"  - {m['label']} ({m['name']})")
    print(f"\n⚠️ 注意: API 限速 10次/分钟，5 个模型约需 30 秒")

    # 加载 BERT
    print("\n加载 BERT 分类模型...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bert_model = RumorClassifier.load(args.model_pt, device=device)
    tokenizer = AutoTokenizer.from_pretrained(bert_model.model_name)
    print(f"设备: {device} | 模型: {bert_model.model_name}")

    if args.sample:
        # 批量模式
        samples = load_val_samples(args.sample, args.seed)
        print(f"\n从 val.csv 抽 {len(samples)} 条样本进行批量对比...")
        all_results = []

        for i, s in enumerate(samples):
            print(f"\n[{i+1}/{len(samples)}] id={s['id']}")
            clean = clean_text(s["text"])
            encoded = tokenizer(clean, padding="max_length", truncation=True,
                                max_length=128, return_tensors="pt")
            with torch.no_grad():
                logits = bert_model(encoded["input_ids"].to(device),
                                    encoded["attention_mask"].to(device))
            probs = torch.softmax(logits, dim=1)
            pred = 1 if probs[0, 1].item() > 0.46 else 0
            conf = probs[0, pred].item()
            label_name = "谣言" if pred == 1 else "非谣言"

            results = run_single_comparison(
                text=s["text"], label=pred, label_name=label_name,
                confidence=conf, models=models, api_key=args.api_key,
            )
            for r in results:
                r["sample_id"] = s["id"]
                r["true_label"] = s["label"]
                r["pred_label"] = pred
            all_results.extend(results)

        # 保存
        out_path = args.output or str(OUTPUTS_DIR / "llm_comparison.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {out_path}")

    else:
        # 单条模式
        text = args.text or "BREAKING: Ferguson police chief says officer shot unarmed teen."
        clean = clean_text(text)
        encoded = tokenizer(clean, padding="max_length", truncation=True,
                            max_length=128, return_tensors="pt")
        with torch.no_grad():
            logits = bert_model(encoded["input_ids"].to(device),
                                encoded["attention_mask"].to(device))
        probs = torch.softmax(logits, dim=1)
        pred = 1 if probs[0, 1].item() > 0.46 else 0
        conf = probs[0, pred].item()
        label_name = "谣言" if pred == 1 else "非谣言"

        print(f"\n🔍 检测结果: {label_name} (置信度: {conf:.2%})\n")

        results = run_single_comparison(
            text=text, label=pred, label_name=label_name,
            confidence=conf, models=models, api_key=args.api_key,
        )
        print_comparison_table(text, results)


if __name__ == "__main__":
    main()
