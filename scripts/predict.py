#!/usr/bin/env python3
"""推理入口脚本（端到端：分类 + 可解释性）。

用法：
    # 交互式推理（含分类 + LLM 解释）
    python scripts/predict.py

    # 单条推理（含解释）
    python scripts/predict.py --text "BREAKING: Ferguson police chief says..."

    # 仅分类（不调 LLM，离线可用）
    python scripts/predict.py --text "..." --no-explain

    # 带注意力高亮
    python scripts/predict.py --text "..." --attention
"""

import argparse
import io
import os
import sys
from pathlib import Path

# Windows 控制台 UTF-8 输出
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# 国内环境 HuggingFace 镜像配置
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
    generate_attention_highlight,
    visualize_attention_heatmap,
)
from model.trainer import load_threshold

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def predict_single(
    text: str,
    model: RumorClassifier,
    tokenizer,
    device: str,
    max_length: int = 128,
    return_attentions: bool = False,
) -> dict:
    """对单条推文进行推理。

    判定规则：P(谣言) > τ，其中 τ 由 outputs/threshold.json 提供；
    未校准时回退到 0.5，等价于原 argmax 行为。

    Args:
        text: 原始推文文本。
        model: 已加载的分类模型。
        tokenizer: HuggingFace tokenizer。
        device: 设备。
        max_length: token 最大长度。
        return_attentions: 是否返回注意力权重。

    Returns:
        包含 label, label_name, confidence, threshold 的字典。
        如果 return_attentions=True，还包含 attentions, input_ids, attention_mask。
    """
    clean = clean_text(text)
    encoded = tokenizer(
        clean,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    threshold = load_threshold()
    model.eval()

    # 单次前向传播（需要注意力时用 return_attentions=True）
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, return_attentions=return_attentions)

    if return_attentions:
        logits, attentions = outputs
    else:
        logits = outputs
        attentions = None

    probs = torch.softmax(logits, dim=1)
    pred = 1 if probs[0, 1].item() > threshold else 0
    confidence = probs[0, pred].item()

    result = {
        "label": pred,
        "label_name": "谣言" if pred == 1 else "非谣言",
        "confidence": confidence,
        "threshold": threshold,
    }

    if return_attentions:
        result["attentions"] = attentions
        result["input_ids"] = input_ids
        result["attention_mask"] = attention_mask

    return result


def explain_with_llm(
    text: str,
    label: int,
    label_name: str,
    confidence: float,
    api_key: str = None,
    model_name: str = "deepseek-chat",
    base_url: str = None,
) -> str:
    """调用 LLM 生成判断依据。

    Args:
        text: 原始推文。
        label: 分类标签。
        label_name: 标签名称。
        confidence: 置信度。
        api_key: SJTU API key。未提供时从环境变量获取。
        model_name: LLM 模型名。
        base_url: API 基础地址。

    Returns:
        判断依据文本。
    """
    try:
        from model.llm_client import LLMClient
        from model.prompts import build_explanation_prompt

        client = LLMClient(
            api_key=api_key,
            base_url=base_url,
            model=model_name,
        )

        messages = build_explanation_prompt(text, label, label_name, confidence)
        explanation = client.chat(messages, model=model_name)
        return explanation

    except Exception as e:
        return f"（LLM 解释生成失败: {e}）"


def main():
    parser = argparse.ArgumentParser(description="谣言检测端到端推理（分类 + 可解释性）")
    parser.add_argument(
        "--model",
        type=str,
        default=str(PROJECT_ROOT / "outputs" / "final_model.pt"),
        help="模型文件路径",
    )
    parser.add_argument("--text", type=str, default=None, help="待检测的推文文本")
    parser.add_argument("--max_length", type=int, default=128, help="最大 token 长度")
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="设备（cuda/cpu），默认自动检测",
    )
    parser.add_argument(
        "--no-explain",
        action="store_true",
        help="不生成 LLM 解释（仅分类，离线可用）",
    )
    parser.add_argument(
        "--attention",
        action="store_true",
        help="显示注意力高亮",
    )
    parser.add_argument(
        "--heatmap",
        type=str,
        default=None,
        help="保存注意力热力图路径（如 outputs/heatmap.png）",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="deepseek-chat",
        help="LLM 模型名（默认 deepseek-chat）",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="SJTU LLM API key（默认从环境变量 SJTU_API_KEY 读取）",
    )

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    # 检查模型文件是否存在
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"错误: 模型文件不存在: {model_path}")
        print("请先运行训练脚本生成模型：")
        print("  python scripts/train.py")
        sys.exit(1)

    # 加载模型
    model = RumorClassifier.load(str(model_path), device=device)
    tokenizer = AutoTokenizer.from_pretrained(model.model_name)
    print(f"模型: {model.model_name}")

    def run_inference(text: str):
        """执行单次推理并输出结果。"""
        # 分类推理
        result = predict_single(
            text, model, tokenizer, device,
            args.max_length,
            return_attentions=(args.attention or args.heatmap is not None),
        )

        print(f"\n{'='*50}")
        print(f"📝 输入文本: {text[:120]}{'...' if len(text) > 120 else ''}")
        print(f"{'='*50}")
        print(f"🔍 检测结果: {result['label_name']}")
        print(f"📊 置信度:   {result['confidence']:.4f} ({result['confidence']:.2%})")

        # 注意力高亮
        if args.attention and result.get("attentions"):
            importance = extract_token_importance(result["attentions"])
            highlight = generate_attention_highlight(
                tokenizer, result["input_ids"], importance
            )
            print(f"\n🎯 注意力高亮:")
            print(f"   {highlight}")
            print(f"   （**加粗**=高关注  *斜体*=中等关注）")

            # Top-5 关键词
            top_words = get_top_keywords(
                tokenizer, result["input_ids"], importance, top_k=5
            )
            words_str = ", ".join([f"「{w}」({s:.3f})" for w, s in top_words])
            print(f"\n🔑 关键决策词: {words_str}")

        # 注意力热力图
        if args.heatmap and result.get("attentions"):
            visualize_attention_heatmap(
                result["attentions"],
                tokenizer,
                result["input_ids"],
                save_path=str(PROJECT_ROOT / args.heatmap),
                title=f"注意力热力图 — {result['label_name']} ({result['confidence']:.1%})",
            )

        # LLM 解释
        if not args.no_explain:
            print(f"\n💡 判断依据:")
            explanation = explain_with_llm(
                text=text,
                label=result["label"],
                label_name=result["label_name"],
                confidence=result["confidence"],
                api_key=args.api_key,
                model_name=args.llm_model,
            )
            print(f"   {explanation}")

        print(f"{'='*50}")

    if args.text:
        # 单条推理模式
        run_inference(args.text)
    else:
        # 交互式模式
        print("\n交互式推理模式（输入 quit 退出）")
        print("-" * 50)
        while True:
            try:
                text = input("\n请输入推文文本: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if text.lower() == "quit":
                break
            if not text:
                continue

            run_inference(text)


if __name__ == "__main__":
    main()
