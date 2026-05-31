# 人工智能导论大作业：可解释谣言检测

本仓库用于完成《人工智能导论》课程大作业，目标是实现一个可解释的谣言检测系统。

## 任务目标

- 输入：一条文本（推文）
- 输出1：二分类标签
  - `0` 表示非谣言
  - `1` 表示谣言
- 输出2：判断依据文本（可解释性说明）

详细要求见 [doc/guidance.md](doc/guidance.md)。

## 当前仓库结构

```text
.
├─ data/
│  ├─ train.csv
│  └─ val.csv
├─ doc/
│  ├─ guidance.md
│  └─ template.md
└─ AGENTS.md
```

## 数据说明

当前数据包含以下字段：

- `id`：样本 ID
- `text`：输入文本
- `label`：监督标签（0/1）
- `event`：事件标识

## 开发建议（初始化阶段）

建议后续补充如下目录：

- `src/`：模型、训练、推理、解释生成逻辑
- `scripts/`：训练与评估入口脚本
- `outputs/`：预测结果与评估产物（已在 `.gitignore` 忽略）

## 运行环境建议

- Python 3.10+
- 常见依赖：pandas, scikit-learn, torch, transformers（按方案选择）

## 后续里程碑

- 完成最小可运行基线（训练 + 在 `val.csv` 评估）
- 增加解释生成模块（规则、检索增强或大模型）
- 在报告模板中补充结果分析与可解释性分析

## 提交要求提醒

最终提交请确保仓库至少包含：

- `README.md`（环境与运行说明）
- `report.pdf`（参考 [doc/template.md](doc/template.md)）
- 可复现代码与必要支持文件
