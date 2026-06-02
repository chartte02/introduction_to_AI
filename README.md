# 人工智能导论大作业：可解释谣言检测

本仓库用于完成《人工智能导论》课程大作业，目标是实现一个可解释的谣言检测系统。

## 任务目标

- 输入：一条文本（推文）
- 输出1：二分类标签
  - `0` 表示非谣言
  - `1` 表示谣言
- 输出2：判断依据文本（可解释性说明）

详细要求见 [doc/guidance.md](doc/guidance.md)。

## 仓库结构

```text
.
├─ data/
│  ├─ train.csv              # 训练集（2840 条推文，7 个事件）
│  └─ val.csv                # 验证集（401 条推文）
├─ doc/
│  ├─ guidance.md             # 课程任务说明
│  ├─ template.md             # 大作业报告模板
│  └─ division.md             # 小组分工
├─ model/                  # 核心模型代码
│  ├─ preprocessing.py        # 文本预处理 pipeline（5 步清洗）
│  ├─ data.py                 # 数据加载 + 留一事件交叉验证划分
│  ├─ model.py                # BERT 分类器（twitter-roberta-base + 分类头）
│  └─ trainer.py              # 训练、评估、交叉验证、最终模型训练
├─ scripts/                # 运行入口
│  ├─ train.py                # 训练入口（支持 --cv 交叉验证 / 全量训练）
│  ├─ eval.py                 # 评估入口（在 val.csv 上报告指标）
│  ├─ predict.py              # 单条推理入口（交互式 / 命令行）
│  └─ run_cv.py               # 便捷 CV 训练脚本（含日志输出）
├─ outputs/                # 模型权重与评估产物（.gitignore 忽略）
├─ .venv/                     # Python 虚拟环境（.gitignore 忽略）
├─ requirements.txt
├─ AGENTS.md                  # AI 编码代理工作指引
└─ README.md
```

## 数据说明

字段：

- `id`：样本 ID
- `text`：输入文本（推文）
- `label`：监督标签（0=非谣言, 1=谣言）
- `event`：事件标识（0-6，对应 7 个真实世界新闻事件）

## 环境搭建

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活虚拟环境（Windows Git Bash）
source .venv/Scripts/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置 HuggingFace 镜像（国内环境）
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="*"
```

## 运行命令

### 训练

```bash
# 留一事件交叉验证（7 折，评估泛化能力）
python scripts/train.py --cv

# 在全部训练集上训练最终模型
python scripts/train.py

# 指定超参
python scripts/train.py --model bert-base-uncased --epochs 8 --lr 3e-5 --cv
```

### 评估

```bash
# 在 val.csv 上评估最终模型
python scripts/eval.py

# 评估指定模型
python scripts/eval.py --model outputs/final_model.pt --data data/val.csv
```

### 推理

```bash
# 交互式推理
python scripts/predict.py

# 单条推理
python scripts/predict.py --text "BREAKING: Ferguson police chief says..."
```

## 技术方案

### 分类模型

- 模型：`cardiffnlp/twitter-roberta-base`（1.25 亿参数，推文语料预训练）
- 分类头：`[CLS] vector → Dropout(0.1) → Linear(768, 2)`
- 预处理：HTML 解码 → URL→`[URL]` → @mention→`@USER` → `#` 去除 → 空白合并

### 训练策略

- **留一事件交叉验证**（Leave-One-Event-Out CV）：7 折，每折用 6 个事件训练、1 个 held-out 事件验证，确保模型泛化到未见过的事件
- **最终模型**：在全部 train.csv 上训练，val.csv 上评估

### 解释生成

- 使用 SJTU 校内大模型 API（`claw.sjtu.edu.cn`）生成判断依据
- 架构：BERT 分类器 → 标签 + 原文 → LLM → 判断依据文本

## 运行环境

- Python 3.8+
- 依赖：transformers, torch, pandas, scikit-learn
- 模型：首次运行自动下载 ~500MB 模型文件

## 提交清单

- `README.md`：环境与运行说明（本文件）
- `report.pdf`：大作业报告（参考 doc/template.md）
- 可复现代码与必要支持文件
