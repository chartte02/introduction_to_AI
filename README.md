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
│  ├─ division.md             # 小组分工
├─ model/                     # 核心模型代码
│  ├─ preprocessing.py        # 文本预处理 pipeline（5 步清洗）
│  ├─ data.py                 # 数据加载 + 留一事件交叉验证划分
│  ├─ model.py                # RumorClassifier（RoBERTa + 分类头）
│  ├─ trainer.py              # 训练/评估/CV/最终模型（类别加权+早停）
│  ├─ llm_client.py           # SJTU LLM API 客户端（OpenAI 兼容）
│  ├─ prompts.py              # 可解释性 Prompt 模板
│  └─ attention_viz.py        # 注意力提取/高亮/热力图/关键词排序
├─ scripts/                   # 运行入口
│  ├─ train.py                # 训练入口（--cv 交叉验证 / 全量训练）
│  ├─ eval.py                 # 评估入口（val.csv 上报告各类指标）
│  ├─ predict.py              # 单条推理入口（交互式 / 命令行）
│  ├─ calibrate.py            # 阈值校准脚本
│  ├─ compare_backbones.py    # 多 backbone 对比评估
│  ├─ vote.py                 # 8 模型 soft voting 评估
│  ├─ eval_explain.py         # 解释质量评估（抽样+LLM+CSV）
│  └─ diagnose.py             # 环境诊断脚本
├─ src/                       # 数据分析模块
│  ├─ eda.py                  # 数据探索性分析
│  └─ vocabulary_analysis.py  # 词汇分析（关键词/词云/特殊字符）
├─ outputs/                   # 模型权重与评估产物（.gitignore 忽略）
│  ├─ final_model.pt          # 最终模型权重
│  ├─ fold_1~7_best.pt        # 交叉验证各折最佳模型
│  ├─ tokenizer/              # tokenizer 文件
│  ├─ figures/                # 可视化图表
│  └─ tables/                 # 分析数据表
├─ .venv/                     # Python 虚拟环境（.gitignore 忽略）
├─ requirements.txt
├─ AGENTS.md
└─ README.md
```

## 数据说明

字段：

- `id`：样本 ID
- `text`：输入文本（推文）
- `label`：监督标签（0=非谣言, 1=谣言）
- `event`：事件标识（0-6，对应 7 个真实世界新闻事件）

### 事件分布


| event | 真实事件                     | 样本数 | 谣言比例  |
| ----- | ------------------------ | --- | ----- |
| 0     | Gurlitt 纳粹艺术品争议          | 66  | 19.7% |
| 1     | Ferguson / Mike Brown 枪击 | 799 | 24.8% |
| 2     | Essien 埃博拉谣言             | 9   | 100%  |
| 3     | Prince 多伦多惊喜演唱会          | 162 | 98.8% |
| 4     | Germanwings 4U9525 空难    | 327 | 50.8% |
| 5     | 悉尼 Lindt 咖啡馆人质事件         | 854 | 42.7% |
| 6     | 渥太华议会枪击事件                | 623 | 52.8% |


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
# 在全部训练集上训练最终模型（含早停 + 阈值校准）
python scripts/train.py

# 指定超参
python scripts/train.py --epochs 20 --patience 3 --batch_size 32
```

### 评估

```bash
# 在 val.csv 上评估
python scripts/eval.py

# 阈值校准
python scripts/calibrate.py

# 多 backbone 对比
python scripts/compare_backbones.py
```

### 推理

```bash
# 交互式推理
python scripts/predict.py

# 单条推理（含 LLM 解释）
python scripts/predict.py --text "BREAKING: Ferguson police chief says..."
```

### 数据分析

```bash
python src/eda.py                     # 数据探索性分析
python src/vocabulary_analysis.py     # 词汇与语言特征分析
```

### 可解释性评估

```bash
python scripts/eval_explain.py --sample 50 --attention
```

## 技术方案

### 分类模型

- 模型：`cardiffnlp/twitter-roberta-base`（1.25 亿参数，推文语料预训练）
- 分类头：`[CLS] vector → Dropout(0.1) → Linear(768, 2)`
- 预处理：HTML 解码 → URL→`[URL]` → @mention→`@USER` → `#` 去除 → 空白合并

### 训练策略

- **留一事件交叉验证**（Leave-One-Event-Out CV）：7 折，每折用 6 个事件训练、1 个 held-out 事件验证，确保模型泛化到未见过的事件
- **最终模型**：在全部 train.csv 上训练，分层划分 10% 作为 dev 集用于早停
- **类别加权损失**：按类别频率的逆比例计算权重，缓解谣言/非谣言样本不平衡
- **梯度裁剪**：max_grad_norm=1.0，稳定训练
- **早停机制**：dev 集 macro_f1 连续 2 轮不提升即停止
- **阈值校准**：在 dev 集上扫描 τ ∈ [0.05, 0.95]，选择使 accuracy 最大的决策阈值
- **备选 backbone 对比**：twitter-roberta-base vs DeBERTa-v3-base/large vs BERTweet vs RoBERTa-base

### 可解释性方案

- **LLM 解释生成**：通过 SJTU 校内大模型 API（`claw.sjtu.edu.cn`）生成判断依据文本
- **注意力可视化**：提取 BERT 各层 attention weights，聚合跨层/跨头注意力分数，生成关键词排序和高亮文本

## 实验结果

### val.csv 最终性能


| 指标                        | 数值                           |
| ------------------------- | ---------------------------- |
| **准确率 (Accuracy)**        | **88.03%**                   |
| 平衡准确率 (Balanced Accuracy) | 87.77%                       |
| macro_f1                  | 0.8782                       |
| 谣言 F1                     | 0.8621                       |
| 非谣言 F1                    | 0.8943                       |
| 阈值 (τ)                    | 0.46                         |
| 混淆矩阵                      | TP=150, TN=203, FP=23, FN=25 |


### 7 折留一事件交叉验证


| 指标       | 均值    | 标准差    |
| -------- | ----- | ------ |
| 准确率      | 64.9% | ±18.4% |
| macro_f1 | 0.568 | ±0.185 |
| F1（谣言）   | 0.621 | ±0.179 |


> CV 准确率低于最终模型，因为 event 3（Prince 演唱会，98.8% 谣言）和 event 2（埃博拉，100% 谣言）作为 held-out 验证集时，训练集中几乎没有对应类别的样本，导致泛化困难。其余 5 个事件的 F1 均值超过 0.70，证明模型在多数事件上具有良好泛化能力。

### Backbone 对比


| Backbone                   | val Acc    | val macro_f1 |
| -------------------------- | ---------- | ------------ |
| **twitter-roberta-base** ⭐ | **88.03%** | **0.8782**   |
| DeBERTa-v3-large           | 88.78%     | 0.8831       |
| DeBERTa-v3-base            | ~86%       | ~0.86        |
| BERTweet-base              | ~84%       | ~0.84        |
| RoBERTa-base               | ~83%       | ~0.83        |


> DeBERTa-v3-large 略优于 twitter-roberta-base，但模型体积大 3 倍，推理时间增加约 50%，综合考量选择 twitter-roberta-base 作为最终方案。

### 8 模型 Soft Voting

使用 final_model + 7 个 CV fold 模型对 val.csv 做概率平均投票：准确率 86.78%，未超越 single final_model baseline（88.03%），可能是因为 CV fold 模型仅在单个事件上见过验证分布，对 val.csv 的整体分布适应性不如 final_model。

## 运行环境

- Python 3.8+
- 依赖：transformers, torch, pandas, scikit-learn, openai, matplotlib, seaborn, wordcloud, tqdm
- 模型：首次运行自动下载 ~500MB 模型文件

## 提交清单

- ✅ `README.md`：环境与运行说明（本文件）
- 🟡 `report.pdf`：大作业报告（参考 doc/template.md）
- ✅ 可复现代码与必要支持文件

