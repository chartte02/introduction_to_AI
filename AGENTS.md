# AGENTS

本文件帮助 AI 编码代理在本仓库快速进入有效工作状态。

> 最后更新：2026-06-06  
> 项目阶段：✅ 代码开发完成 → 🟡 报告撰写 + 提交准备阶段

## 1) 项目定位

- 课程：上海交通大学《人工智能导论》大作业（2026）
- 任务：可解释的谣言检测（二分类 + 文字判断依据）
- 目标输出：
  - 分类标签：0=非谣言，1=谣言
  - 解释文本：说明判断依据

权威任务说明：`doc/guidance.md`  
报告模板：`doc/template.md`  
分工方案：`doc/division.md`

## 2) 当前仓库现状

### ✅ 已完成模块

| 模块 | 负责成员 | 状态 | 核心文件 |
|------|---------|------|---------|
| 预处理 + 分类模型 | B | ✅ 完成 | `model/preprocessing.py`, `model/data.py`, `model/model.py`, `model/trainer.py` |
| 数据分析 + 可视化 | A | ✅ 完成 | `src/eda.py`, `src/vocabulary_analysis.py` |
| 可解释性模块 | C | ✅ 完成 | `model/llm_client.py`, `model/prompts.py`, `model/attention_viz.py` |
| 项目集成 + 脚本 | D | ✅ 完成 | `scripts/train.py`, `scripts/eval.py`, `scripts/predict.py`, `README.md` |
| 大作业报告 | D (主笔) | 🟡 待撰写 | `report.pdf`（不在仓库中，需生成提交） |

### 📊 关键训练结果

**最终模型**（`cardiffnlp/twitter-roberta-base`，epoch 3 早停）：
- **val.csv 准确率：88.03%**
- **val.csv macro_f1：0.8782**
- 校准阈值 τ = 0.46（在 dev 集上扫描的最优 accuracy 阈值）
- 混淆矩阵：TP=150, TN=203, FP=23, FN=25

**7 折留一事件交叉验证**：
- 平均准确率：64.9% ± 18.4%（各事件差异大，event 3/4 极难）
- 平均 macro_f1：0.568 ± 0.185
- 结果文件：`outputs/cv_results.json`

**Backbone 对比实验**：twitter-roberta-base > DeBERTa-v3-large > DeBERTa-v3-base > BERTweet > RoBERTa-base  
**8 模型 soft voting**：未超越 baseline single final_model（acc 86.78% vs 88.03%）

### 🎨 已生成的可视化素材

`outputs/figures/`：
- `label_distribution.png` — 标签分布饼图+柱状图
- `text_length_analysis.png` — 文本长度直方图+密度图+箱线图
- `event_distribution.png` — 事件分布柱状图
- `keyword_comparison.png` — 谣言 vs 非谣言关键词对比
- `wordcloud_comparison.png` — 词云对比
- `special_characters_comparison.png` — URL/@/#/!/? 使用率对比

`outputs/tables/`：
- `label_distribution.csv`, `text_length_stats.csv`, `event_distribution.csv`
- `rumor_keywords.csv`, `non_rumor_keywords.csv`, `special_characters_analysis.csv`

## 3) 目录结构

```text
.
├─ data/
│  ├─ train.csv              # 训练集（2840 条推文，7 个事件）
│  └─ val.csv                # 验证集（401 条推文）
├─ doc/
│  ├─ guidance.md             # 课程任务说明
│  ├─ template.md             # 大作业报告模板
│  ├─ division.md             # 小组分工
│  ├─ ai_pipeline.md          # AI 框架链路总结
│  └─ 人工智能导论大作业2026.pdf  # 课程说明原文
├─ model/                     # 核心模型代码
│  ├─ preprocessing.py        # 文本预处理 pipeline（5 步清洗）
│  ├─ data.py                 # 数据加载 + 留一事件 CV 划分 + 分层 split
│  ├─ model.py                # RumorClassifier（RoBERTa + 分类头）
│  ├─ trainer.py              # 训练/评估/CV/最终模型（含类别加权+早停）
│  ├─ llm_client.py           # SJTU LLM API 客户端（OpenAI 兼容）
│  ├─ prompts.py              # 可解释性 Prompt 模板
│  └─ attention_viz.py        # 注意力提取/高亮/热力图/关键词排序
├─ scripts/                   # 运行入口
│  ├─ train.py                # 训练入口（--cv 交叉验证 / 全量训练）
│  ├─ eval.py                 # 评估入口（val.csv 上报告各类指标）
│  ├─ predict.py              # 单条推理入口（交互式 / 命令行）
│  ├─ run_cv.py               # 便捷 CV 训练脚本（含日志输出）
│  ├─ calibrate.py            # 阈值校准脚本（扫 τ ∈ [0.05, 0.95]）
│  ├─ compare_backbones.py    # 多 backbone 对比评估
│  ├─ vote.py                 # 8 模型 soft voting 评估
│  ├─ eval_explain.py         # 解释质量评估（抽样+LLM+CSV）
│  └─ diagnose.py             # 环境诊断脚本
├─ src/                       # 数据分析模块
│  ├─ eda.py                  # 数据探索性分析（标签分布/长度/事件）
│  └─ vocabulary_analysis.py  # 词汇分析（关键词/词云/特殊字符）
├─ outputs/                   # 输出产物（.gitignore 忽略）
│  ├─ final_model.pt          # 最终模型权重（~500MB，需训练生成）
│  ├─ fold_1_best.pt ~ fold_7_best.pt  # CV 各折最佳模型
│  ├─ tokenizer/              # tokenizer 文件
│  ├─ cv_results.json         # 交叉验证结果
│  ├─ threshold.json          # 阈值校准结果
│  ├─ vote_results.json       # soft voting 结果
│  ├─ backbone_comparison.json
│  ├─ figures/                # EDA + 词汇分析图表（6 张）
│  └─ tables/                 # 分析数据表（6 个 CSV）
├─ .venv/                     # Python 虚拟环境（.gitignore）
├─ requirements.txt
├─ AGENTS.md                  # 本文件
└─ README.md
```

## 4) 数据约定

`train.csv`/`val.csv` 字段：
- `id`：样本 ID
- `text`：推文文本（模型唯一输入）
- `label`：监督标签（0=非谣言，1=谣言）
- `event`：事件标识（0-6，共 7 个新闻事件，**仅用于 CV 划分，不参与训练**）

处理数据规则：
- 不改动原始数据文件
- 所有中间产物写入 `outputs/` 目录
- 训练与验证严格区分，不将 `val.csv` 混入训练
- 所有随机种子 = 42

## 5) 代码与沟通约定

- 默认使用中文：注释、文档、对用户说明均优先中文
- 新增代码优先可复现：固定随机种子、记录关键超参
- 解释性输出应与分类结果同批次产出
- 训练输入仅使用 `text`，不使用 `event` 作为特征

## 6) 模型与训练

### 分类模型

选用 `cardiffnlp/twitter-roberta-base`（1.25 亿参数），在 5800 万条推文上预训练。

分类头设计：`[CLS] vector → Dropout(0.1) → Linear(768, 2)`

### 训练策略关键配置

| 配置项 | 值 | 说明 |
|--------|-----|------|
| batch_size | 32 (GPU) / 8 (CPU) | |
| learning_rate | 2e-5 | |
| num_epochs | 10 (实际 epoch 3 早停) | |
| 类别加权损失 | True | 缓解谣言/非谣言比例不平衡 |
| 梯度裁剪 | 1.0 | 稳定训练 |
| 选择指标 | macro_f1 | 对不平衡数据更公平 |
| dev_ratio | 0.1 | 分层划分，用于早停 |
| patience | 2 | 连续 2 轮不提升即早停 |
| 阈值校准 | τ=0.46 | 在 dev 集上扫 τ 取最优 accuracy |

### 备选模型

- `bert-base-uncased` / `roberta-base`（`--model` 参数切换）
- 远程服务器上额外训练了 DeBERTa-v3-base/large、BERTweet-base，但 twitter-roberta-base 表现最佳

### HuggingFace 网络配置

代码已配置自动使用 `hf-mirror.com` 镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export no_proxy="*"
```
首次运行自动下载模型文件（~500MB）。

## 7) 文本预处理 Pipeline

```
原始推文
  ├─ html.unescape()           # HTML 实体解码
  ├─ URL → [URL]               # 保留"存在链接"信号
  ├─ @mention → @USER          # 保留"引用行为"信号
  ├─ # 去除                    # 保留标签文字语义
  ├─ 合并多余空白
  └─→ BERT Tokenizer (max_length=128)
```

## 8) 事件泛化策略（Leave-One-Event-Out CV）

训练和验证包含相同的 7 个事件。留一事件交叉验证确保模型泛化到未见过的事件。

| event | 真实事件 | 样本数 | 谣言比例 |
|-------|---------|--------|---------|
| 0 | Gurlitt 纳粹艺术品 | 66 | 19.7% |
| 1 | Ferguson/Mike Brown | 799 | 24.8% |
| 2 | Essien 埃博拉谣言 | 9 | 100% |
| 3 | Prince 惊喜演唱会 | 162 | 98.8% |
| 4 | Germanwings 空难 | 327 | 50.8% |
| 5 | 悉尼人质事件 | 854 | 42.7% |
| 6 | 渥太华枪击 | 623 | 52.8% |

## 9) 运行命令

### 环境搭建
```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows Git Bash
pip install -r requirements.txt
```

### 训练
```bash
python scripts/train.py                    # 全量训练最终模型
python scripts/train.py --epochs 20 --patience 3  # 自定义参数
```

### 评估
```bash
python scripts/eval.py                     # 在 val.csv 上评估
python scripts/calibrate.py                # 阈值校准
```

### 推理
```bash
python scripts/predict.py                  # 交互式推理
python scripts/predict.py --text "BREAKING: ..."
```

### 可解释性评估
```bash
python scripts/eval_explain.py --sample 50 --attention  # LLM 解释 + 注意力分析
```

### 数据分析
```bash
python src/eda.py                          # EDA
python src/vocabulary_analysis.py          # 词汇分析
```

## 10) 提交前检查清单

- ✅ 分类输出为 0/1，val.csv 上准确率 88.03%
- ✅ 解释文本可读（LLM + 注意力双重方案）
- 🟡 `README.md` — 需更新实际结果数据
- 🟡 `report.pdf` — **待撰写**（30 分，最大项）
- ⚠️ 模型权重文件（`final_model.pt`）不在仓库中，需在提交说明中标注如何获取/复现

## 11) 避免事项

- 不要在无依据时编造数据字段或评分结果
- 不要把课程要求复制到多个文件造成维护重复，优先链接原文档
- 不要提交依赖本地私有路径的脚本
- 不要将大文件（模型权重 .pt）提交到 Git 仓库
