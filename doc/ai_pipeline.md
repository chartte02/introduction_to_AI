# 可解释谣言检测 — AI 框架链路总结

> 本文档记录本项目的完整 AI 管线设计，供小组成员理解和后续开发参考。
>
> 撰写日期：2026-05-31  
> 当前阶段：链路可行性验证通过（测试状态），待正式训练

---

## 1. 整体架构

```
输入推文（text）
    │
    ├─→ [预处理 pipeline] ──→ 清洗后文本
    │
    ├─→ [BERT 分类器] ──→ label (0/1)          ← ai_model/
    │
    └─→ [SJTU LLM API] ──→ 判断依据文本          ← 用户自行集成
```

- **分类**：基于 `twitter-roberta-base` fine-tune，本地训练 + 推理
- **解释**：调用 SJTU 校内 LLM（DeepSeek 等），纯 API 调用，无需训练

---

## 2. 数据链路

### 2.1 数据概况

| 字段 | 含义 | 来源 |
|------|------|------|
| `id` | 样本 ID | 原始数据 |
| `text` | 推文文本 | 原始数据 — **模型唯一输入** |
| `label` | 0=非谣言, 1=谣言 | 监督标签 |
| `event` | 事件 ID（0-6） | 元数据 — **不参与训练，仅用于 CV 划分** |

train.csv: 2840 条, val.csv: 401 条

### 2.2 事件分布

| event | 真实事件 | 样本数 | 谣言比例 |
|-------|---------|--------|---------|
| 0 | Gurlitt 纳粹艺术品争议 | 66 | 19.7% |
| 1 | Ferguson / Mike Brown 枪击 | 799 | 24.8% |
| 2 | Essien 埃博拉谣言 | 9 | 100% |
| 3 | Prince 多伦多惊喜演唱会 | 162 | 98.8% |
| 4 | Germanwings 4U9525 空难 | 327 | 50.8% |
| 5 | 悉尼 Lindt 咖啡馆人质事件 | 854 | 42.7% |
| 6 | 渥太华议会枪击事件 | 623 | 52.8% |

---

## 3. 预处理链路

```
原始推文
  │
  ├─ html.unescape()           # &amp; → &    &gt; → >    &lt; → <
  ├─ URL → [URL]               # http://t.co/abc → [URL]（保留"存在链接"信号）
  ├─ @mention → @USER          # @CNN → @USER（保留"引用行为"信号）
  ├─ # 去除                    # #Ferguson → Ferguson（保留文字语义）
  ├─ 合并多余空白              # 清理碎片
  │
  └─→ BERT Tokenizer           # tokenize + padding + truncation（max_length=128）
```

**设计依据**：
- URL 本身对 BERT 无意义（被切成 http : / / t . co），但"推文是否含链接"可能是谣言信号
- @mention 具体用户名是噪音，但"引用/转发"行为模式保留
- `#` 导致 RoBERTa tokenizer 错误切分（`#Ferguson` → `#` + `Fer` + `##gus` + `##on`），去除后正常

---

## 4. 分类模型链路

### 4.1 模型架构

```
text → RoBERTa Encoder (125M params)
           │
           └─→ [CLS] vector (768 dim)
                    │
                    ├─→ Dropout(0.1)
                    │
                    └─→ Linear(768 → 2)
                            │
                            └─→ logits → softmax → [P(非谣言), P(谣言)]
```

### 4.2 选用模型

`cardiffnlp/twitter-roberta-base`
- 在 5800 万条推文上预训练，对推文特有的非正式语言、hashtag、emoji 天然适配
- 125M 参数，单卡可训练，CPU 亦可行（单 epoch ≈ 6 分钟）
- 备选：`bert-base-uncased`、`roberta-base`（通过 `--model` 参数切换）

### 4.3 训练超参（默认）

| 参数 | 值 |
|------|-----|
| learning_rate | 2e-5 |
| batch_size | 16 (GPU) / 8 (CPU) |
| num_epochs | 3~5 |
| warmup_ratio | 0.1 |
| weight_decay | 0.01 |
| dropout | 0.1 |
| max_length | 128 |
| optimizer | AdamW |
| scheduler | Linear warmup + linear decay |

---

## 5. 训练链路：留一事件交叉验证

### 5.1 为什么需要

train 和 val 包含完全相同的 7 个事件。如果直接随机划分，模型会通过事件特有的语言风格（如 `#Ferguson` 高频出现）走捷径，而不是真正学习谣言语言模式。

### 5.2 实施方案

```
Fold 1: train=[1,2,3,4,5,6], val=[0]   → 在 event 0 上评估
Fold 2: train=[0,2,3,4,5,6], val=[1]   → 在 event 1 上评估
Fold 3: train=[0,1,3,4,5,6], val=[2]   → 在 event 2 上评估
...
Fold 7: train=[0,1,2,3,4,5], val=[6]   → 在 event 6 上评估
```

- 报告 7 折平均 Acc ± std 和平均 F1 ± std
- 最终模型：用最优超参在全部 train.csv 上训练，在 val.csv 上评估

### 5.3 运行方式

```bash
# 交叉验证
python ai_scripts/train.py --cv --epochs 5

# 全量训练最终模型
python ai_scripts/train.py --epochs 5
```

---

## 6. 评估指标

| 指标 | 计算 | 用途 |
|------|------|------|
| Accuracy | (TP+TN)/(总数) | 主要报告指标 |
| Precision | TP/(TP+FP) | 谣言判定的准确性 |
| Recall | TP/(TP+FN) | 谣言召回完整性 |
| F1 | 2×P×R/(P+R) | 精确率与召回率的平衡 |
| 7-fold mean±std | — | 泛化能力证据 |

---

## 7. 推理链路

```bash
# 交互式
python ai_scripts/predict.py

# 命令行单条
python ai_scripts/predict.py --text "推文内容"

# 输出示例
# → 谣言（置信度: 0.8723）
```

### 推理流程

```
输入 text
  │
  ├─ clean_text() 预处理
  ├─ tokenizer encode
  ├─ model forward → logits
  ├─ softmax → probability
  │
  └─→ {"label": 0/1, "label_name": "谣言/非谣言", "confidence": 0.XX}
```

---

## 8. 大模型解释链路（待集成）

```
BERT label + text → SJTU LLM API → 判断依据文本
```

由小组成员自行对接 SJTU API。建议 prompt 模板：

```
你是一个谣言检测专家。以下是一条推文，已被分类模型判定为「{标签}」。
请你分析这条推文，用 2-3 句话解释判断依据。
从以下角度考虑：信息来源可靠性、措辞煽动性、是否有可验证事实、
是否依赖情绪而非事实。

推文：{text}
```

---

## 9. 文件清单

| 文件 | 职责 |
|------|------|
| `ai_model/preprocessing.py` | 5 步文本清洗 |
| `ai_model/data.py` | CSV 加载、RumorDataset、CV fold 划分 |
| `ai_model/model.py` | RumorClassifier（RoBERTa + 分类头） |
| `ai_model/trainer.py` | train_epoch / evaluate / CV / final_train |
| `ai_scripts/train.py` | 训练入口（--cv 启停交叉验证） |
| `ai_scripts/eval.py` | 评估入口 |
| `ai_scripts/predict.py` | 单条推理入口 |
| `ai_scripts/run_cv.py` | 便捷 CV 训练（含日志文件输出） |

---

## 10. 已知限制与后续工作

- [ ] 当前仅跑通 1 epoch 烟雾测试，需要完整训练（7折×5轮 ≈ 3.5 小时 / CPU，约 1 小时 / GPU）
- [ ] 类别不平衡（event 2 全部谣言、event 3 98.8% 谣言）需要加权损失或过采样
- [ ] 大模型解释生成待集成
- [ ] 最终报告待撰写
