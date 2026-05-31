#!/usr/bin/env python3
"""轻量级CV训练入口 - 直接运行 7 折交叉验证。

输出同时写入 ai_outputs/cv_training.log，方便监控进度。
运行方式：python ai_scripts/run_cv.py
"""

import io
import sys
from pathlib import Path

# ── 最早修复：Windows 控制台 UTF-8 ──
if sys.platform == "win32" and hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8")

# ── 项目根目录加入路径 ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 日志文件 ──
OUTPUTS_DIR = PROJECT_ROOT / "ai_outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUTS_DIR / "cv_training.log"

# ── Tee: 同时输出到控制台和日志文件 ──
_real_stdout = sys.stdout
_log_file = open(LOG_FILE, "w", encoding="utf-8")


class _Tee:
    def write(self, data):
        _real_stdout.write(data)
        _log_file.write(data)
        _log_file.flush()

    def flush(self):
        _real_stdout.flush()
        _log_file.flush()


from ai_model.trainer import run_cross_validation

# 在所有 import 完成后再切替换 stdout（避免与 trainer.py 的包装冲突）
sys.stdout = _Tee()

print("Starting 7-fold CV training...")
print(f"Log file: {LOG_FILE}")
print(f"Model: cardiffnlp/twitter-roberta-base")
print(f"Epochs: 3, Batch: 8, LR: 2e-5")
print(f"Device: cpu")
print("=" * 50)

summary = run_cross_validation(
    model_name="cardiffnlp/twitter-roberta-base",
    max_length=128,
    batch_size=8,
    learning_rate=2e-5,
    num_epochs=3,
    device="cpu",
)

print("\nCV training completed!")
_log_file.close()
