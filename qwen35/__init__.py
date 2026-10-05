"""Mirai Qwen3.5 inference on Asahi Linux."""
import os

# Configure BLAS before importing NumPy in any CLI or verification module.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
