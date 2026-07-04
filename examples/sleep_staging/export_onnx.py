# -*- coding: utf-8 -*-
"""
ONNX export + INT8 quantization pipeline for ParaSleep.

Usage
-----
    python export_onnx.py --checkpoint parasleep.pth --output parasleep_int8.onnx

Workflow:
    1. Load trained ParaSleep checkpoint
    2. Export to FP32 ONNX
    3. INT8 static quantization (via ONNX Runtime)
    4. Verify accuracy on test data
    5. Benchmark inference speed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

import os, argparse, time
import numpy as np

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import (quantize_static, QuantType, QuantFormat,
                                       CalibrationDataReader)

from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
from metabci.brainda.paradigms.sleep import SleepParadigm
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod


# =============================================================================
# Calibration data reader (required for static quantization)
# =============================================================================

class SleepCalibrationDataReader(CalibrationDataReader):
    """Feeds calibration data to ONNX Runtime's static quantizer."""

    def __init__(self, data: np.ndarray, batch_size: int = 1):
        """data shape: (n_epochs, 1, 3000) float32"""
        self.data = data
        self.batch_size = batch_size
        self.iter = 0
        self.n_samples = data.shape[0]

    def get_next(self):
        if self.iter >= self.n_samples:
            return None
        end = min(self.iter + self.batch_size, self.n_samples)
        batch = self.data[self.iter:end]
        self.iter = end
        return {"X": batch}

    def rewind(self):
        self.iter = 0


# =============================================================================
# Export
# =============================================================================

def export_to_onnx(model, onnx_path: str, n_channels: int = 3, opset: int = 17):
    """Export PyTorch model to FP32 ONNX.

    Parameters
    ----------
    model : nn.Module
        Trained ParaSleep model (float32, eval mode).
    onnx_path : str
        Output .onnx file path.
    opset : int
        ONNX opset version. 17+ recommended for GELU/MHA support.
    """
    model.eval()
    model.cpu()

    dummy = torch.randn(1, n_channels, 3000, dtype=torch.float32)
    input_names = ["X"]
    output_names = ["output"]
    dynamic_axes = {
        "X": {0: "batch_size"},
        "output": {0: "batch_size"},
    }

    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
    )

    # Verify
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[Export] FP32 ONNX saved to: {onnx_path}")
    print(f"[Export] Model size: {Path(onnx_path).stat().st_size / 1024:.1f} KB")


# =============================================================================
# INT8 quantization
# =============================================================================

def quantize_int8(
    fp32_path: str,
    int8_path: str,
    calibration_data: np.ndarray,
):
    """Apply INT8 static quantization to an FP32 ONNX model.

    Parameters
    ----------
    fp32_path : str
        Path to the FP32 ONNX model.
    int8_path : str
        Output path for the INT8 quantized model.
    calibration_data : np.ndarray
        Representative data for calibration (n_samples, 1, 3000).
    """
    reader = SleepCalibrationDataReader(calibration_data[:500])  # 500 samples for calibration

    quantize_static(
        model_input=fp32_path,
        model_output=int8_path,
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        weight_type=QuantType.QInt8,
        activation_type=QuantType.QUInt8,
        per_channel=False,
        reduce_range=False,
    )

    fp32_size = Path(fp32_path).stat().st_size / 1024
    int8_size = Path(int8_path).stat().st_size / 1024
    print(f"[Quantize] INT8 model saved to: {int8_path}")
    print(f"[Quantize] FP32: {fp32_size:.1f} KB → INT8: {int8_size:.1f} KB "
          f"(reduction: {100 * (1 - int8_size / fp32_size):.1f}%)")


# =============================================================================
# Verification
# =============================================================================

def verify_accuracy(onnx_path: str, X_test: np.ndarray, y_test: np.ndarray,
                    pytorch_model=None):
    """Compare ONNX model predictions against ground truth (and optionally PyTorch).

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model (FP32 or INT8).
    X_test : np.ndarray, shape (n, 1, 3000)
        Test data.
    y_test : np.ndarray, shape (n,)
        Ground truth labels.
    pytorch_model : nn.Module, optional
        If provided, also check ONNX vs PyTorch consistency.

    Returns
    -------
    accuracy : float
    """
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    correct = 0
    total = len(y_test)
    pt_match = 0

    for i in range(total):
        x = X_test[i:i + 1].astype(np.float32)
        onnx_out = session.run(["output"], {"X": x})[0]
        onnx_pred = int(onnx_out.argmax())

        if onnx_pred == y_test[i]:
            correct += 1

        if pytorch_model is not None:
            with torch.no_grad():
                pt_out = pytorch_model(
                    torch.from_numpy(x)
                )
                if isinstance(pt_out, (list, tuple)):
                    pt_out = torch.stack(list(pt_out)).mean(0)
                pt_out = pt_out.numpy()
            if np.argmax(onnx_out) == np.argmax(pt_out):
                pt_match += 1

    acc = correct / total * 100
    print(f"[Verify] Accuracy: {acc:.2f}% ({correct}/{total})")

    if pytorch_model is not None:
        consistency = pt_match / total * 100
        print(f"[Verify] ONNX vs PyTorch consistency: {consistency:.2f}%")

    return acc


# =============================================================================
# Benchmark
# =============================================================================

def benchmark(onnx_path: str, X_test: np.ndarray, n_warmup: int = 10,
              n_runs: int = 100):
    """Benchmark ONNX model inference latency.

    Parameters
    ----------
    onnx_path : str
        Path to the ONNX model.
    X_test : np.ndarray
        Test data.
    n_warmup : int
        Warmup iterations.
    n_runs : int
        Measurement iterations.

    Returns
    -------
    mean_latency_ms : float
    """
    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )
    x = X_test[0:1].astype(np.float32)

    # Warmup
    for _ in range(n_warmup):
        session.run(["output"], {"X": x})

    # Measure
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(["output"], {"X": x})
        times.append(time.perf_counter() - t0)

    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"[Benchmark] {mean_ms:.2f} ± {std_ms:.2f} ms per epoch (n={n_runs})")
    return mean_ms


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="ParaSleep ONNX export + INT8 quantization")
    parser.add_argument("--checkpoint", type=str,
                        default="examples/sleep_staging/exp_ctx3_causal.pth",
                        help="Path to trained ParaSleep .pth checkpoint")
    parser.add_argument("--output", type=str, default="parasleep_int8.onnx",
                        help="Output ONNX path (default: parasleep_int8.onnx)")
    parser.add_argument("--context", type=int, default=3,
                        help="Context window size (default: 3)")
    parser.add_argument("--cache", type=str, default='examples/sleep_staging/data_cache',
                        help="Cache directory for calibration data")
    parser.add_argument("--data-root", type=str,
                        default=r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette",
                        help="Path to sleep-edf data root (fallback)")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip INT8 quantization (FP32 only)")
    parser.add_argument("--causal", action="store_true",
                        help="Use causal context (for models trained with --causal)")
    parser.add_argument("--verify", type=int, default=5,
                        help="Number of test subjects for verification (default: 5)")
    args = parser.parse_args()

    mode_str = 'causal' if args.causal else 'center'

    print("=" * 60)
    print("ParaSleep ONNX Export + INT8 Quantization")
    print("=" * 60)

    # ---- 1. Load model ----
    print(f"\n[1/5] Loading trained model (ctx={args.context} {mode_str})...")
    raw_cls = lwmod.ParaSleep.module
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_absolute():
        ckpt_path = _PROJECT_ROOT / ckpt_path
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    n_ch = int(args.context)

    # Auto-detect architecture
    use_ta = any(k.startswith('transformer.') for k in state.keys())
    target_idx = 'last' if (use_ta and args.causal) else 'center'
    arch = 'TA' if use_ta else 'Base'

    model = raw_cls(n_channels=n_ch, n_samples=3000, n_classes=5,
                    use_temporal_attention=use_ta,
                    target_index=target_idx).float()
    model.load_state_dict(state, strict=False)
    model.eval()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {arch} | target_index: {target_idx} | Params: {total_params:,}")

    # ---- 2. Load calibration data (from cache if available) ----
    print("\n[2/5] Loading calibration data...")
    import glob as _glob
    cache_dir = str(_PROJECT_ROOT / args.cache) if not Path(args.cache).is_absolute() else args.cache
    # Try new naming first, fall back to old patterns
    patterns = [
        f'*_FpzCz_sr100_ctx{args.context}_{mode_str}_5class.npz',
        f'*_ctx{args.context}_{mode_str}.npz',
        f'*_FpzCz_sr100_ctx{args.context}_center_5class.npz',
        f'*_ctx{args.context}_center.npz',
        '*.npz',
    ]
    cal_files = []
    for pat in patterns:
        cal_files = sorted(_glob.glob(os.path.join(cache_dir, pat)))
        if cal_files:
            break
    cal_files = cal_files[:args.verify]
    X_list, y_list = [], []
    for f in cal_files:
        d = np.load(f)
        X_list.append(d['X']); y_list.append(d['y'])
    X_cal = np.concatenate(X_list).astype(np.float32)
    y_cal = np.concatenate(y_list).astype(np.int64)
    print(f"  Subjects: {len(cal_files)}")
    print(f"  Samples: {X_cal.shape[0]}")

    # ---- 3. Export to ONNX ----
    print("\n[3/5] Exporting to FP32 ONNX...")
    fp32_path = args.output.replace(".onnx", "_fp32.onnx")
    export_to_onnx(model, fp32_path, n_channels=n_ch)

    # ---- 4. INT8 quantization ----
    if args.skip_quantize:
        int8_path = fp32_path
        print("\n[4/5] INT8 quantization skipped (--skip-quantize)")
    else:
        print("\n[4/5] Applying INT8 static quantization...")
        int8_path = args.output
        quantize_int8(fp32_path, int8_path, X_cal)

    # ---- 5. Verify & benchmark ----
    print("\n[5/5] Verification & Benchmark...")
    print(f"\n--- FP32 Model ---")
    verify_accuracy(fp32_path, X_cal, y_cal, pytorch_model=model)
    benchmark(fp32_path, X_cal)

    if not args.skip_quantize:
        print(f"\n--- INT8 Model ---")
        verify_accuracy(int8_path, X_cal, y_cal)
        benchmark(int8_path, X_cal)

    print(f"\n{'=' * 60}")
    print("Done.")
    print(f"  FP32: {fp32_path}")
    if not args.skip_quantize:
        print(f"  INT8: {int8_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
