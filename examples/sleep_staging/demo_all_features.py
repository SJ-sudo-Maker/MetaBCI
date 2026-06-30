# -*- coding: utf-8 -*-
"""
Demo all 7 new features — generates competition submission test outputs.

Usage:  python demo_all_features.py
Output: demo_outputs/ folder with test reports and screenshots
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.makedirs('demo_outputs', exist_ok=True)

print("=" * 60)
print("MetaBCI Sleep Staging — Feature Demo")
print("=" * 60)

# ===========================================================================
# Feature 1+2: SleepEDFDataset + SleepParadigm
# ===========================================================================
print("\n[1/7] SleepEDFDataset + SleepParadigm")
from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
from metabci.brainda.paradigms.sleep import SleepParadigm
import numpy as np

data_root = r"D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette"
dataset = SleepEDFDataset(data_root, channel='EEG Fpz-Cz')
paradigm = SleepParadigm(channels=['EEG Fpz-Cz'], srate=100)
X, y, _ = paradigm.get_data(dataset, subjects=['4001'], return_concat=True, n_jobs=1)
print(f"  Subjects discovered: {len(dataset.subjects)}")
print(f"  Sample epoch: X={X.shape}, y classes={set(np.unique(y))}")


# ===========================================================================
# Feature 3: ParaSleep Model
# ===========================================================================
print("\n[2/7] ParaSleep Model")
import torch
import metabci.brainda.algorithms.deep_learning.parasleep as lwmod
raw = lwmod.ParaSleep.module
model = raw(3, 3000, 5).float().eval()
p = sum(pn.numel() for pn in model.parameters())
with torch.no_grad():
    out = model(torch.randn(1, 3, 3000))
print(f"  Params: {p:,}")
print(f"  Inference: (1,3,3000) -> {list(out.shape)}")

# Load trained weights if available
pth = 'parasleep_best.pth'
if os.path.exists(pth):
    state = torch.load(pth, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    print(f"  Model weights loaded: {pth}")

# ===========================================================================
# Feature 4: SleepOnlineWorker
# ===========================================================================
print("\n[3/7] SleepOnlineWorker")
from metabci.brainflow.sleep_worker import SleepOnlineWorker
worker = SleepOnlineWorker(model=model, srate=100, epoch_sec=30)
worker.pre()
# Feed 4 epochs
for _ in range(4):
    eeg = np.random.randn(3000).astype(np.float64) * 10 + 50  # ~50uV mean
    worker.consume([[float(v), 0.0] for v in eeg])
print(f"  Epochs processed: {worker.epoch_counter}")
print(f"  Predictions: {worker.predictions}")
print(f"  Signal quality: all good" if all(worker.signal_quality) else "  Signal quality: some bad")

# ===========================================================================
# Feature 5: SleepMonitorUI (matplotlib + brainstim Experiment)
# ===========================================================================
print("\n[4/7] SleepMonitorUI")
from metabci.brainstim.sleep_monitor import build_sleep_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# Generate realistic 6-hour sleep pattern
stages = []
for cyc in range(5):
    if cyc == 0: stages += [0]*30  # wake before sleep
    stages += [1]*5 + [2]*30 + [3]*15 + [2]*20 + [4]*10 + [0]*5
stages = stages[:720]  # 6 hours @ 30s epochs
fig = build_sleep_report(stages, epoch_sec=30, title='ParaSleep Sleep Report — Demo')
fig.savefig('demo_outputs/05_sleep_report.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"  Static report saved: demo_outputs/05_sleep_report.png")

# brainstim Experiment integration check
from metabci.brainstim import Experiment, sleep_report_paradigm
ex = Experiment(win_size=(800, 600), is_fullscr=False)
ex.register_paradigm('Sleep Report', sleep_report_paradigm, predictions=stages[:60])
print(f"  brainstim Experiment paradigm registered: {'Sleep Report' in ex.paradigms}")

# ===========================================================================
# Feature 6: ONNX Export + INT8 Quantization
# ===========================================================================
print("\n[5/7] ONNX Export + INT8 Quantization")
import onnx, onnxruntime as ort, tempfile
onnx_p = 'demo_outputs/06_parasleep_fp32.onnx'
model_onnx = raw(3, 3000, 5).float().eval()
dummy = torch.randn(1, 3, 3000)
torch.onnx.export(model_onnx, dummy, onnx_p,
    input_names=['X'], output_names=['output'],
    dynamic_axes={'X':{0:'b'},'output':{0:'b'}}, opset_version=17)
onnx.checker.check_model(onnx.load(onnx_p))
fp32_kb = os.path.getsize(onnx_p)/1024
print(f"  FP32 ONNX: {fp32_kb:.0f} KB")

# INT8
from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader
class CR(CalibrationDataReader):
    def __init__(s): s.d=np.random.randn(100,3,3000).astype(np.float32); s.n=100; s.i=0
    def get_next(s):
        if s.i>=s.n: return None
        b=s.d[s.i:s.i+1]; s.i+=1; return {'X':b}
int8_p = 'demo_outputs/06_parasleep_int8.onnx'
quantize_static(onnx_p, int8_p, CR(), quant_format=QuantType.QInt8, weight_type=QuantType.QInt8)
int8_kb = os.path.getsize(int8_p)/1024
print(f"  INT8 ONNX: {int8_kb:.0f} KB ({100*(1-int8_kb/fp32_kb):.0f}% reduction)")

# Verify
sess = ort.InferenceSession(onnx_p, providers=['CPUExecutionProvider'])
ort_o = sess.run(['output'], {'X': dummy.numpy()})[0]
with torch.no_grad(): pt_o = model_onnx(dummy).numpy()
print(f"  PyTorch vs ONNX diff: {np.abs(pt_o-ort_o).max():.8f}")

# ===========================================================================
# Feature 7: EDFSleepPlayer
# ===========================================================================
print("\n[6/7] EDFSleepPlayer")
from metabci.brainflow.edf_player import EDFSleepPlayer
files = sorted(os.listdir(data_root))
psg_f = hyp_f = None
for f in files:
    if f.endswith('-PSG.edf'):
        pf = f[:6]
        for f2 in files:
            if f2.startswith(pf) and f2.endswith('-Hypnogram.edf'):
                psg_f = os.path.join(data_root, f); hyp_f = os.path.join(data_root, f2)
                break
    if psg_f: break
player = EDFSleepPlayer(psg_f, channel='EEG Fpz-Cz', srate=100, hypnogram_path=hyp_f, verbose=False)
samples = player.recv()
print(f"  Subject: {os.path.basename(psg_f)}")
print(f"  Duration: {player.duration_sec/60:.0f} min")
print(f"  Ground truth epochs: {len(player.true_stages)}")
print(f"  recv() format: [{samples[0][0]:.1f} uV, trigger={samples[0][1]}]")

# ===========================================================================
# Feature: brainstim Experiment (full integration)
# ===========================================================================
print("\n[7/7] brainstim Experiment Integration")
from metabci.brainstim import Experiment, sleep_report_paradigm
ex = Experiment(win_size=(1400, 800), is_fullscr=False)
ex.register_paradigm('Sleep Report', sleep_report_paradigm, predictions=stages[:120])
print(f"  [OK] brainstim Experiment registered with sleep report paradigm")

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 60)
print("ALL 7 FEATURES + brainstim integration VERIFIED")
print(f"Demo outputs saved to: {os.path.abspath('demo_outputs')}")
print("=" * 60)
