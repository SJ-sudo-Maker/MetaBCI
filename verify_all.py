# -*- coding: utf-8 -*-
"""Comprehensive verification of all 7 MetaBCI sleep staging tasks."""
import sys
sys.path.insert(0, '.')
import os, warnings, tempfile
warnings.filterwarnings('ignore')

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')

errors = []
ok = lambda msg: print(f'  [OK] {msg}')
fail = lambda msg: errors.append(msg) or print(f'  [FAIL] {msg}')

print('=' * 60)
print('MetaBCI Sleep Project — Full Verification')
print('=' * 60)

# =============================================
# 1. brainda: Dataset + Paradigm + Model
# =============================================
print('\n[1] brainda module')

try:
    from metabci.brainda.datasets.sleep_edf import SleepEDFDataset
    ok('SleepEDFDataset import')
except Exception as e: fail(f'SleepEDFDataset: {e}')

try:
    from metabci.brainda.paradigms.sleep import SleepParadigm
    p = SleepParadigm()
    ok('SleepParadigm import')
except Exception as e: fail(f'SleepParadigm: {e}')

try:
    import metabci.brainda.algorithms.deep_learning.parasleep as lwmod
    raw = lwmod.ParaSleep.module
    m = raw(1, 3000, 5).float()
    pcount = sum(pn.numel() for pn in m.parameters())
    assert pcount == 131645
    m.eval()
    with torch.no_grad():
        out = m(torch.randn(4, 1, 3000))
    assert out.shape == (4, 5)
    m.train()
    m(torch.randn(2,1,3000)).sum().backward()
    m2 = raw(1,3000,5).double().eval()
    with torch.no_grad():
        m2(torch.randn(2,1,3000,dtype=torch.float64))
    sk = lwmod.ParaSleep(1,3000,5)
    assert hasattr(sk,'fit') and hasattr(sk,'predict')
    ok(f'ParaSleep: {pcount} params, fp32+fp64+grad+SkorchNet')
except Exception as e: fail(f'ParaSleep: {e}')

# =============================================
# 2. brainflow: Worker + EDF Player
# =============================================
print('\n[2] brainflow module')

try:
    from metabci.brainflow.sleep_worker import SleepOnlineWorker
    from metabci.brainflow.workers import ProcessWorker
    assert issubclass(SleepOnlineWorker, ProcessWorker)
    worker = SleepOnlineWorker(model=m.float().eval())
    ok('SleepOnlineWorker')
except Exception as e: fail(f'SleepOnlineWorker: {e}')

try:
    from metabci.brainflow.edf_player import EDFSleepPlayer, quick_test
    from metabci.brainflow.amplifiers import BaseAmplifier, Marker
    assert issubclass(EDFSleepPlayer, BaseAmplifier)

    data_root = r'D:\sleep eeg\sleep-edf-database-expanded-1.0.0\sleep-cassette'
    files = os.listdir(data_root)
    psg = hyp = None
    for f in sorted(files):
        if f.endswith('-PSG.edf'):
            pf = f[:6]
            for f2 in files:
                if f2.startswith(pf) and f2.endswith('-Hypnogram.edf'):
                    psg = os.path.join(data_root, f)
                    hyp = os.path.join(data_root, f2)
                    break
        if psg: break

    player = EDFSleepPlayer(psg, channel='EEG Fpz-Cz', srate=100,
                            hypnogram_path=hyp, verbose=False)
    assert player.n_samples > 0 and player.true_stages is not None
    s = player.recv()
    assert len(s) == 10 and len(s[0]) == 2
    marker = Marker(interval=[0, 30], srate=100, events=None)
    assert marker.latency == 3000
    ok(f'EDFSleepPlayer: {player.duration_sec/60:.0f}min, {len(player.true_stages)} epochs')
except Exception as e: fail(f'EDFSleepPlayer: {e}')

try:
    from metabci.brainflow import (SleepOnlineWorker, EDFSleepPlayer,
                                    quick_test, BaseAmplifier, Marker, ProcessWorker)
    ok('brainflow __init__ exports')
except Exception as e: fail(f'brainflow __init__: {e}')

# =============================================
# 3. brainstim: SleepMonitorUI
# =============================================
print('\n[3] brainstim module')

try:
    from metabci.brainstim.sleep_monitor import (SleepMonitorUI, build_sleep_report,
                                                  generate_report)
    preds = [0,0,2,2,3,3,2,4,4,2,0,1,2,2,3,2,4,0,2,2]
    fig = build_sleep_report(preds)
    import matplotlib.pyplot as plt; plt.close(fig)
    ok('build_sleep_report')
except Exception as e: fail(f'build_sleep_report: {e}')

try:
    monitor = SleepMonitorUI(epoch_sec=30)
    for s in preds:
        monitor.update(s)
    assert len(monitor.predictions) == len(preds)
    ok('SleepMonitorUI real-time')
except Exception as e: fail(f'SleepMonitorUI: {e}')

try:
    from metabci.brainstim import SleepMonitorUI, build_sleep_report, generate_report
    ok('brainstim __init__ exports')
except Exception as e: fail(f'brainstim __init__: {e}')

# =============================================
# 4. ONNX export
# =============================================
print('\n[4] ONNX export + INT8')

try:
    import onnx, onnxruntime
    from onnxruntime.quantization import quantize_static, QuantType, CalibrationDataReader

    tmp = tempfile.gettempdir()
    onnx_path = os.path.join(tmp, '_verify_parasleep.onnx')
    m_onnx = raw(1, 3000, 5).float().eval()
    dummy = torch.randn(1, 1, 3000)
    torch.onnx.export(m_onnx, dummy, onnx_path,
        input_names=['X'], output_names=['output'],
        dynamic_axes={'X':{0:'batch'}, 'output':{0:'batch'}},
        opset_version=17, do_constant_folding=True)
    onnx.checker.check_model(onnx.load(onnx_path))

    sess = onnxruntime.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    ort_out = sess.run(['output'], {'X': dummy.numpy()})[0]
    with torch.no_grad():
        pt_out = m_onnx(dummy).numpy()
    assert np.abs(ort_out - pt_out).max() < 1e-4

    class CR(CalibrationDataReader):
        def __init__(s): s.d = np.random.randn(100,1,3000).astype(np.float32); s.n=len(s.d); s.i=0
        def get_next(s):
            if s.i>=s.n: return None
            b = s.d[s.i:s.i+1]; s.i += 1; return {'X':b}

    int8_path = os.path.join(tmp, '_verify_int8.onnx')
    quantize_static(onnx_path, int8_path, CR(), quant_format=QuantType.QInt8,
                    weight_type=QuantType.QInt8)
    fk = os.path.getsize(onnx_path)/1024
    ik = os.path.getsize(int8_path)/1024
    ok(f'ONNX: FP32={fk:.0f}KB, INT8={ik:.0f}KB ({100*(1-ik/fk):.0f}% reduction)')
except Exception as e: fail(f'ONNX: {e}')

# =============================================
# 5. Pipeline
# =============================================
print('\n[5] End-to-end pipeline')

try:
    dataset = SleepEDFDataset(data_root, channel='EEG Fpz-Cz', subjects=['4001'])
    paradigm = SleepParadigm(channels=['EEG Fpz-Cz'], srate=100)
    assert paradigm.is_valid(dataset)
    X, y, _ = paradigm.get_data(dataset, subjects=['4001'], return_concat=True, n_jobs=1)
    assert X.shape[1] == 1 and X.shape[2] == 3000 and len(np.unique(y)) >= 3
    m_pipe = raw(1,3000,5).float().eval()
    with torch.no_grad():
        out = m_pipe(torch.from_numpy(X[:4]).float())
    assert out.shape == (4,5)
    ok(f'Pipeline: {X.shape[0]} epochs, {len(np.unique(y))} classes → model OK')
except Exception as e: fail(f'Pipeline: {e}')

# =============================================
# SUMMARY
# =============================================
print('\n' + '=' * 60)
if errors:
    print(f'FAILED {len(errions)} checks:')
    for e in errors:
        print(f'  ✗ {e}')
else:
    print('ALL CHECKS PASSED — 0 errors across 5 modules')
print('=' * 60)
