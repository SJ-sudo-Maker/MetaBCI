# -*- coding: utf-8 -*-
"""
MetaBCI Sleep Staging — unified command line entry.

Single entry point replacing scattered scripts (train_server, demo_metric, etc.).
All commands use the same config, pipeline, and data paths.

Usage
-----
    python cli.py prepare  --data F:/sleep-edf/.../sleep-cassette --cache F:/sleep_cache_chronov3
    python cli.py split    --cache F:/sleep_cache_chronov3
    python cli.py train    --cache F:/sleep_cache_chronov3 --epochs 60 --save model.pth
    python cli.py evaluate --model model.pth --cache F:/sleep_cache_chronov3
    python cli.py export   --model model.pth
    python cli.py demo     --model model.pth --cache F:/sleep_cache_chronov3 --subject 4241
"""

import sys, os, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from metabci.brainda.pipelines.sleep_staging import SleepStagingPipeline


def _resolve_path(cli_val, env_key, default):
    """Resolve path: CLI arg > env var > default."""
    val = cli_val if cli_val is not None else None
    if val is None:
        val = os.environ.get(env_key)
    return val if val else default

def _build_config(args):
    """Build pipeline config dict from CLI args."""
    device = getattr(args, 'device', 'auto')
    if device == 'auto':
        import torch
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return {
        "data_root": _resolve_path(
            getattr(args, 'data', None), "SLEEP_DATA",
            r"F:\sleep-edf\sleep-edf-database-expanded-1.0.0\sleep-cassette"),
        "cache_dir": _resolve_path(
            getattr(args, 'cache', None), "SLEEP_CACHE",
            r"F:\sleep_cache_chronov3"),
        "context": getattr(args, 'context', 3),
        "causal": getattr(args, 'causal', True),
        "label_mode": "5class",
        "channel": "EEG Fpz-Cz",
        "epochs": getattr(args, 'epochs', 60),
        "batch_size": getattr(args, 'batch', 128),
        "lr": getattr(args, 'lr', 1e-3),
        "wd": getattr(args, 'wd', 1e-2),
        "device": device,
        "cache_version": "chronov3",
        "seed": 42,
        "train_subjects": getattr(args, 'subjects', 68),
        "test_subjects": getattr(args, 'test', 10),
        "save_path": _resolve_path(
            getattr(args, 'save', None), None, 'parasleep_best.pth'),
    }


def cmd_prepare(args):
    """Build chronov3 caches from raw EDF."""
    config = _build_config(args)
    pipeline = SleepStagingPipeline(config)
    pipeline.prepare()


def cmd_split(args):
    """Show subject-wise train/test split."""
    config = _build_config(args)
    pipeline = SleepStagingPipeline(config)
    split = pipeline.split()
    # Save split
    save = getattr(args, 'save', 'parasleep_best.pth')
    np.savez(save.replace('.pth', '_split.npz'),
             train_subjects=np.array(split['train_subjects']),
             test_subjects=np.array(split['test_subjects']),
             train_records=np.array(split['train_records']),
             test_records=np.array(split['test_records']))
    print(f"Split saved to {save.replace('.pth', '_split.npz')}")


def cmd_train(args):
    """Train ParaSleep model."""
    config = _build_config(args)
    pipeline = SleepStagingPipeline(config)
    pipeline.split()
    pipeline.train()


def cmd_evaluate(args):
    """Evaluate model on holdout test set."""
    config = _build_config(args)
    pipeline = SleepStagingPipeline(config)
    pipeline.split()
    model_path = getattr(args, 'model', None) or config['save_path']
    pipeline.evaluate(model_path)


def cmd_export(args):
    """Export model to ONNX."""
    config = _build_config(args)
    pipeline = SleepStagingPipeline(config)
    model_path = getattr(args, 'model', None) or config['save_path']
    output = getattr(args, 'output', 'parasleep.onnx')
    pipeline.export_onnx(model_path, output)


def cmd_demo(args):
    """Run online sleep staging demo."""
    config = _build_config(args)
    pipeline = SleepStagingPipeline(config)
    model_path = getattr(args, 'model', None) or config['save_path']
    subject = getattr(args, 'subject', '4241')
    pipeline.demo(model_path, subject)
    print("For full visualization, run: python demo_e2e.py")


def main():
    parser = argparse.ArgumentParser(description="MetaBCI Sleep Staging CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # prepare
    p = sub.add_parser("prepare")
    p.add_argument("--data", type=str)
    p.add_argument("--cache", type=str)
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--causal", action="store_true", default=True)

    # split
    p = sub.add_parser("split")
    p.add_argument("--cache", type=str)
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--subjects", type=int, default=68)
    p.add_argument("--test", type=int, default=10)
    p.add_argument("--save", type=str, default="parasleep_best.pth")

    # train
    p = sub.add_parser("train")
    p.add_argument("--data", type=str)
    p.add_argument("--cache", type=str)
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-2)
    p.add_argument("--subjects", type=int, default=68)
    p.add_argument("--test", type=int, default=10)
    p.add_argument("--save", type=str, default="parasleep_best.pth")
    p.add_argument("--device", type=str, default="cuda")

    # evaluate
    p = sub.add_parser("evaluate")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--cache", type=str)
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--causal", action="store_true", default=True)
    p.add_argument("--subjects", type=int, default=68)
    p.add_argument("--test", type=int, default=10)

    # export
    p = sub.add_parser("export")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--output", type=str, default="parasleep.onnx")
    p.add_argument("--context", type=int, default=3)

    # demo
    p = sub.add_parser("demo")
    p.add_argument("--model", type=str)
    p.add_argument("--cache", type=str)
    p.add_argument("--subject", type=str, default="4241")
    p.add_argument("--context", type=int, default=3)
    p.add_argument("--causal", action="store_true", default=True)

    args = parser.parse_args()

    commands = {
        "prepare": cmd_prepare,
        "split": cmd_split,
        "train": cmd_train,
        "evaluate": cmd_evaluate,
        "export": cmd_export,
        "demo": cmd_demo,
    }
    commands[args.command](args)


if __name__ == "__main__":
    import numpy as np
    main()
