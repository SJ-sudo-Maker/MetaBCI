# -*- coding: utf-8 -*-
"""
SleepMonitorUI: research-oriented sleep report visualization.

Uses standard hypnogram layout with step-plot rendering.
"""

from typing import Optional, List, Dict
import numpy as np

import matplotlib
try:
    import tkinter
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# ---- Chinese font support ----
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

# ---- Sleep stage color palette ----
STAGE_COLORS = {
    0: "#E8E8E8",  # W   — light gray (wake)
    1: "#A8D8EA",  # N1  — light blue
    2: "#7EC8A8",  # N2  — green
    3: "#3A6B9F",  # N3  — deep blue
    4: "#C0392B",  # REM — red
}
STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]

# Y-axis: spaced positions for visual separation between stages
STAGE_Y_POS = {0: 0.5, 1: 2.0, 2: 3.5, 3: 5.0, 4: 6.5}


# ===========================================================================
# Core hypnogram renderer (sleep step-plot style)
# ===========================================================================

def build_sleep_report(
    predictions: List[int],
    epoch_sec: int = 30,
    title: str = "Sleep Report",
) -> Figure:
    """Generate a research-oriented sleep report.

    Parameters
    ----------
    predictions : list of int
        Sleep stage labels (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    epoch_sec : int
        Seconds per epoch (default 30).
    title : str
        Report title.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_epochs = len(predictions)
    if n_epochs == 0:
        raise ValueError("Empty predictions list")

    time_hours = np.arange(n_epochs + 1) * epoch_sec / 3600  # epoch boundaries
    time_centers = (time_hours[:-1] + time_hours[1:]) / 2

    # Map stages to spaced Y positions
    y_data = np.array([STAGE_Y_POS[int(p)] for p in predictions])
    y_bottom = np.array([STAGE_Y_POS[int(p)] - 0.5 for p in predictions])
    y_top = np.array([STAGE_Y_POS[int(p)] + 0.5 for p in predictions])

    # Stage stats
    stages_unique, counts = np.unique(predictions, return_counts=True)
    stage_count = {i: 0 for i in range(5)}
    for s, c in zip(stages_unique, counts):
        stage_count[int(s)] = c

    # Sleep metrics
    sleep_onset = _find_sleep_onset(predictions)
    latency_min = sleep_onset * epoch_sec / 60
    total_min = n_epochs * epoch_sec / 60
    # TST = actual sleep epochs (N1+N2+N3+REM), excluding Wake during the night
    tst_epochs = sum(stage_count[i] for i in [1, 2, 3, 4])
    tst_min = tst_epochs * epoch_sec / 60
    efficiency = (tst_min / max(total_min, 1)) * 100

    # ---- Build figure ----
    fig = plt.figure(figsize=(16, 8), facecolor='white')
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[2, 1],
                          hspace=0.35, wspace=0.25)

    # (A) Hypnogram — sleep step-plot style
    ax_hypno = fig.add_subplot(gs[:, 0], facecolor='#FAFAFA')

    # Draw each epoch as a colored horizontal bar with clear boundaries
    for i in range(n_epochs):
        color = STAGE_COLORS.get(int(predictions[i]), "#888888")
        ax_hypno.fill_between(
            [time_hours[i], time_hours[i + 1]],
            y_bottom[i], y_top[i],
            color=color, alpha=0.95, edgecolor='white', linewidth=0.3,
        )

    # Y axis: stage order with spacing
    ax_hypno.set_yticks([STAGE_Y_POS[s] for s in range(5)])
    ax_hypno.set_yticklabels(STAGE_NAMES, fontsize=10, fontweight='bold')
    ax_hypno.set_ylim(-0.8, 7.8)
    ax_hypno.invert_yaxis()
    ax_hypno.set_ylabel("Sleep Stage", fontsize=11)

    # X axis: hours with grid
    max_h = time_hours[-1]
    ax_hypno.set_xlim(0, max_h)
    ax_hypno.set_xlabel("Time (hours)", fontsize=11)
    ax_hypno.xaxis.set_major_locator(plt.MultipleLocator(1.0 if max_h > 3 else 0.5))
    ax_hypno.xaxis.set_minor_locator(plt.MultipleLocator(0.5 if max_h > 3 else 0.25))
    ax_hypno.grid(axis='x', which='major', color='#CCCCCC', linewidth=0.5, alpha=0.7)
    ax_hypno.set_title(title, fontsize=14, fontweight='bold', pad=10)

    # Duration annotation
    dur_min = n_epochs * epoch_sec / 60
    info = (f"Total: {dur_min:.0f} min  |  "
            f"Sleep Latency: {latency_min:.0f} min  |  "
            f"TST: {tst_min:.0f} min  |  "
            f"Efficiency: {efficiency:.0f}%")
    ax_hypno.text(0.5, -0.08, info, transform=ax_hypno.transAxes, ha="center",
                  fontsize=9, color="#555555")

    # (B) Stage distribution pie
    ax_pie = fig.add_subplot(gs[0, 1])
    _draw_stage_pie(ax_pie, stage_count, n_epochs)

    # (C) Metrics table
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    _draw_metrics_table(ax_table, total_min, latency_min, tst_min,
                        efficiency, stage_count, n_epochs, epoch_sec)

    return fig


# ===========================================================================
# Helpers
# ===========================================================================

def _find_sleep_onset(predictions: List[int]) -> int:
    """First epoch of 3 consecutive non-Wake epochs."""
    for i in range(len(predictions) - 2):
        if all(p != 0 for p in predictions[i:i + 3]):
            return i
    return 0


def _draw_stage_pie(ax, stage_count, n_epochs):
    labels, sizes, colors = [], [], []
    for i in range(5):
        if stage_count[i] > 0:
            labels.append(f"{STAGE_NAMES[i]} ({100*stage_count[i]/n_epochs:.1f}%)")
            sizes.append(stage_count[i])
            colors.append(STAGE_COLORS[i])
    wedges, _, _ = ax.pie(sizes, labels=None, colors=colors, autopct="%1.1f%%",
                           startangle=90, pctdistance=0.6, textprops={"fontsize": 9})
    ax.legend(wedges, labels, loc="lower center", ncol=2, fontsize=8)
    ax.set_title("Stage Distribution", fontweight="bold")


def _draw_metrics_table(ax, total_min, latency_min, tst_min,
                        efficiency, stage_count, n_epochs, epoch_sec):
    n1_pct = 100 * stage_count[1] / max(n_epochs, 1)
    n3_pct = 100 * stage_count[3] / max(n_epochs, 1)
    rem_pct = 100 * stage_count[4] / max(n_epochs, 1)
    nrem_min = (stage_count[1] + stage_count[2] + stage_count[3]) * epoch_sec / 60

    metrics = [
        ("Recording Duration", f"{total_min:.1f} min"),
        ("Sleep Latency", f"{latency_min:.1f} min"),
        ("Total Sleep Time", f"{tst_min:.1f} min"),
        ("Sleep Efficiency", f"{efficiency:.1f}%"),
        ("", ""),
        ("NREM Sleep", f"{nrem_min:.1f} min"),
        ("REM Sleep", f"{stage_count[4] * epoch_sec / 60:.1f} min"),
        ("Deep Sleep (N3)", f"{n3_pct:.1f}%"),
        ("Light Sleep (N1)", f"{n1_pct:.1f}%"),
        ("REM %", f"{rem_pct:.1f}%"),
    ]

    ax.text(0.5, 0.95, "Sleep Metrics", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="center", va="top")
    y = 0.82
    for label, value in metrics:
        if label:
            ax.text(0.05, y, label, transform=ax.transAxes, fontsize=9, va="top")
            ax.text(0.95, y, value, transform=ax.transAxes, fontsize=9, va="top",
                    ha="right", fontweight="bold", color="#2C3E50")
        y -= 0.075


# ===========================================================================
# Real-time monitor
# ===========================================================================

class SleepMonitorUI:
    """Real-time sleep sleep stage monitor.

    Opens a window that updates as new predictions arrive.
    """

    def __init__(self, epoch_sec: int = 30,
                 title: str = "MetaBCI Sleep Monitor"):
        self.epoch_sec = epoch_sec
        self.title = title
        self.predictions: List[int] = []
        self._fig: Optional[Figure] = None

    def open(self, block: bool = False):
        plt.ion()
        self._fig = plt.figure(figsize=(14, 5), facecolor='white')
        self._fig.canvas.manager.set_window_title(self.title)
        self._render_empty()
        if block:
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.1)

    def update(self, stage: int):
        if stage not in range(5):
            raise ValueError(f"Stage must be 0-4, got {stage}")
        self.predictions.append(stage)
        if self._fig is not None:
            self._render_current()
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        plt.ioff()

    def save(self, path: str):
        fig = build_sleep_report(self.predictions, self.epoch_sec,
                                 title=self.title)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def _render_empty(self):
        if self._fig is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, "Waiting for data...",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray")
        ax.axis("off")
        self._fig.canvas.draw_idle()

    def _render_current(self):
        if self._fig is None or len(self.predictions) == 0:
            return
        self._fig.clear()
        n = len(self.predictions)
        time_hours = np.arange(n + 1) * self.epoch_sec / 3600
        total_min = n * self.epoch_sec / 60

        ax = self._fig.add_subplot(111, facecolor='#FAFAFA')

        for i in range(n):
            color = STAGE_COLORS.get(int(self.predictions[i]), "#888888")
            yb = STAGE_Y_POS[int(self.predictions[i])] - 0.5
            yt = STAGE_Y_POS[int(self.predictions[i])] + 0.5
            ax.fill_between(
                [time_hours[i], time_hours[i + 1]], yb, yt,
                color=color, alpha=0.95, edgecolor='white', linewidth=0.3,
            )

        ax.set_yticks([STAGE_Y_POS[s] for s in range(5)])
        ax.set_yticklabels(STAGE_NAMES, fontsize=10, fontweight='bold')
        ax.set_ylim(-0.8, 7.8)
        ax.invert_yaxis()
        ax.set_xlim(0, max(time_hours[-1], 0.01))
        ax.set_xlabel("Time (hours)", fontsize=11)
        ax.set_ylabel("Sleep Stage", fontsize=11)
        ax.grid(axis='x', which='major', color='#CCCCCC', linewidth=0.5, alpha=0.7)

        # Current stage indicator
        latest = int(self.predictions[-1])
        ax.text(0.99, 0.95, f"Current: {STAGE_NAMES[latest]}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=14, fontweight="bold", color=STAGE_COLORS[latest],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))

        ax.set_title(f"{self.title}  —  {total_min:.0f} min",
                     fontsize=12, fontweight="bold")
        plt.tight_layout()


# ===========================================================================
# Convenience
# ===========================================================================

def generate_report(
    predictions: List[int], output_path: str,
    epoch_sec: int = 30, title: str = "Sleep Report",
):
    """Save a sleep sleep report to file."""
    fig = build_sleep_report(predictions, epoch_sec, title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved to: {output_path}")


# ===========================================================================
# brainstim Experiment-compatible paradigm (uses framework.py)
# ===========================================================================

def sleep_report_paradigm(win, predictions=None, epoch_sec=30):
    """Render sleep hypnogram in a PsychoPy window — compatible with
    brainstim's Experiment.register_paradigm().

    Usage:
        ex = Experiment(win_size=(1400, 800))
        ex.register_paradigm("Sleep Report", sleep_report_paradigm,
                            predictions=[0,0,2,2,3,3,4,2,...])
        ex.run()
    """
    from psychopy import visual, event, core

    if predictions is None or len(predictions) == 0:
        text = visual.TextStim(win, "No predictions available.",
                               color="white", height=30)
        text.draw(); win.flip(); core.wait(1); return

    n = len(predictions)
    time_hours = np.arange(n) * epoch_sec / 3600
    max_h = max(time_hours[-1], 1)
    bar_width = max_h / max(n, 1) * win.size[0] * 0.9

    # Background
    bg = visual.Rect(win, width=win.size[0], height=win.size[1],
                     fillColor="#FAFAFA", lineColor=None)
    bg.draw()

    # Title
    title = visual.TextStim(win, "Sleep Report — MetaBCI",
                            pos=(0, win.size[1] / 2 - 30),
                            color="#333333", height=28, bold=True)
    title.draw()

    # Y-axis labels
    for stage_id, (name, y_pos) in enumerate(zip(STAGE_NAMES, [0, -60, -120, -180, -240])):
        label = visual.TextStim(win, name, pos=(-win.size[0] / 2 + 50, y_pos),
                                color=STAGE_COLORS[stage_id], height=22, bold=True)
        label.draw()

    # Draw hypnogram bars
    bar_h = 30
    for i in range(n):
        color = STAGE_COLORS.get(int(predictions[i]), "#888888")
        x_pos = -win.size[0] / 2 + 70 + (i + 0.5) * bar_width
        y_pos = -int(predictions[i]) * 60
        rect = visual.Rect(win, width=bar_width, height=bar_h,
                           pos=(x_pos, y_pos),
                           fillColor=color, lineColor=color, lineWidth=1)
        rect.draw()

    # Time axis
    for h in range(int(max_h) + 1):
        x_pos = -win.size[0] / 2 + 70 + (h / max_h) * (n * bar_width)
        tick = visual.TextStim(win, f"{h}h", pos=(x_pos, -win.size[1] / 2 + 40),
                               color="#666666", height=16)
        tick.draw()

    win.flip()
    core.wait(3)  # display for 3 seconds
    event.clearEvents()
