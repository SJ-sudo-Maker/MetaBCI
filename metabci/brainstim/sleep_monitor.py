# -*- coding: utf-8 -*-
"""
SleepMonitorUI: sleep report visualization for offline and real-time use.

Displays hypnogram, stage distribution, and sleep statistics.
"""

from typing import Optional, List, Dict
import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Stage colors (AASM convention)
STAGE_COLORS = {
    0: "#FF6B6B",  # W  — red
    1: "#FFD93D",  # N1 — yellow
    2: "#6BCB77",  # N2 — green
    3: "#4D96FF",  # N3 — blue
    4: "#9B59B6",  # R  — purple
}

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_NAMES_FULL = {
    0: "Wake",
    1: "N1 (Light)",
    2: "N2 (Intermediate)",
    3: "N3 (Deep)",
    4: "REM",
}


# ---------------------------------------------------------------------------
# Core report generator (no GUI dependency)
# ---------------------------------------------------------------------------

def build_sleep_report(
    predictions: List[int],
    epoch_sec: int = 30,
    title: str = "Sleep Report",
) -> Figure:
    """Generate a sleep report figure from a list of stage predictions.

    Parameters
    ----------
    predictions : list of int
        Sleep stage labels per epoch (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    epoch_sec : int
        Seconds per epoch (default 30).
    title : str
        Report title.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The completed report figure.
    """
    n_epochs = len(predictions)
    if n_epochs == 0:
        raise ValueError("Empty predictions list")

    total_minutes = n_epochs * epoch_sec / 60
    stages_unique, counts = np.unique(predictions, return_counts=True)
    stage_count = {i: 0 for i in range(5)}
    for s, c in zip(stages_unique, counts):
        stage_count[int(s)] = c

    # Derived metrics
    sleep_onset_epochs = _find_sleep_onset(predictions)
    sleep_latency_min = sleep_onset_epochs * epoch_sec / 60
    sleep_period_epochs = n_epochs - sleep_onset_epochs
    waso = sum(1 for i in range(sleep_onset_epochs, n_epochs)
               if predictions[i] == 0) * epoch_sec / 60  # W after sleep onset

    tst_min = (n_epochs - sleep_onset_epochs - waso * 60 / epoch_sec) * epoch_sec / 60
    tst_min = max(0, tst_min)

    sleep_efficiency = (
        (tst_min / (total_minutes - sleep_latency_min) * 100)
        if total_minutes > sleep_latency_min else 0
    )

    # ---- Build figure ----
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[2, 1],
                          hspace=0.35, wspace=0.3)

    # (A) Hypnogram (top-left, spans full height)
    ax_hypno = fig.add_subplot(gs[:, 0])
    _draw_hypnogram(ax_hypno, predictions, epoch_sec, stage_count)

    # (B) Stage distribution pie (top-right)
    ax_pie = fig.add_subplot(gs[0, 1])
    _draw_stage_pie(ax_pie, stage_count, n_epochs)

    # (C) Metrics table (bottom-right)
    ax_table = fig.add_subplot(gs[1, 1])
    ax_table.axis("off")
    _draw_metrics_table(ax_table, total_minutes, sleep_latency_min,
                        tst_min, sleep_efficiency, stage_count, n_epochs, epoch_sec)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=0.98)
    return fig


def _find_sleep_onset(predictions: List[int]) -> int:
    """First epoch of 3 consecutive non-Wake epochs (standard definition)."""
    for i in range(len(predictions) - 2):
        if all(p != 0 for p in predictions[i:i + 3]):
            return i
    return 0


def _draw_hypnogram(ax, predictions, epoch_sec, stage_count):
    """Draw sleep stage hypnogram."""
    n = len(predictions)
    time_axis = np.arange(n) * epoch_sec / 3600  # hours

    # Draw as filled regions
    prev = 0
    for e in range(n):
        color = STAGE_COLORS.get(int(predictions[e]), "#888888")
        ax.fill_between([time_axis[prev], time_axis[min(e + 1, n - 1)]],
                        5.5, -0.5, color=color, alpha=0.85)
        prev = e

    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
    ax.set_ylim(-0.5, 5.5)
    ax.invert_yaxis()
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Sleep Stage")
    ax.set_title("Hypnogram", fontweight="bold")

    # Duration annotation
    dur_min = n * epoch_sec / 60
    info = (f"Total: {dur_min:.0f} min | "
            f"W: {stage_count[0]} | REM: {stage_count[4]} | "
            f"N1+N2+N3: {stage_count[1]+stage_count[2]+stage_count[3]}")
    ax.text(0.5, -0.12, info, transform=ax.transAxes, ha="center",
            fontsize=8, color="gray")

    # Grid
    ax.grid(axis="x", alpha=0.3)


def _draw_stage_pie(ax, stage_count, n_epochs):
    """Draw stage distribution pie chart."""
    labels = []
    sizes = []
    colors = []
    for i in range(5):
        if stage_count[i] > 0:
            labels.append(f"{STAGE_NAMES[i]} ({100*stage_count[i]/n_epochs:.1f}%)")
            sizes.append(stage_count[i])
            colors.append(STAGE_COLORS[i])

    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, colors=colors, autopct="%1.1f%%",
        startangle=90, pctdistance=0.6,
        textprops={"fontsize": 9},
    )
    ax.legend(wedges, labels, loc="lower center", ncol=2, fontsize=8)
    ax.set_title("Stage Distribution", fontweight="bold")


def _draw_metrics_table(ax, total_min, latency_min, tst_min,
                        efficiency, stage_count, n_epochs, epoch_sec):
    """Draw sleep metrics summary table."""
    # Calculate percentages
    n1_pct = 100 * stage_count[1] / max(n_epochs, 1)
    n3_pct = 100 * stage_count[3] / max(n_epochs, 1)
    rem_pct = 100 * stage_count[4] / max(n_epochs, 1)

    nrem_min = (stage_count[1] + stage_count[2] + stage_count[3]) * epoch_sec / 60

    metrics = [
        ("Recording Duration", f"{total_min:.1f} min"),
        ("Sleep Latency", f"{latency_min:.1f} min"),
        ("Total Sleep Time (TST)", f"{tst_min:.1f} min"),
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
            ax.text(0.05, y, label, transform=ax.transAxes,
                    fontsize=9, va="top")
            ax.text(0.95, y, value, transform=ax.transAxes,
                    fontsize=9, va="top", ha="right", fontweight="bold",
                    color="#2C3E50")
        y -= 0.075


# ---------------------------------------------------------------------------
# Real-time monitor (non-blocking window)
# ---------------------------------------------------------------------------

class SleepMonitorUI:
    """Real-time sleep stage monitor window.

    Opens a matplotlib window that updates as new predictions arrive.
    Suitable for integration with SleepOnlineWorker or offline playback.

    Parameters
    ----------
    epoch_sec : int
        Seconds per epoch (default 30).
    title : str
        Window title.

    Examples
    --------
    >>> monitor = SleepMonitorUI()
    >>> monitor.open()
    >>> for stage in [0, 0, 2, 2, 3, 3, 4, 2]:  # streaming predictions
    ...     monitor.update(stage)
    ...     time.sleep(0.1)  # simulate real-time
    >>> monitor.close()
    """

    def __init__(self, epoch_sec: int = 30,
                 title: str = "Sleep Monitor — MetaBCI"):
        self.epoch_sec = epoch_sec
        self.title = title
        self.predictions: List[int] = []
        self._fig: Optional[Figure] = None
        self._ani = None

    # ---- Public API ----

    def open(self, block: bool = False):
        """Open the monitor window.

        Parameters
        ----------
        block : bool
            If True, blocks until window is closed. Default False (non-blocking).
        """
        plt.ion()
        self._fig = plt.figure(figsize=(12, 5))
        self._fig.canvas.manager.set_window_title(self.title)
        self._render_empty()
        if block:
            plt.show(block=True)
        else:
            plt.show(block=False)
            plt.pause(0.1)

    def update(self, stage: int):
        """Push a new sleep stage prediction (0-4) and refresh the display.

        Parameters
        ----------
        stage : int
            Sleep stage: 0=W, 1=N1, 2=N2, 3=N3, 4=REM.
        """
        if stage not in range(5):
            raise ValueError(f"Stage must be 0-4, got {stage}")
        self.predictions.append(stage)
        if self._fig is not None:
            self._render_current()
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()

    def close(self):
        """Close the monitor window."""
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
        plt.ioff()

    def save(self, path: str):
        """Save the current report to a file (PNG/PDF/SVG).

        Parameters
        ----------
        path : str
            Output file path. Extension determines format.
        """
        fig = build_sleep_report(self.predictions, self.epoch_sec,
                                 title=self.title)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---- Internal ----

    def _render_empty(self):
        """Render initial empty state."""
        if self._fig is None:
            return
        self._fig.clear()
        ax = self._fig.add_subplot(111)
        ax.text(0.5, 0.5, "Waiting for data...\n\n"
                "SleepOnlineWorker will push predictions here.",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=14, color="gray")
        ax.axis("off")
        self._fig.canvas.draw_idle()

    def _render_current(self):
        """Render current hypnogram in the monitor window."""
        if self._fig is None or len(self.predictions) == 0:
            return
        self._fig.clear()

        n = len(self.predictions)
        time_axis = np.arange(n) * self.epoch_sec / 60  # minutes
        total_min = n * self.epoch_sec / 60

        ax = self._fig.add_subplot(111)
        for i in range(n):
            color = STAGE_COLORS.get(int(self.predictions[i]), "#888888")
            ax.fill_between(
                [time_axis[i], time_axis[min(i + 1, n - 1)]],
                5.5, -0.5, color=color, alpha=0.85,
            )

        # Latest stage indicator
        latest = int(self.predictions[-1])
        ax.text(0.99, 0.95, f"Current: {STAGE_NAMES[latest]}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=14, fontweight="bold",
                color=STAGE_COLORS[latest],
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                         alpha=0.9))

        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["W", "N1", "N2", "N3", "REM"])
        ax.set_ylim(-0.5, 5.5)
        ax.invert_yaxis()
        ax.set_xlabel("Time (minutes)")
        ax.set_ylabel("Sleep Stage")
        ax.set_title(f"{self.title} — {total_min:.0f} min recorded",
                     fontweight="bold")
        ax.grid(axis="x", alpha=0.3)


# ---------------------------------------------------------------------------
# Quick command-line report
# ---------------------------------------------------------------------------

def generate_report(
    predictions: List[int],
    output_path: str,
    epoch_sec: int = 30,
    title: str = "Sleep Report",
):
    """Generate and save a sleep report from predictions.

    Parameters
    ----------
    predictions : list of int
        Sleep stage labels (0=W, 1=N1, 2=N2, 3=N3, 4=REM).
    output_path : str
        File path for the report image (e.g. "report.png").
    epoch_sec : int
        Seconds per epoch.
    title : str
        Report title.
    """
    fig = build_sleep_report(predictions, epoch_sec, title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Report saved to: {output_path}")
