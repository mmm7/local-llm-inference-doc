
#!/usr/bin/env python3

import csv
import math
import matplotlib
matplotlib.use("svg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator


CSV_FILE = "hardware.csv"


def number(row, *keys):
    """Return the first available numeric value, or None."""
    for key in keys:
        value = (row.get(key) or "").strip()
        if value:
            try:
                result = float(value)
                if math.isfinite(result):
                    return result
            except ValueError:
                pass
    return None


def load_data(filename=CSV_FILE):
    with open(filename, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No data rows found in {filename}")

    if "hardware" not in rows[0]:
        raise ValueError("CSV must contain a 'hardware' column")

    return rows


def label_style(name, label_offsets):
    """Return (dx, dy, ha, va) in points for a hardware label."""
    dx, dy = (label_offsets or {}).get(name, (6, 6))
    ha = "right" if dx < 0 else "left"
    va = "top" if dy < 0 else "bottom"
    return dx, dy, ha, va


def plain_log_tick(value, _pos):
    """Format log-axis ticks as plain numbers (not 10^n)."""
    if not math.isfinite(value) or value <= 0:
        return ""
    if value >= 1000:
        return f"{value:.0f}"
    if abs(value - round(value)) < 1e-9 * max(value, 1):
        return f"{value:.0f}"
    return f"{value:g}"


def apply_plain_log_ticks(ax):
    """More major log ticks, labeled with ordinary numbers."""
    majors = (1.0, 2.0, 3.0, 5.0, 7.0)
    ax.xaxis.set_major_locator(LogLocator(base=10, subs=majors))
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=majors))
    ax.xaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(plain_log_tick))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=tuple(range(2, 10))))


def make_plot(rows, x_keys, y_keys, xlabel, ylabel,
              title, filename, log_log=False, label_offsets=None):
    points = []

    for row in rows:
        x = number(row, *x_keys)
        y = number(row, *y_keys)

        if x is None or y is None:
            continue
        if log_log and (x <= 0 or y <= 0):
            continue
        points.append((x, y, row["hardware"]))

    if not points:
        print(f"Skipping {filename}: no complete data points")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    x = [p[0] for p in points]
    y = [p[1] for p in points]

    ax.scatter(
        x, y,
        s=80,
        color="#2878B5",
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )

    for xv, yv, name in points:
        dx, dy, ha, va = label_style(name, label_offsets)
        ax.annotate(
            name, (xv, yv),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=8,
            ha=ha,
            va=va,
        )

    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
        apply_plain_log_ticks(ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35, which="both" if log_log else "major")
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(filename, format="svg")
    plt.close(fig)

    print(f"Created {filename} ({len(points)} points)")


def add_usd_per_gb(rows):
    """Derive USD/GB from price and memory capacity where both are present."""
    for row in rows:
        price = number(row, "price_usd")
        memory = number(row, "memory_gb", "ram_gb")
        if price is not None and memory is not None and memory > 0:
            row["usd_per_gb"] = f"{price / memory:.6g}"
    return rows


def main():
    rows = add_usd_per_gb(load_data())

    # Accept both the old and new memory/bandwidth column names.
    bandwidth_label_offsets = {
        "NVIDIA RTX 3090": (8, -8),
        "MacBook Pro M5 Max 16-inch": (8, -8),
    }

    make_plot(
        rows,
        ("memory_gb", "ram_gb"),
        ("memory_bandwidth_gbs", "bandwidth_gbs"),
        "Memory capacity (GB)",
        "Memory bandwidth (GB/s)",
        "Memory Capacity vs. Memory Bandwidth",
        "hardware_bandwidth.svg",
        log_log=True,
        label_offsets=bandwidth_label_offsets,
    )

    make_plot(
        rows,
        ("fp32_tflops",),
        ("memory_bandwidth_gbs", "bandwidth_gbs"),
        "FP32 performance (TFLOP/s)",
        "Memory bandwidth (GB/s)",
        "FP32 vs. Memory Bandwidth",
        "hardware_fp32_bandwidth.svg",
    )

    make_plot(
        rows,
        ("fp16_tensor_tflops",),
        ("memory_bandwidth_gbs", "bandwidth_gbs"),
        "Dense FP16 Tensor performance (TFLOP/s)",
        "Memory bandwidth (GB/s)",
        "FP16 Tensor Performance vs. Memory Bandwidth",
        "hardware_fp16_bandwidth.svg",
    )

    make_plot(
        rows,
        ("price_usd",),
        ("fp16_tensor_tflops",),
        "Price (USD)",
        "Dense FP16 Tensor performance (TFLOP/s)",
        "Price vs. FP16 Tensor Performance",
        "hardware_price_performance.svg",
    )

    # Manual label offsets (points) for crowded USD/GB log-log positions.
    # Negative dx → label left of point; negative dy → below.
    usd_gb_label_offsets = {
        "Mac mini M6": (8, 0),
        "Mac mini M5 Pro": (8, 6),
        "MacBook Pro M5 Max 16-inch": (8, 6),
        "Mac Studio M5 Max": (8, -8),
        "Mac Studio M5 Ultra": (8, 6),
        "NVIDIA RTX 3090": (8, 6),
        "NVIDIA RTX 4090": (8, -8),
        "NVIDIA RTX PRO 6000 Blackwell Workstation": (-8, 8),
        "NVIDIA RTX 5090": (8, 0),
        "NVIDIA RTX PRO 5000 Blackwell": (8, -10),
        "NVIDIA H200 SXM": (-8, 0),
        "NVIDIA B200 SXM": (8, -8),
        "NVIDIA H100 SXM": (-8, -10),
    }

    make_plot(
        rows,
        ("usd_per_gb",),
        ("memory_bandwidth_gbs", "bandwidth_gbs"),
        "Price per GB (USD/GB)",
        "Memory bandwidth (GB/s)",
        "USD/GB vs. Memory Bandwidth",
        "hardware_usd_per_gb_bandwidth.svg",
        log_log=True,
        label_offsets=usd_gb_label_offsets,
    )


if __name__ == "__main__":
    main()
