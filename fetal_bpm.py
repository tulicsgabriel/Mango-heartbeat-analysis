#!/usr/bin/env python3
"""
Analyze fetal heartbeat from a WAV file.

Usage:
    python fetal_bpm.py --plot path/to/heartbeat.wav
"""

import argparse
import csv
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile
from scipy.signal import hilbert, find_peaks


def load_wav_mono_normalized(wav_path):
    """Load, convert to mono, and normalize waveform to [-1, 1]."""
    fs, data = wavfile.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    data -= data.mean()
    max_abs = np.max(np.abs(data))
    if max_abs > 0:
        data /= max_abs
    return fs, data


def analyze_fetal_heartbeat(
    wav_path,
    max_expected_bpm=220,  # upper bound for fetal HR
):
    # --- Load audio ---
    fs, data = load_wav_mono_normalized(wav_path)

    duration_sec = len(data) / fs

    # --- Build envelope with Hilbert transform ---
    analytic = hilbert(data)
    envelope = np.abs(analytic)

    # Smooth envelope with a short moving average (50 ms)
    win_size = int(0.05 * fs)
    if win_size > 1:
        kernel = np.ones(win_size) / win_size
        envelope = np.convolve(envelope, kernel, mode="same")

    # --- Peak detection (heart beats) ---
    min_interval_sec = 60.0 / max_expected_bpm
    min_distance_samples = int(min_interval_sec * fs * 0.8)

    prom_thresh = 0.3 * np.percentile(envelope, 95)

    peaks, _ = find_peaks(
        envelope,
        distance=max(min_distance_samples, 1),
        prominence=prom_thresh,
    )

    if len(peaks) < 5:
        raise RuntimeError(
            f"Not enough beats detected after trimming edges (found {len(peaks)})."
        )

    # --- Compute times & intervals for ALL beats ---
    beat_times_all = peaks / fs  # seconds
    rr_all = np.diff(beat_times_all)  # intervals between consecutive beats

    # --- TRIM FIRST AND LAST BEATS / INTERVALS ---
    # Drop first and last beats for stats (keep only "inner" beats)
    inner_peaks = peaks[1:-1]
    inner_beat_times = beat_times_all[1:-1]

    # For intervals: drop first and last interval (they touch edges)
    if len(rr_all) < 3:
        raise RuntimeError("Not enough intervals for trimmed stats.")
    rr = rr_all[1:-1]

    # --- Instantaneous HR (for trimmed intervals) ---
    inst_bpm = 60.0 / rr
    avg_bpm = float(np.mean(inst_bpm))
    median_bpm = float(np.median(inst_bpm))
    std_bpm = float(np.std(inst_bpm, ddof=1))

    min_bpm = float(np.min(inst_bpm))
    max_bpm = float(np.max(inst_bpm))

    # --- HRV-like metrics (from trimmed intervals) ---
    sdnn = float(np.std(rr, ddof=1))  # overall interval spread (s)
    rmssd = float(
        np.sqrt(np.mean(np.diff(rr) ** 2))
    )  # beat-to-beat jitter (s)
    period_mean = float(np.mean(rr))
    period_median = float(np.median(rr))
    period_std = float(np.std(rr, ddof=1))
    period_cv = float(period_std / period_mean) if period_mean > 0 else np.nan

    # --- Amplitude per beat (from envelope) ---
    beat_amplitudes_all = envelope[peaks]
    inner_amplitudes = beat_amplitudes_all[1:-1]

    amp_mean = float(np.mean(inner_amplitudes))
    amp_median = float(np.median(inner_amplitudes))
    amp_std = float(np.std(inner_amplitudes, ddof=1))
    amp_min = float(np.min(inner_amplitudes))
    amp_max = float(np.max(inner_amplitudes))
    amp_cv = float(amp_std / amp_mean) if amp_mean > 0 else np.nan

    results = {
        "duration_sec": float(duration_sec),
        "num_beats_total": int(len(peaks)),
        "num_beats_used": int(len(inner_peaks)),
        "beat_times_sec": inner_beat_times,  # times of beats used (s)
        "inst_bpm": inst_bpm,  # HR from trimmed intervals
        "avg_bpm": avg_bpm,
        "median_bpm": median_bpm,
        "std_bpm": std_bpm,
        "min_bpm": min_bpm,
        "max_bpm": max_bpm,
        "period_mean_sec": period_mean,
        "period_median_sec": period_median,
        "period_std_sec": period_std,
        "period_cv": period_cv,
        "sdnn_sec": sdnn,
        "rmssd_sec": rmssd,
        "beat_amplitudes": inner_amplitudes,  # amplitudes for inner beats
        "amp_mean": amp_mean,
        "amp_median": amp_median,
        "amp_std": amp_std,
        "amp_min": amp_min,
        "amp_max": amp_max,
        "amp_cv": amp_cv,
    }
    return results


def plot_results(results, wav_path, out_prefix="fetal_heartbeat"):
    """
    Simple, easy-to-interpret plots that are SAVED to disk:
      1) Instantaneous HR vs time  -> <out_prefix>_hr.png
      2) Beat amplitude vs time    -> <out_prefix>_amp.png
    """
    beat_times = np.asarray(results["beat_times_sec"])
    amps = np.asarray(results["beat_amplitudes"])
    inst_bpm = np.asarray(results["inst_bpm"])
    fs, waveform = load_wav_mono_normalized(wav_path)
    time_axis = np.arange(len(waveform)) / fs

    # For HR, intervals sit between beats → use midpoints for the x-axis
    if len(beat_times) >= 2:
        hr_times = (beat_times[1:] + beat_times[:-1]) / 2.0
    else:
        hr_times = np.arange(len(inst_bpm))

    # --- Plot instantaneous HR ---
    plt.figure()
    plt.plot(hr_times, inst_bpm, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Instantaneous heart rate (bpm)")
    plt.title("Fetal heart rate vs time")
    plt.grid(True)
    hr_path = f"{out_prefix}_hr.png"
    plt.savefig(hr_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Plot amplitude per beat ---
    plt.figure()
    plt.plot(beat_times, amps, marker="o")
    plt.xlabel("Time (s)")
    plt.ylabel("Beat amplitude (relative)")
    plt.title("Beat amplitude vs time")
    plt.grid(True)
    amp_path = f"{out_prefix}_amp.png"
    plt.savefig(amp_path, dpi=150, bbox_inches="tight")
    plt.close()

    # --- Waveform + spectrogram (Praat-like stack) ---
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1]},
    )
    axes[0].plot(time_axis, waveform, color="tab:blue", linewidth=0.8)
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Waveform")
    axes[0].grid(True, alpha=0.3)

    axes[1].specgram(
        waveform,
        Fs=fs,
        NFFT=1024,
        noverlap=768,
        cmap="magma",
        scale="dB",
    )
    axes[1].set_ylabel("Frequency (Hz)")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_title("Spectrogram")

    wave_spec_path = f"{out_prefix}_wave_spec.png"
    fig.tight_layout()
    fig.savefig(wave_spec_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return hr_path, amp_path, wave_spec_path


def save_metrics_to_csv(results, wav_path):
    """Persist a single-row CSV with the summary metrics."""
    base_name = os.path.splitext(os.path.basename(wav_path))[0]
    wav_timestamp = datetime.fromtimestamp(os.path.getmtime(wav_path))
    timestamp_label = wav_timestamp.strftime("%Y%m%d_%H%M%S")
    csv_name = f"{base_name}_{timestamp_label}_metrics.csv"
    csv_path = os.path.join(os.path.dirname(wav_path), csv_name)

    metrics_row = {
        "wav_file": os.path.basename(wav_path),
        "wav_timestamp": wav_timestamp.isoformat(),
        "duration_sec": results["duration_sec"],
        "num_beats_total": results["num_beats_total"],
        "num_beats_used": results["num_beats_used"],
        "avg_bpm": results["avg_bpm"],
        "median_bpm": results["median_bpm"],
        "std_bpm": results["std_bpm"],
        "min_bpm": results["min_bpm"],
        "max_bpm": results["max_bpm"],
        "period_mean_sec": results["period_mean_sec"],
        "period_median_sec": results["period_median_sec"],
        "period_std_sec": results["period_std_sec"],
        "period_cv": results["period_cv"],
        "sdnn_sec": results["sdnn_sec"],
        "rmssd_sec": results["rmssd_sec"],
        "amp_mean": results["amp_mean"],
        "amp_median": results["amp_median"],
        "amp_std": results["amp_std"],
        "amp_min": results["amp_min"],
        "amp_max": results["amp_max"],
        "amp_cv": results["amp_cv"],
    }

    fieldnames = list(metrics_row.keys())
    with open(csv_path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(metrics_row)

    return csv_path


def main():
    parser = argparse.ArgumentParser(
        description="Estimate fetal heart rate from a WAV file."
    )
    parser.add_argument("wav_path", help="Path to the WAV file")
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save simple plots (HR vs time, amplitude vs time) as PNG files",
    )
    args = parser.parse_args()

    results = analyze_fetal_heartbeat(args.wav_path)
    # results = analyze_fetal_heartbeat("mango_heartbeat_20225_nov20.wav")

    print(f"Clip duration                : {results['duration_sec']:.2f} s")
    print(f"Detected beats (total)       : {results['num_beats_total']}")
    print(f"Beats used for stats (inner) : {results['num_beats_used']}")
    print()
    print("Heart rate (from inner beats):")
    print(f"  Mean HR                    : {results['avg_bpm']:.1f} bpm")
    print(f"  Median HR                  : {results['median_bpm']:.1f} bpm")
    print(f"  Std HR                     : {results['std_bpm']:.1f} bpm")
    print(
        f"  Min / Max HR               : {results['min_bpm']:.1f} / {results['max_bpm']:.1f} bpm"
    )
    print()
    print("Beat period T (inner intervals):")
    print(f"  Mean T                     : {results['period_mean_sec']:.3f} s")
    print(f"  Median T                   : {results['period_median_sec']:.3f} s")
    print(f"  Std T                      : {results['period_std_sec']:.3f} s")
    print(f"  CV T (std/mean)            : {results['period_cv']:.3f}")
    print()
    print("Beat-interval variation (HRV-like measures):")
    print(
        f"  SDNN  (overall timing spread)   : {results['sdnn_sec']:.3f} s"
    )
    print(
        f"  RMSSD (beat-to-beat jitter)     : {results['rmssd_sec']:.3f} s"
    )
    print()
    print("Beat amplitude (inner beats, 0–1 scale):")
    print(f"  Mean amplitude              : {results['amp_mean']:.3f}")
    print(f"  Median amplitude            : {results['amp_median']:.3f}")
    print(f"  Std amplitude               : {results['amp_std']:.3f}")
    print(
        f"  Min / Max amplitude         : {results['amp_min']:.3f} / {results['amp_max']:.3f}"
    )
    print(f"  CV (std/mean)               : {results['amp_cv']:.3f}")

    csv_path = save_metrics_to_csv(results, args.wav_path)
    print()
    print(f"Metrics saved to             : {csv_path}")

    if args.plot:
        base = os.path.splitext(os.path.basename(args.wav_path))[0]
        hr_path, amp_path, wave_spec_path = plot_results(
            results, args.wav_path, out_prefix=base
        )
        print()
        print("Plots saved:")
        print(f"  Heart rate vs time    -> {hr_path}")
        print(f"  Amplitude vs time     -> {amp_path}")
        print(f"  Wave + spectrogram    -> {wave_spec_path}")


if __name__ == "__main__":
    main()
