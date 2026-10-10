"""Картинки и анимации полей течения."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colormaps
from PIL import Image


def _masked(field: np.ndarray, obstacle: np.ndarray) -> np.ndarray:
    """Поле с NaN в узлах цилиндра, транспонированное для imshow (x вправо)."""
    out = field.astype(float).copy()
    out[obstacle] = np.nan
    return out.T


def plot_fields(
    speed: np.ndarray,
    vorticity: np.ndarray,
    obstacle: np.ndarray,
    title: str,
    path: Path,
    speed_max: float,
    vort_max: float,
) -> None:
    """Модуль скорости и завихренность в один момент времени."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8.2))
    panels = (
        (speed, "viridis", 0, speed_max, "Velocity magnitude |u|"),
        (vorticity, "RdBu_r", -vort_max, vort_max, "Vorticity ω_z"),
    )
    for ax, (field, cmap, vmin, vmax, label) in zip(axes, panels, strict=True):
        cm = colormaps[cmap].copy()
        cm.set_bad("#3a3a3a")  # цилиндр
        im = ax.imshow(_masked(field, obstacle), origin="lower", cmap=cm, vmin=vmin, vmax=vmax)
        ax.set(xlabel="x, lattice units", ylabel="y")
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=label)
    fig.suptitle(title)
    fig.tight_layout()
    _save(fig, path)


def plot_probe(
    time: np.ndarray,
    signal: np.ndarray,
    freqs: np.ndarray,
    spectrum: np.ndarray,
    peak: float,
    strouhal: float,
    path: Path,
) -> None:
    """Сигнал поперечной скорости в точке за цилиндром и его спектр."""
    fig, (ax_t, ax_f) = plt.subplots(1, 2, figsize=(13, 4))
    ax_t.plot(time, signal, lw=0.8)
    ax_t.set(xlabel="Time step", ylabel="u_y at the probe",
             title="Transverse velocity behind the cylinder")
    ax_f.plot(freqs * 1000, spectrum)
    ax_f.axvline(peak * 1000, color="tab:red", ls="--",
                 label=f"f = {peak * 1000:.3f}·10⁻³ 1/step,  St = {strouhal:.3f}")
    ax_f.set(xlabel="Frequency, 10⁻³ 1/step", ylabel="Amplitude",
             title="Spectrum (after the transient)", xlim=(0, 5 * peak * 1000))
    for ax in (ax_t, ax_f):
        ax.grid(alpha=0.3)
    ax_f.legend()
    fig.tight_layout()
    _save(fig, path)


def save_vorticity_gif(
    frames: np.ndarray,
    obstacle: np.ndarray,
    path: Path,
    vort_max: float,
    fps: int = 20,
) -> None:
    """GIF завихренности. Кадры раскрашиваются напрямую, без matplotlib."""
    cmap = colormaps["RdBu_r"]
    images = []
    for frame in frames:
        norm = np.clip((frame + vort_max) / (2 * vort_max), 0, 1)
        rgba = cmap(norm)
        rgba[obstacle] = (0.23, 0.23, 0.23, 1.0)
        rgb = (rgba[..., :3] * 255).astype(np.uint8).transpose(1, 0, 2)[::-1]
        image = Image.fromarray(rgb).resize((rgb.shape[1] * 2, rgb.shape[0] * 2))
        # 64 цвета вместо 256 — GIF становится в несколько раз легче, а на
        # плавной палитре RdBu разница почти незаметна.
        images.append(image.quantize(colors=64, method=Image.Quantize.MEDIANCUT))
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=int(1000 / fps), loop=0, optimize=True)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
