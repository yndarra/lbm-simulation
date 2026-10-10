"""Точка входа: расчёт обтекания цилиндра и построение картинок.

Расчёт долгий (десятки минут на CPU), поэтому он разделён на два шага:

    python main.py run     # посчитать и сохранить результаты в data/run.npz
    python main.py plot    # построить картинки по сохранённым результатам
    python main.py         # оба шага подряд

Параметры по умолчанию соответствуют варианту 20 лабораторной работы:
``u = 0.01 + 6·(0.1 − 0.01)/(N − 1)``, ``Re = 20 + 6·(1000 − 20)/(N − 1)``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from lbm import CylinderFlow, FlowParams
from lbm import visualization as vis

DATA = Path("data/run.npz")


def variant_params(n: int) -> FlowParams:
    """Параметры варианта ``n`` из методички (скорость и Re на входе)."""
    u = 0.01 + 6 * (0.1 - 0.01) / (n - 1)
    re = int(20 + 6 * (1000 - 20) / (n - 1))
    return FlowParams(u_inlet=u, reynolds=re)


def run(params: FlowParams, n_steps: int, frame_every: int, frames_from: int) -> None:
    """Посчитать течение и сохранить кадры завихренности и сигнал датчика."""
    sim = CylinderFlow(params)
    # Датчик — точка на 2 диаметра за цилиндром, чуть выше оси.
    probe = (params.cx + 4 * params.radius, params.cy + params.radius // 2)

    print(f"Сетка {params.nx}×{params.ny}, u = {params.u_inlet:.4f}, Re = {params.reynolds:g}, "
          f"ω = {params.omega:.4f}, шагов: {n_steps}")
    probe_signal = np.empty(n_steps, dtype=np.float32)
    frames, frame_steps = [], []
    started = time.perf_counter()
    for k in range(n_steps):
        sim.step()
        probe_signal[k] = sim.velocity_at(*probe)[1]
        if sim.time >= frames_from and sim.time % frame_every == 0:
            frames.append(sim.vorticity().astype(np.float32))
            frame_steps.append(sim.time)
        if sim.time % 5000 == 0:
            elapsed = time.perf_counter() - started
            print(f"  шаг {sim.time:>6} / {n_steps}  ({elapsed / 60:.1f} мин)")

    DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        DATA,
        vorticity=np.array(frames),
        frame_steps=np.array(frame_steps),
        speed=sim.speed().astype(np.float32),
        last_vorticity=sim.vorticity().astype(np.float32),
        probe=probe_signal,
        obstacle=sim.obstacle,
        u_inlet=params.u_inlet,
        reynolds=params.reynolds,
        radius=params.radius,
    )
    print(f"Результаты сохранены в {DATA}")


def plot(out: Path) -> None:
    """Картинки по сохранённым результатам."""
    d = np.load(DATA)
    obstacle = d["obstacle"]
    u, re, r = float(d["u_inlet"]), float(d["reynolds"]), int(d["radius"])
    n_steps = d["probe"].size

    # Шкала цвета завихренности — по 99-му перцентилю, чтобы пики у самой
    # стенки цилиндра не «съедали» контраст вихрей в следе.
    vort_max = float(np.percentile(np.abs(d["vorticity"]), 99))
    vis.plot_fields(
        d["speed"], d["last_vorticity"], obstacle,
        f"Flow past a cylinder, Re = {re:g}, step {n_steps}",
        out / "fields.png", speed_max=2.2 * u, vort_max=vort_max,
    )
    # Каждый второй кадр: анимация остаётся плавной, а файл — вдвое меньше.
    vis.save_vorticity_gif(
        d["vorticity"][::2], obstacle, out / "vortex_street.gif", vort_max, fps=12
    )

    # Частота схода вихрей. Берём вторую половину сигнала (после переходного
    # процесса), период считаем по пересечениям среднего снизу вверх — это
    # точнее, чем по пику спектра, у которого разрешение всего 1/длина.
    signal = d["probe"][n_steps // 2:].astype(float)
    signal -= signal.mean()
    ups = np.flatnonzero((signal[:-1] < 0) & (signal[1:] >= 0))
    period = float(np.mean(np.diff(ups)))
    peak = 1 / period
    strouhal = peak * 2 * r / u
    # Спектр — для иллюстрации; дополняем нулями, чтобы кривая была гладкой.
    padded = np.zeros(8 * signal.size)
    padded[:signal.size] = signal * np.hanning(signal.size)
    spectrum = np.abs(np.fft.rfft(padded))
    freqs = np.fft.rfftfreq(padded.size, d=1.0)
    print(f"Частота схода вихрей: {peak:.3e} 1/шаг, период {period:.0f} шагов "
          f"({ups.size - 1} периодов), число Струхаля St = f·D/U = {strouhal:.3f}")
    vis.plot_probe(np.arange(n_steps), d["probe"], freqs, spectrum, peak, strouhal,
                   out / "strouhal.png")
    print(f"Картинки сохранены в {out.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="LBM D2Q9 flow past a cylinder")
    parser.add_argument("action", nargs="?", default="all", choices=["all", "run", "plot"])
    parser.add_argument("--variant", type=int, default=20, help="номер варианта (5, 10, 15, 20)")
    parser.add_argument("--steps", type=int, default=40_000, help="число шагов по времени")
    parser.add_argument("--frame-every", type=int, default=100, help="шаг записи кадров")
    parser.add_argument("--frames-from", type=int, default=20_000,
                        help="с какого шага записывать кадры для анимации")
    parser.add_argument("--out", type=Path, default=Path("images"), help="папка для картинок")
    args = parser.parse_args()

    if args.action in ("all", "run"):
        run(variant_params(args.variant), args.steps, args.frame_every, args.frames_from)
    if args.action in ("all", "plot"):
        plot(args.out)


if __name__ == "__main__":
    main()
