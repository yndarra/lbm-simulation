"""Решёточный метод Больцмана (LBM) D2Q9 для обтекания цилиндра.

Вместо уравнений Навье–Стокса моделируется кинетика «частиц» на решётке:
в каждом узле хранятся 9 функций распределения ``f_i`` — доли частиц,
движущихся с одной из 9 решёточных скоростей ``c_i`` (D2Q9: покой, 4 соседа
по сторонам и 4 по диагоналям). Один шаг по времени состоит из двух этапов:

1. **Столкновение** (BGK, одно время релаксации): распределения в каждом
   узле релаксируют к локальному равновесию
   ``f_i ← f_i − ω·(f_i − f_i^eq)``.
2. **Перенос** (streaming): каждая ``f_i`` сдвигается в соседний узел по
   своему направлению ``c_i``.

Макроскопические плотность и скорость — моменты распределений:
``ρ = Σ f_i``, ``ρ·u = Σ f_i·c_i``, давление ``p = ρ·c_s²``, ``c_s² = 1/3``.
Вязкость задаётся через число Рейнольдса: ``ν = u·r / Re``,
``ω = 1 / (3ν + 1/2)``.

Граничные условия:

* слева — заданная скорость (условие Зоу–Хе);
* справа — «мягкий» выход: неизвестные распределения копируются из
  предпоследнего столбца (нулевой градиент);
* сверху и снизу — периодические;
* на цилиндре — отражение (bounce-back): частица, влетевшая в стенку,
  разворачивается назад, что даёт условие прилипания ``u = 0``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Решёточные скорости D2Q9. Порядок выбран так, чтобы противоположное
# направление к i было (i + 4) по модулю 8 (кроме покоя).
C = np.array([
    [0, 0],                                   # 0: покой
    [1, 0], [1, 1], [0, 1], [-1, 1],          # 1–4
    [-1, 0], [-1, -1], [0, -1], [1, -1],      # 5–8
])
# Веса: 4/9 для покоя, 1/9 для осевых, 1/36 для диагональных направлений.
W = np.array([4 / 9, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36, 1 / 9, 1 / 36])
# Индекс противоположного направления: OPPOSITE[i] = j, если c_j = −c_i.
OPPOSITE = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4])

# Группы направлений по знаку x-компоненты скорости — для граничных условий.
RIGHT = np.flatnonzero(C[:, 0] == 1)    # [1, 2, 8] — летят вправо
MIDDLE = np.flatnonzero(C[:, 0] == 0)   # [0, 3, 7]
LEFT = np.flatnonzero(C[:, 0] == -1)    # [4, 5, 6] — летят влево
# Пары для Зоу–Хе: каждому направлению из RIGHT — строго противоположное.
# (В исходной версии проекта диагональные направления были сопоставлены
# неправильно, из-за чего входное условие нарушало баланс импульса.)
RIGHT_OPPOSITES = OPPOSITE[RIGHT]


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """Равновесные распределения (разложение Максвелла до второго порядка).

    ``f_i^eq = w_i·ρ·(1 + 3(c_i·u) + 9/2·(c_i·u)² − 3/2·u²)``

    Возвращает массив формы ``(9, nx, ny)``.
    """
    cu = C[:, 0, None, None] * ux + C[:, 1, None, None] * uy
    usq = ux**2 + uy**2
    return W[:, None, None] * rho * (1 + 3 * cu + 4.5 * cu**2 - 1.5 * usq)


@dataclass
class FlowParams:
    """Параметры задачи в решёточных единицах (шаг сетки и времени = 1)."""

    nx: int = 420               # длина канала, узлов
    ny: int = 180               # ширина канала, узлов
    u_inlet: float = 0.04       # скорость на входе (должна быть ≪ c_s ≈ 0.577)
    reynolds: float = 150.0     # число Рейнольдса по радиусу цилиндра
    cx: int | None = None       # центр цилиндра; по умолчанию — nx/4, ny/2
    cy: int | None = None
    radius: int | None = None   # радиус цилиндра; по умолчанию — ny/9

    def __post_init__(self) -> None:
        if self.cx is None:
            self.cx = self.nx // 4
        if self.cy is None:
            self.cy = self.ny // 2
        if self.radius is None:
            self.radius = self.ny // 9

    @property
    def viscosity(self) -> float:
        """Кинематическая вязкость ``ν = u·r / Re``."""
        return self.u_inlet * self.radius / self.reynolds

    @property
    def omega(self) -> float:
        """Частота релаксации BGK ``ω = 1 / τ``, ``τ = 3ν + 1/2``.

        Для устойчивости нужно ``ω < 2``; чем ближе к 2, тем меньше запас.
        """
        return 1.0 / (3.0 * self.viscosity + 0.5)


class CylinderFlow:
    """Решатель LBM D2Q9 для течения в канале с цилиндром."""

    def __init__(self, params: FlowParams) -> None:
        self.p = params
        nx, ny = params.nx, params.ny
        x, y = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
        # Маска узлов, занятых цилиндром.
        self.obstacle = (x - params.cx) ** 2 + (y - params.cy) ** 2 <= params.radius**2

        # Входной профиль скорости с крошечным синусоидальным возмущением
        # (1e-4). Без него поток идеально симметричен, и вихревая дорожка
        # Кармана возникает лишь от ошибок округления — очень нескоро.
        self.inlet_ux = params.u_inlet * (1 + 1e-4 * np.sin(2 * np.pi * np.arange(ny) / (ny - 1)))

        rho = np.ones((nx, ny))
        ux = np.broadcast_to(self.inlet_ux, (nx, ny)).copy()
        uy = np.zeros((nx, ny))
        self.f = equilibrium(rho, ux, uy)
        self.time = 0

    # ------------------------------------------------------------------
    def macroscopic(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Плотность и компоненты скорости из распределений."""
        rho = self.f.sum(axis=0)
        ux = np.tensordot(C[:, 0], self.f, axes=1) / rho
        uy = np.tensordot(C[:, 1], self.f, axes=1) / rho
        return rho, ux, uy

    def step(self) -> None:
        """Один шаг по времени."""
        f = self.f

        # 1. Выход справа: распределения, летящие внутрь области (влево),
        #    берём из предпоследнего столбца.
        f[LEFT, -1, :] = f[LEFT, -2, :]

        # 2. Макроскопические величины.
        rho, ux, uy = self.macroscopic()

        # 3. Вход слева (Зоу–Хе): задаём скорость, а плотность восстанавливаем
        #    из известных распределений — покоящихся/вертикальных и летящих
        #    влево (они пришли изнутри области).
        ux[0, :] = self.inlet_ux
        uy[0, :] = 0.0
        rho[0, :] = (f[MIDDLE, 0, :].sum(axis=0) + 2 * f[LEFT, 0, :].sum(axis=0)) / (1 - ux[0, :])

        # 4. Равновесие и неизвестные входящие распределения: неравновесная
        #    часть летящих вправо берётся у противоположного направления.
        feq = equilibrium(rho, ux, uy)
        f[RIGHT, 0, :] = feq[RIGHT, 0, :] + f[RIGHT_OPPOSITES, 0, :] - feq[RIGHT_OPPOSITES, 0, :]

        # 5. Столкновение (BGK).
        f_out = f - self.p.omega * (f - feq)

        # 6. Отражение от цилиндра: в узлах препятствия разворачиваем
        #    распределения (до столкновения) в противоположные направления.
        f_out[:, self.obstacle] = f[:, self.obstacle][OPPOSITE]

        # 7. Перенос: сдвигаем каждое распределение вдоль своей скорости.
        for i, (cx, cy) in enumerate(C):
            f[i] = np.roll(f_out[i], shift=(cx, cy), axis=(0, 1))

        self.time += 1

    def run(self, n_steps: int, callback=None, every: int = 100) -> None:
        """Сделать ``n_steps`` шагов; ``callback(self)`` вызывается каждые ``every`` шагов."""
        for _ in range(n_steps):
            self.step()
            if callback is not None and self.time % every == 0:
                callback(self)

    # ------------------------------------------------------------------
    # Производные поля для визуализации
    # ------------------------------------------------------------------
    def velocity_at(self, i: int, j: int) -> tuple[float, float]:
        """Скорость в одном узле — дёшево, без расчёта полей целиком."""
        fij = self.f[:, i, j]
        rho = fij.sum()
        return float(C[:, 0] @ fij / rho), float(C[:, 1] @ fij / rho)

    def speed(self) -> np.ndarray:
        _, ux, uy = self.macroscopic()
        return np.hypot(ux, uy)

    def vorticity(self) -> np.ndarray:
        """Завихренность ``ω_z = ∂u_y/∂x − ∂u_x/∂y`` (центральные разности)."""
        _, ux, uy = self.macroscopic()
        return np.gradient(uy, axis=0) - np.gradient(ux, axis=1)
