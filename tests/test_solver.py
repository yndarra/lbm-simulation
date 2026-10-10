"""Тесты решателя LBM D2Q9."""

import numpy as np
import pytest

from lbm import CylinderFlow, FlowParams
from lbm.solver import OPPOSITE, RIGHT, RIGHT_OPPOSITES, C, W, equilibrium


def test_weights_sum_to_one():
    assert W.sum() == pytest.approx(1.0)


def test_opposite_directions():
    np.testing.assert_array_equal(C[OPPOSITE], -C)
    # Пары для условия Зоу–Хе действительно противоположны.
    np.testing.assert_array_equal(C[RIGHT_OPPOSITES], -C[RIGHT])


def test_equilibrium_moments():
    rng = np.random.default_rng(0)
    rho = 1 + 0.05 * rng.random((5, 4))
    ux = 0.05 * rng.standard_normal((5, 4))
    uy = 0.05 * rng.standard_normal((5, 4))
    feq = equilibrium(rho, ux, uy)
    np.testing.assert_allclose(feq.sum(axis=0), rho)
    np.testing.assert_allclose(np.tensordot(C[:, 0], feq, axes=1), rho * ux)
    np.testing.assert_allclose(np.tensordot(C[:, 1], feq, axes=1), rho * uy)


def test_relaxation_parameter():
    p = FlowParams(u_inlet=0.04, reynolds=100, radius=20)
    assert p.viscosity == pytest.approx(0.04 * 20 / 100)
    assert p.omega == pytest.approx(1 / (3 * p.viscosity + 0.5))
    assert 0 < p.omega < 2


def small_flow(steps):
    sim = CylinderFlow(FlowParams(nx=120, ny=50, u_inlet=0.04, reynolds=40))
    sim.run(steps)
    return sim


def test_flow_stays_finite_and_near_incompressible():
    sim = small_flow(1500)
    rho, ux, uy = sim.macroscopic()
    assert np.all(np.isfinite(rho))
    # В слабосжимаемом режиме LBM плотность отклоняется от 1 на проценты.
    assert np.abs(rho - 1).max() < 0.05


def test_no_slip_on_cylinder_and_wake_behind_it():
    sim = small_flow(1500)
    p = sim.p
    speed = sim.speed()
    u_in = p.u_inlet
    # Прямо за цилиндром скорость гораздо меньше, чем во входящем потоке.
    assert speed[p.cx + p.radius + 2, p.cy] < 0.5 * u_in
    # Сбоку от цилиндра поток ускоряется (сужение канала).
    assert speed[p.cx, p.cy + p.radius + 3] > u_in
