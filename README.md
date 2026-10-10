# Lattice Boltzmann Simulation of Flow Past a Cylinder

[![tests](https://github.com/yndarra/lbm-simulation/actions/workflows/tests.yml/badge.svg)](https://github.com/yndarra/lbm-simulation/actions/workflows/tests.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**English** | [Русский](README.ru.md)

A vectorized NumPy implementation of the **lattice Boltzmann method (LBM)** on a
D2Q9 lattice. It simulates flow past a circular cylinder in a channel, where a
**Kármán vortex street** forms behind the cylinder, and measures the vortex
shedding frequency (Strouhal number).

<p align="center">
  <img src="images/vortex_street.gif" width="720" alt="Vorticity field showing the Kármán vortex street">
</p>
<p align="center"><sub>Vorticity, steps 20 000 to 40 000. Red is counter-clockwise rotation, blue is clockwise.</sub></p>

> Lab work for the *Computer Modelling* university course.

## Method

LBM does not solve the Navier–Stokes equations directly. It tracks the kinetics
of fictitious particles: every lattice node stores 9 distribution functions
`fᵢ`, one for each discrete velocity `cᵢ` (at rest, 4 axis neighbors and 4
diagonal neighbors). Each time step consists of two operations:

1. **Collision** (BGK, single relaxation time): `fᵢ ← fᵢ − ω·(fᵢ − fᵢᵉᑫ)`
2. **Streaming**: each `fᵢ` moves to the neighboring node along `cᵢ`

Density and velocity are moments of the distributions: `ρ = Σ fᵢ` and
`ρu = Σ fᵢcᵢ`. Viscosity comes from the Reynolds number:
`ν = u·r / Re` and `ω = 1 / (3ν + ½)`.

| Boundary | Condition |
|---|---|
| Inlet (left) | Prescribed velocity, Zou–He scheme |
| Outlet (right) | Zero gradient for the unknown incoming populations |
| Top and bottom | Periodic |
| Cylinder | Bounce-back (no-slip) |

The inlet velocity carries a tiny `10⁻⁴` sinusoidal perturbation. Without it the
flow is perfectly symmetric, and the vortex street would only appear much later,
triggered by round-off errors.

### Compared with the original version

* **Inlet bug fixed.** In the Zou–He condition, the diagonal incoming
  populations were paired with the wrong outgoing directions, so the momentum
  balance at the inlet was broken. The pairs are now built from an explicit
  `OPPOSITE` table, and a unit test checks them.
* **About 2.4× faster.** A step takes about 10 ms instead of 24 ms on the
  420 × 180 grid, thanks to array layout `(9, nx, ny)`, vectorized equilibrium
  and bounce-back limited to the obstacle nodes.
* **Much less data.** The original notebook stored every field as three
  240 MB `.npy` files. Now only vorticity frames and a probe signal are saved,
  in a compressed ~1 MB `.npz`.

## Results

The default parameters are variant 20 of the assignment:

| Parameter | Value |
|---|---|
| Grid | 420 × 180 |
| Cylinder | center (105, 90), radius 20 |
| Inlet velocity `u` | 0.0384 (lattice units) |
| Reynolds number (by radius) | 329 |
| Relaxation `ω` | 1.972 |
| Steps | 40 000 (6.6 min on a laptop CPU) |

![Velocity and vorticity fields](images/fields.png)

For the first ~15 000 steps the wake stays symmetric. Then it becomes unstable,
and vortices of alternating sign start shedding from the top and bottom of the
cylinder.

![Probe signal and spectrum](images/strouhal.png)

The transverse velocity `u_y` at a probe two diameters downstream oscillates
with a period of **≈ 4150 steps**. That gives a Strouhal number

```
St = f·D / U ≈ 0.25
```

which is close to the classic value of about 0.2 for a cylinder. It is slightly
higher here, as expected, because the domain is periodic in `y`. The flow
effectively sees a row of cylinders only 4.5 diameters apart, and that
confinement speeds up the flow around each one. Only four full periods fit into
the averaging window, so this estimate is rough.

## Getting started

```bash
git clone https://github.com/yndarra/lbm-simulation.git
cd lbm-simulation
python -m venv .venv
.venv/Scripts/activate        # Windows; on Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

python main.py run            # simulate and save data/run.npz (≈ 7 min)
python main.py plot           # draw the figures from the saved data
python main.py                # both steps
python main.py run --variant 10 --steps 60000
```

To run the tests:

```bash
pip install -r requirements-dev.txt
pytest
```

The tests check the D2Q9 weights and opposite directions, the moments of the
equilibrium distribution, and that a short simulation stays stable, nearly
incompressible and shows a wake behind the cylinder.

## Project structure

```
lbm/
├── solver.py         # D2Q9 lattice, equilibrium, FlowParams, CylinderFlow
└── visualization.py  # field plots, probe spectrum, vorticity GIF
tests/                # lattice and solver tests
main.py               # run / plot entry point
```

## References

* T. Krüger et al. *The Lattice Boltzmann Method: Principles and Practice.* Springer, 2017.
* Q. Zou, X. He. *On pressure and velocity boundary conditions for the lattice Boltzmann BGK model.* Phys. Fluids 9, 1591 (1997).
* [Palabos cylinder example](https://palabos.unige.ch/), the classic reference setup for this problem.

## License

[MIT](LICENSE)
