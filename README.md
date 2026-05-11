[README.md](https://github.com/user-attachments/files/27609653/README.md)
# Wind Turbine Blade Optimization Using CAD and CFD

A computational design tool for optimizing 5MW wind turbine blades using Blade Element Momentum (BEM) theory, structural analysis, and multi-phase optimization algorithms.

## Overview

This project optimizes a 5MW wind turbine blade (NREL 5MW reference turbine) to maximize aerodynamic efficiency while satisfying structural constraints. Starting from a constant chord/twist baseline, the optimizer improves the power coefficient (Cp) from **0.355 → 0.505**, a **42% improvement**, while keeping all stresses within allowable limits.

## Results

| Parameter | Initial | Optimized | Change |
|-----------|---------|-----------|--------|
| Power Coefficient (Cp) | 0.355 | 0.505 | +42% |
| Power Output | 4.02 MW | 5.71 MW | +42% |
| Max Chord | 3.00 m | 5.00 m | +67% |
| Max Stress Ratio | 0.610 | 0.842 | — |

All stresses remain below allowable limits (ratio < 1.0) with a safety factor of 2.5.

## Turbine Specifications

- **Rated Power:** 5 MW
- **Rotor Diameter:** 126 m
- **Number of Blades:** 3
- **Rated Wind Speed:** 11.4 m/s
- **Airfoil:** NACA 64-A17
- **Material:** Epoxy-GFRP (Premium) — σ_tensile = 1100 MPa, σ_compress = 600 MPa

## Methodology

### Aerodynamic Analysis
Blade Element Momentum (BEM) theory with Prandtl tip loss correction. Each blade station is analyzed iteratively for axial and tangential induction factors, converging on lift/drag forces and power output.

### Structural Analysis
A hollow box-beam model estimates flapwise/edgewise bending moments, axial tensile loads from centrifugal force, and shear stresses. Stresses are checked against material allowables at all 20 span stations.

### Optimization — Two Phases
1. **Phase 1 (Rule-Based, 15 iterations):** Adjusts chord to relieve overstressed sections and pitch to target optimal angle of attack.
2. **Phase 2 (Genetic Algorithm, ~65 iterations):** BLX-α crossover + Gaussian mutation. Maximizes Cp with a penalty for constraint violations. Also supports derivative-based (SLSQP) and hybrid modes.

## Repository Structure

```
├── wind_turbine_blade_design_v4.py   # Main design and optimization script
├── blade_data.json                   # All design data (turbine, material, results)
├── initial_design.csv                # Baseline blade (constant chord/twist)
├── optimized_design.csv              # Optimized blade station-by-station data
├── optimization_history.csv          # Cp and stress ratio at every iteration
├── comparison_summary.csv            # Before/after summary
├── naca64a17_coords.csv              # NACA 64-A17 airfoil coordinates
├── blade_geometry.stl                # 3D blade geometry (STL)
├── blade_geometry.obj                # 3D blade geometry (OBJ)
└── blade_geometry.igs                # 3D blade geometry (IGES)
```

## Usage

```bash
pip install numpy scipy matplotlib
python wind_turbine_blade_design_v4.py
```

Output files are saved to `./output/`. Set `interactive=True` in `main()` to launch the matplotlib slider-based viewer for exploring blade stations.

### Optimization Mode

Edit `TurbineConfig` at the top of the script to switch methods:

```python
optimization_method = 'GA'          # Genetic Algorithm (default)
optimization_method = 'derivative'  # SLSQP / L-BFGS-B
optimization_method = 'hybrid'      # Derivative + GA polish
```

## License
MIT
