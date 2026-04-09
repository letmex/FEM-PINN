# TM Single-Notch Tensile Case

This example solves one fixed case:
- geometry: real pre-cracked plate (notch is geometric boundary)
- loading: uniaxial tensile displacement
- model: 2D plane-stress thermo-mechanical mixed-mode phase-field PINN
- mesh: pure triangular FE-style mesh

## Monolithic Single-Network Architecture

Implemented in [network.py](d:/ProgramData/PINN/pinn_fem/external_refs/phase-field-fracture-with-pidl/source/network.py):
- `MonolithicTMPhaseNet`
- input: `[x, y]`
- output: `[T_raw, ux_raw, uy_raw, d_raw]`
- one hidden-stack MLP (no trunk/head split, no branch parameter groups)
- optional geometric input normalization + adaptive activation

Physical mapping is implemented in [field_computation_tm.py](d:/ProgramData/PINN/pinn_fem/external_refs/phase-field-fracture-with-pidl/source/field_computation_tm.py):
- `map_temperature(T_raw -> T_phys)`
- `map_displacement([ux_raw, uy_raw] -> [ux_phys, uy_phys])`
- `map_phase_field(d_raw -> d_phys)` with irreversibility:
  `d = d_prev + (1-d_prev)*sigmoid(d_raw)`

## Training Flow

In [model_train.py](d:/ProgramData/PINN/pinn_fem/external_refs/phase-field-fracture-with-pidl/source/model_train.py):
1. thermal substep
2. mechanical substep
3. history candidate update
4. phase substep

All three optimizer substeps update the **same full network parameters**.

## History Variables

Primary history state is element-level only:
- `HI_elem`
- `HII_elem`
- `He_elem = HI_elem + (GcI/GcII)*HII_elem`

Node-level `HI/HII/He` is derived from element history only for output/visualization.

## Boundary Conditions

- mechanical:
  - top boundary: `uy = 4e-8 * t`
  - bottom: configurable (`uy_only` default or `uxuy`)
  - notch faces: traction-free natural boundary
- thermal:
  - initial `T = T0`
  - COMSOL-aligned default fallback: right boundary `T=T0`, others insulated
- phase:
  - natural zero-flux boundary
  - initial material-domain damage `d0 = 0`

Boundary extraction priority in [input_data_from_mesh.py](d:/ProgramData/PINN/pinn_fem/external_refs/phase-field-fracture-with-pidl/source/input_data_from_mesh.py):
1. Physical Groups from `.msh` (if available)
2. geometric fallback

## Outputs

`results/` contains:
- `field_data/field_step_XXXX.csv`
- `curves/reaction_displacement_macro_stress_strain.csv`
- `losses/loss_trace.csv`, `losses/loss_per_step.csv`
- `figures/` plots

Model checkpoints:
- `best_models/trained_unified_XXXX.pt`

## Run

```bash
python main.py
```

Short check:

```bash
python run_short_refactor_check.py
```
