# phase-field-fracture-with-pidl

Code associated with the paper "[Phase-field modeling of fracture with physics-informed deep learning](https://www.sciencedirect.com/science/article/pii/S0045782524003608)".

## Repository Layout
- `source/`: core PINN modules (network, training, energy/loss, utilities).
- `examples/`: problem-specific setups.
- `examples/TM_single_notch_tensile/`: 2D thermo-mechanical mixed-mode phase-field PINN case for a single-notch plate under uniaxial tension.

## Added Thermo-Mechanical Mixed-Mode Case
The new case extends the original project skeleton to solve:
- transient thermal conduction (`T`)
- quasi-static thermo-elastic displacement (`ux`, `uy`)
- mixed-mode phase-field damage (`d`)
- explicit history variables (`HI`, `HII`)
- total mixed-mode driving force (`He`)

with explicit thermo-mechanical network coupling and time stepping:
`T_{n+1} -> u_{n+1}(x,y,T_{n+1}) -> He_{n+1}^{cand} -> d_{n+1} -> update(HI,HII)`.

## Network Coupling
The TM example now uses a monolithic single network (`source/network.py: MonolithicTMPhaseNet`):
- input: `[x, y]`
- output: `[T_raw, ux_raw, uy_raw, d_raw]`
- one shared hidden MLP, no branch-specific heads or parameter-group training

Physical fields are constructed in `source/field_computation_tm.py` via:
- `map_temperature(T_raw -> T_phys)`
- `map_displacement([ux_raw,uy_raw] -> [ux_phys,uy_phys])`
- `map_phase_field(d_raw,d_prev -> d_phys)`

## Step Convergence
For each time step, inner fixed-point iterations are executed and the following are monitored before advancing:
- relative total loss change
- relative field change of `T`, `u`, `d`
- relative history change of `HI`, `HII`

Controlled in `examples/TM_single_notch_tensile/config.py` via:
- `max_inner_iters`, `conv_patience`
- `tol_loss`, `tol_field_T`, `tol_field_u`, `tol_field_d`, `tol_hist`
- `require_converged_step`

## Note On Default Thermal BC
The report-default case imposes `T=T0` on selected thermal boundaries with `T(x,0)=T0`, so thermal evolution can be weak.
This is expected from the boundary setup and does not imply missing thermo-mechanical coupling in code.

## Run
From the repository root:

```bash
cd examples/TM_single_notch_tensile
python main.py
```

Optional network override:

```bash
python main.py <hidden_layers> <neurons> <seed> <activation> <init_coeff>
```

## Outputs
Under `examples/TM_single_notch_tensile/<run_dir>/results/`:
- `field_data/field_step_XXXX.csv`: nodal `x,y,T,ux,uy,d,HI,HII,He`
- `curves/reaction_displacement_macro_stress_strain.csv`
- `figures/`: field maps + curve plots + loss plots + energy plots
- `losses/loss_trace.csv`: iteration-level loss trace
- `losses/loss_per_step.csv`: step-level loss and physics summary
- `losses/loss_step_XXXX.csv`: per-step summary
- `losses/loss_step_XXXX_iter.csv/.npy/.png`: per-step iterative branch losses

Model weights are saved in:
- `best_models/trained_unified_XXXX.pt`
