# Clamidomoni: fork with pulsed, modulated and speckle light

Fork of [Davidelama/Clamidomoni](https://github.com/Davidelama/Clamidomoni). The model is the same; this fork adds
light that changes in time, a settable seed, and the map makers of its two main campaigns, pulsed light and speckle
light. Only the simulator and what is needed to build its input maps are versioned here: analysis scripts and
simulation output are not. The parameter reference of the original README follows below (sections 1–16).

## What the fork adds to the simulator

Every new key is optional. With a key absent (or 0) the simulator behaves as before.

| Keys | What they do | Section |
|---|---|---|
| `seed_offset` | Shifts the random seed: the RNG gets `seed_offset + sim_number`. Without it, with `NperCore : 1` every run uses seed 0, so "different seeds" give identical runs. | 16 |
| `PULSED`, `light_on_time`, `light_off_time`, `pulse_start_time`, `LIGHT_START_OFF`, `dark_tbr` | Light switched on and off in cycles. | 14 |
| `MODULATED`, `modulation_id`, `modulation_update` | Light map multiplied by a smooth A(t) read from `modulation.txt`. | 16 |
| `t_stop`, `log_interval`, `print_skip_exp` | Run length and frame interval in seconds, instead of powers of `Nblock`. | 16 |
| `steer_kappa`, `D_r_override`, `omega_chiral` | Known steering, rotational noise and chirality, to test trajectory analyses. | 15 |

## Layout and scripts

```text
Clamidomoni.cpp, Clamidomoni.h, rapidyaml.h   the simulator
parameters_multi.yaml                         default parameters (continuous light)
inputSimulations/                             ring maps: profiles, calibration and the scripts that build the maps
pulsed/                                       pulsed light: example parameters
pulsed/modulated/                             modulated light: flat rings, A(t) curves, example parameters
speckle/                                      speckle light: map maker, example parameters
```

**`inputSimulations/`** (run the scripts from the repository root: their paths start with `inputSimulations/`)

| File | What it does |
|---|---|
| `ill_maker.py` | Original map maker. From a radial profile `radialEnergyDensityMap[_large_ring]_<P>µW_good.txt` it builds the 2D light map of a radially symmetric ring and, through `tumblingVsEnergyDensity.txt`, its tumbling-rate map. Settings at the top of the file: `intensity` and `newint` (µW), `large`, `linear`. Writes `tbr_fields/ill_map*.dat` and `tbr_fields/tbr_map*.dat`. |
| `linearizer.py` | Original. Makes the linear-profile version of the small ring (`linearEnergyDensityMap_180µW_good.txt`, plus the S-model one), which `ill_maker.py` reads with `linear=True`. Inputs: `illumination_lin_180µW.tiff`, `fluoTest2.tiff`, `radialEnergyDensityMap_180µW.txt`. |
| `lab_maps.py` | Maps from the lab's own illumination images, one TIFF per session (`SESSIONS` at the top), scaled to that session's power. Unlike the earlier profile extraction, it finds the ring centre in the image (rings are off-centre by up to ~80 px), subtracts the camera background and averages over all angles. Writes `tbr_fields_lab/ill<S>/{ill,tbr}_map_ring_<P>muW.dat`: set `power_input : P` and point the run's `tbr_fields/` to that folder. Paths to set at the top: `BASE`, `TIFFS`, `TUMB`. |
| `tumblingVsEnergyDensity.txt` | Calibration f_R(I): tumbling rate against light energy density. Every map maker reads it. For `MODULATED` runs, copy it into the run's `tbr_fields/` as well. |
| `radialEnergyDensityMap_180µW_good.txt` | Radial profile of the small ring, 180 µW (the ring of the pulsed grid). |
| `radialEnergyDensityMap_large_ring_610µW_good.txt` | Radial profile of the large ring, 610 µW. |
| `radialEnergyDensityMap_180µW.txt`, `illumination_lin_180µW.tiff`, `fluoTest2.tiff` | Inputs of `linearizer.py`. |

**`pulsed/`**

| File | What it does |
|---|---|
| `parameters_pulsed_example.yaml` | A real run of the pulsed grid: small ring at 180 µW, τ_m = 16 s (`time_mem 160000` steps of 1e-4 s), 10 s on and 5 s off, 7 min (`t_stop 420`), one frame every 0.1 s. Maps from `inputSimulations/ill_maker.py`, or from `lab_maps.py` for lab rings. Its `v_drift_*` keys are not read by this simulator and do nothing. |
| `modulated/mod_maps.py` | Five flat rings for modulated light: inner radius 150 µm, outer radius 245, 290, 335, 380 or 420 µm, plateau 1.1e-4 µW/µm² with soft (erf) edges, so the ring size is the only geometric variable. Writes `tbr_fields_mod/rout<R>/{ill,tbr}_map_ring_<P>muW.dat`. Paths at the top: `BASE`, `TUMB`, `OUTD`. |
| `modulated/make_modulated.py` | Builds the run folders of the modulated dataset (5000 runs). Each run gets its own A(t) on 0–120 s every 0.1 s: a Gaussian process with a random correlation time, squashed to [0, 1] and starting from 1, written to `<run>/modulation.txt`; rings assigned as `id % 5`; τ_m 10 s; `seed_offset = id`. Paths at the top: `BASE`, `OUT`. |
| `modulated/parameters_modulated_example.yaml` | Run 0 of that dataset: ring with outer radius 245 µm, τ_m 10 s, map rebuilt every 0.1 s, 120 s. |

**`speckle/`**

| File | What it does |
|---|---|
| `speckle_ill_maker.py` | Speckle light maps on the simulation grid, replacing `ill_maker.py` for speckle. The field is fully developed speckle (exponential intensity at every grain size) and exactly periodic. The grain size ξ is the FWHM of the intensity autocorrelation. The map is normalised to a mean energy density (`--mean-int`, default 3.4e-5 µW/µm²). Options: `--grains` (µm), `--realisations`, `--match-peak`, `--homogeneous` (uniform control), `--outroot`. Writes `<outroot>/xi<ξ>_m<k>/{ill,tbr}_map_ring_<ξ>muW.dat` and `map_stats.json`. |
| `parameters_speckle_example.yaml` | ξ = 40 µm, realisation 0, τ_m 16 s, 15 min. Here `power_input` is the grain size: it only selects the map file (`..._40muW.dat`); the intensity is set by the map itself. |

## Running

```bash
g++ -O3 -std=c++17 -pthread Clamidomoni.cpp -o clamidomoni
python3 inputSimulations/ill_maker.py     # writes tbr_fields/ (run from the repository root)
./clamidomoni                             # reads parameters_multi.yaml and tbr_fields/ in the working directory, writes logs/
```

Each run normally gets its own folder with `parameters_multi.yaml`, a `tbr_fields/` folder or link, and `logs/`.

## History

- `main`: the original commits by Davide Lama; the PINN work of June 2026 (moved to a separate repository; its
  files only remain in the history); pulsed and modulated light (`6c0c997`); the steering keys.
- Tag `pinn-ring-dataset-2026-07`: the simulator that made the PINN ring datasets of July 2026. It takes the seed
  from the command line (`./clamidomoni <seed>`, folded in as `sim_number += seed*10000`) and has `v_drift_x/y`.
  The main branch handles the seed with `seed_offset` instead.

---

# Simulation of Unicellular Algae in Structured Light Fields

This repository contains a simulation framework for modeling unicellular algae moving in externally imposed light fields. The model supports run-and-tumble dynamics, quorum sensing, memory effects, gradient sensing, collision-induced tumbling, and an alternative Active Brownian Particle (ABP) model.

Below is a detailed description of all simulation parameters.

---

# 1. Particle Properties

### `part_diam : 10.0`

Particle diameter in **microns (µm)**.

### `active_speed : 50.0`

Self-propulsion speed in **µm/s**.

### `kBT : 4.141947`

Thermal energy in simulation units (SI scaled to microns, multiplied by 1e9).

### `eta : 1.0`

Dynamic viscosity (SI scaled to microns, ×1e9).

* `1.0` → water
* `2.1` → water/lutidine mixture

### `sub_dist : 0.5`

Distance of particle center from substrate, in units of particle diameters.

---

# 2. Time & Parallelization

### `dt : 0.0001`

Timestep in seconds.
(Use up to 9 decimal places if needed.)

### `NperCore : 1`

Number of simulations per CPU core.
Total simulations = `NperCore × number_of_cores`.

### `Nblock : 2`

### `Nlevel : 18`

Total number of timesteps:

```
Total steps = Nblock^Nlevel
```

For ~1 hour simulations: `Nlevel = 25`

---

# 3. Light Field Configuration

### `power_input : 610`

Average input power for generating the tumbling-rate map.

### Map Type Switches (0 = off, 1 = on)

* `HOMOGENEOUS` → Uniform intensity map
* `GAUSSIAN` → Gaussian intensity profile
* `LARGE` → Large ring geometry
* `LINEAR` → Linear gradient
* `SYMMETRIZED` → Symmetric with respect to ring maximum
* `EXPERIMENT` → Use experimental light map
* `SAME_PEAK` → Same peak height for large/thin ring (otherwise normalized by total intensity)

Map files must follow:

```
mot_fields//mot_field_0_"parameters"
```

Tags:

* `_ring` (default)
* `_gauss`
* `_large`
* `_lin`
* `_symm`
* `_samepeak`

To switch this field on and off periodically in time, see **§14 Pulsed Illumination**.

---

# 4. Boundary Conditions

### `PBC : 1`

* `0` → Hard walls
* `1` → Periodic boundary conditions

---

# 5. Particle Interaction Model

### `PUNCTIFORM : 0`

* `0` → Finite-size particles
* `1` → Point particles

### `WCA : 1`

* `1` → Weeks–Chandler–Andersen (purely repulsive)
* `0` → Lennard-Jones potential

### `LJepsilon : 50`

Lennard-Jones energy depth in units of `kBT`.
High propulsion speeds may require large values.

### `rho : 0.02`

Area fraction (density).

Notes:

* Single particle → no tag needed.
* Multi-particle → `_multi` tag required.
* High density (>20%) → `_hd` tag required.

### `NOCENTER : 400`

If nonzero: no particles initialized within this radius from the ring center.

---

# 6. Interaction Switches

### `INDEPENDENT : 0`

If `1`, removes:

* Collisions
* Speed reduction
* LJ/WCA interactions

### `COLLISIONS : 1`

If `1`, collisions influence tumbling rate.

### `v_reduction : 1`

Enable quorum-sensing-like speed reduction.

### `coll_tbr_rate : 5`

Tumbling rate when particles collide.

---

# 7. Run-and-Tumble Control

### Tumbling Dependencies

* `TBR_I : 1` → Tumbling depends on light intensity
* `TBR_THETA : 1` → Depends on angle to gradient
* `TBR_DIST : 0` → Depends on angle to center (instead of angle to gradient)

If `TBR_I = 0`, you should define the tumbling value with `avg_tbr ≠ 0`.

### `avg_tbr : 0`

Constant tumbling rate (if used).

### `total_tbr_multiplier : 1.0`

Global multiplier applied to all tumbling rates.

### `tbr_amplification : 1`

Amplification of experimental intensity effect (max ~3).

---

# 8. Memory Effects

### `time_mem : 160000`

Number of timesteps used to average tumbling rate.
Set to `0` to disable memory.

### `DELAY : 0`

If `1`, use oldest value instead of average over memory window.

### Threshold-Based Gradient Reversal

These are for the model with a pseudomemory of upper and lower thresholds

* `upper_mem_thresh`
* `lower_mem_thresh`

If `0`, feature disabled.

---

# 9. Gradient Sign & Adaptation

### `grad_sign_thresh : 3.0708352`

If nonzero, reverses gradient effect when TBR reaches threshold (depending on memory).

### `threshold_std : 0`

Gaussian-distributed variation of threshold across particles (recommended 0.02–0.1).

### `threshold_decay_rate : 0.0`

Exponential decay of threshold over time.

* Negative value → increase until 2× threshold.

### `grad_sign_dist : 0`

Distance within which gradient sign reverses.

### `grad_dependence : 0`

Linear scaling of tumbling rate with |gradient|.

### `grad_amplification : 15`

Amplifies experimental gradient effect (max 15).

### `grad_decay_rate : 0.0`

Exponential decay of gradient influence over time.

---

# 10. Light Response Timing

### `rec_jump : 0`

Timestep interval for light response updates.
`0` or `1` → continuous update.

---

# 11. Continuous Taxis Function

### `fk : 0.0`

Sigmoid width for continuous taxis function.

* `< 0.01` → similar to no sigmoid
* `0` → disables continuous sigmoid, get step-function

---

# 12. S-Model (Alternative Adaptive Model)

### `S_model : 0`

If `1`, enables S-model and disables:

* `time_mem`
* Memory thresholds
* `grad_sign_thresh`
* `grad_sign_dist`

Parameters:

* `S_tau : 1.0` → Decay time (s)
* `S_gamma : 0.00005`
* `S_st : 0.1`
* `S_stot : 1.0`

---

# 13. Active Brownian Particle (ABP) Model

### `ABP_model : 0`

If `1`, replaces run-and-tumble with torque-based ABP dynamics.

### `ABP_torque_fac : 0.0`

Torque strength multiplier in ABP model.

---

# 14. Pulsed Illumination (light on / light off)

Switches the light field on and off periodically, with intervals chosen from input.
While the light is **on**, particles read the tumbling-rate map as usual.
While the light is **off**, the light field is simply absent: every particle sees a
uniform tumbling rate `dark_tbr` and a **zero gradient**, so `TBR_THETA` has nothing to
act on. Nothing else changes — swimming speed, collisions and the WCA/LJ interactions
are unaffected by the light.

Everything downstream keeps working during the dark phases: `time_mem` goes on averaging
(over the dark value), the `grad_sign_thresh` state keeps being updated, and in the
S-model the illumination seen by the particle drops to zero so `S` relaxes with `S_tau`.

### `PULSED : 0`

* `0` → light permanently on (identical to the previous behaviour)
* `1` → periodic on/off cycling

### `light_on_time : 10.0`

Duration in **seconds** of each illuminated interval.

### `light_off_time : 10.0`

Duration in **seconds** of each dark interval.

Both are converted to timesteps through `dt`; an interval shorter than `dt` is reported
as a warning and never fires.

### `pulse_start_time : 0.0`

Seconds held in the starting state before the cycling begins, e.g. to let the system
equilibrate under constant light. `0` starts the cycling immediately.

### `LIGHT_START_OFF : 0`

* `0` → the cycle (and the initial hold) starts in the light
* `1` → it starts in the dark

### `dark_tbr : 1.75`

Tumbling rate in **1/s** during the dark phases. The default `1.75` is the experimental
value at zero energy density in `inputSimulations//tumblingVsEnergyDensity.txt`, i.e. the
value the map itself takes far away from the illuminated region. A **negative** value
makes the code use the minimum of the loaded tumbling map instead. If
`tbr_amplification != 1`, this rate is amplified exactly like the map is.

### Output

Log files gain a `_puls<on>on<off>off` tag, so pulsed runs never overwrite continuous
ones, and an **eighth column** holding the state of the light (`1` on, `0` off) at each
logged time. The first seven columns are unchanged.

The map on disk is *not* affected by pulsing: the same `tbr_fields//tbr_map*` files are
used, and `ill_maker.py` needs no change.

### Interaction with other switches

* `TBR_I : 0` makes the tumbling rate independent of the map, so pulsing then only acts
  through the S-model. Use `TBR_I : 1` for pulsing to affect run-and-tumble.
* `rec_jump > 0` means particles only look at the light every `rec_jump` steps, so the
  switching they perceive is sampled at that rate. Keep `rec_jump` well below
  `light_on_time/dt` and `light_off_time/dt`.
* `COLLISIONS : 1` still forces `coll_tbr_rate` on colliding particles during the dark
  phases: collisions do not depend on the light.

---

# 15. Steering, Angular Noise and Chirality (testing keys)

Three optional keys added on 2026-11-18 to test the trajectory analysis against a known ground truth.
All default to 0, which keeps the original model: with the keys absent or 0 the logs are byte-identical
to the previous build.

### `steer_kappa : 0.0`

Continuous steering towards the light gradient between tumbles, in **rad/s**:
`phi += steer_kappa * (cos(phi)*tratey - sin(phi)*tratex) / gmax * dt`, i.e. kappa · sin(theta) · |grad tbr| / gmax.
`gmax` is the 99.9th percentile of |grad tbr| on the map (difference across 10 pixels, one cell diameter);
it is printed at start and appended to header line 2. Measured response: 0.92 of nominal (particle stencil).

### `D_r_override : 0.0`

If non-zero, replaces the thermal rotational diffusion D_r (0.00115 rad²/s) in `Br`, in **rad²/s**.

### `omega_chiral : 0.0`

Constant chiral rotation between tumbles, `phi += omega_chiral * dt`, in **rad/s**.

---

# 16. Seed, Modulated Light and Run Length

### `seed_offset : 0`

Added to the random seed of every run (the RNG gets `seed_offset + sim_number`; `sim_number` stays the index used
to average the moments). Appended to the log name as `_seed<N>` when larger than 0.

### `MODULATED : 0`

`1` multiplies the light map by a time-dependent factor A(t). The simulator reads `modulation.txt` from the working
directory (two columns: time in s, A) and `tbr_fields//tumblingVsEnergyDensity.txt`, and rebuilds the tumbling map
as f_R(A·I) every `modulation_update` seconds.

### `modulation_id : 0`

Label of the A(t) curve, appended to the log name as `_mod<id>`.

### `modulation_update : 0.1`

Interval, in seconds, between two rebuilds of the tumbling map.

### `t_stop : 0.0`

Run length in seconds. `0` keeps the `Nblock^Nlevel` steps.

### `log_interval : 0.0`

Seconds between two logged frames. `0` keeps `print_skip_exp` or the default (2048 frames). Do not set both.

### `print_skip_exp : 0`

Logs every `Nblock^print_skip_exp` steps instead of the default.

---

# Summary of Modeling Modes

| Mode                       | Description                       |
| -------------------------- | --------------------------------- |
| Run-and-tumble             | Default chemotaxis-style dynamics |
| Memory-based               | Uses time-averaged tumbling       |
| Threshold reversal         | Gradient sign switching           |
| Quorum sensing             | Density-dependent speed           |
| Collision-induced tumbling | Extra tumbling on contact         |
| S-model                    | Adaptive internal state model     |
| ABP model                  | Continuous torque-based steering  |
| Pulsed light               | Periodic on/off illumination      |

---

# Notes

* Units are microns and seconds unless otherwise stated.
* High propulsion speeds may require stronger repulsion (high `LJepsilon`).
* Carefully verify map tagging (`_multi`, `_hd`) for dense systems.
* Ensure consistency between tumbling switches and thresholds.

---

This simulation framework allows systematic exploration of light-controlled collective behavior in motile unicellular algae across density regimes, interaction strengths, and adaptive response models.


# Generation of illumination and tumbling rate maps

The script `ill_maker.py` generates spatial maps for simulations starting from **experimental illumination maps**. The script produces:

- An **illumination field map**
- A **tumbling-rate field map**
- Corresponding **parameter configuration files**

The generated outputs can be used in particle-based simulations, active matter models, or light-driven dynamics studies.

---

# Features

- Load experimental energy density maps
- Generate illumination maps from experimental data
- Generate tumbling-rate maps coupled to illumination
- Support for multiple illumination profiles:
  - Ring illumination
  - Gaussian illumination
  - Linear illumination
  - Large-ring illumination
- Optional conversion to an S-model representation
- Adjustable simulation intensity independent of experimental intensity
- Automatic parameter export

# Parameters

The script is controlled through a set of user-defined parameters that specify the illumination type, scaling, input/output paths, and map conversion options.

```python
# Experimental illumination intensity (ring profile)
intensity = 610

# Intensity used for the generated simulation map
# Can differ from the experimental intensity
newint = 610

# Use Gaussian illumination profile
gaussian = False

# Symmetrize the illumination map
# Currently unused
symmetrized = False

# Generate illumination maps for the S-model
Smodel = False

# Conversion factor from standard units to S-model units
factor = 1e6 * 0.2517

# Use linear illumination profile
linear = False

# Use large-ring experimental illumination profile
large = True

# Pixel size in micrometers
pix = 1.041667

# Input experimental illumination map folder
inputname = "inputSimulations//radialEnergyDensityMap"

# Output folder for generated illumination map
outputnamei = "tbr_fields//ill_map"

# Output folder for generated tumbling-rate map
outputnamet = "tbr_fields//tbr_map"

# Preserve peak normalization between maps (instead of overall intensity)
samepeak = False

# Additional normalization parameter
# Currently unused
normalisationPower = 0

---

# Repository Structure

```text
.
├── inputSimulations/
│   └── radialEnergyDensityMap*
│
├── tbr_fields/
│   ├── ill_map*
│   └── tbr_map*
│
└── generate_maps.py