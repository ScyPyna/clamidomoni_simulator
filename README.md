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