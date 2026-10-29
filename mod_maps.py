"""Flat annular maps for the modulated-light dataset (2026-10-22).

Five rings with the inner radius fixed at 150 um and the outer radius at 245, 290, 335, 380 and
420 um (the paper's r_in and its two r_out, plus three steps in between). The light is a flat
plateau with soft edges (an erf of width EDGE_UM), so the only geometric variable is the ring size.

The plateau is 1.1e-4 uW/um^2, ten times below the lab rings. At the lab intensity f_R is
saturated: A between 0.1 and 1 changes the plateau tumbling rate by only 0.13 s^-1 and moves the
threshold contour by ~15 um, so most of the A range would give the same dynamics. At 1.1e-4 the
threshold (5.43e-5) is crossed at A ~ 0.5 and the whole range 0-1 matters.

Output keeps the simulator's format and grid (1159 x 1159 px of 1.041667 um, 371 cells at rho
0.02): tbr_fields_mod/rout<R>/{ill,tbr}_map_ring_<P>muW.dat, with P the total power of the ring
in uW, plus a copy of tumblingVsEnergyDensity.txt, which MODULATED runs read at start-up.
Set power_input to P and point the run's tbr_fields at that folder.
"""
import json
import shutil
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
from scipy.special import erf

BASE = Path("/home/user/Scrivania/pulsed_simulation")
TUMB = BASE/"inputSimulations"/"tumblingVsEnergyDensity.txt"
OUTD = BASE/"tbr_fields_mod"
PIX = 1.041667              # um per pixel, as every previous map
HALF = 579                  # 2*579+1 = 1159 px, the grid of ring_180muW
R_IN = 150.0
R_OUTS = (245, 290, 335, 380, 420)
EDGE_UM = 10.0              # width of the soft edges, about one cell diameter
PLATEAU = 1.1e-4            # uW/um^2 at A = 1
THR = 3.0708352             # working reversal threshold, only for the printed summary

td = np.genfromtxt(TUMB, skip_header=1)
tumble_converter = interp1d(td[:, 2], td[:, 1], kind="slinear", fill_value="extrapolate")


def annulus(r, r_out):
    s = np.sqrt(2)*EDGE_UM
    return 0.5*(erf((r - R_IN)/s) - erf((r - r_out)/s))


ix = np.arange(-HALF, HALF + 1)*PIX
X, Y = np.meshgrid(ix, ix)
R = np.sqrt(X**2 + Y**2)
summary = {}
for r_out in R_OUTS:
    ill = PLATEAU*annulus(R, r_out)
    tbr = tumble_converter(ill)
    power = int(round(ill.sum()*PIX**2))
    folder = OUTD/f"rout{r_out}"
    folder.mkdir(parents=True, exist_ok=True)
    for kind, m in (("ill", ill), ("tbr", tbr)):
        with open(folder/f"{kind}_map_ring_{power}muW.dat", "w") as f:
            f.write(f"{len(ix)} {len(ix)} {ill.mean()} {tbr.mean()} {PIX}\n")
            np.savetxt(f, m)
    shutil.copy(TUMB, folder/TUMB.name)
    above = R[tbr > THR]
    summary[r_out] = dict(power_uW=power, plateau_tbr=round(float(tumble_converter(PLATEAU)), 4),
                          above_threshold_um=[round(float(above.min()), 1), round(float(above.max()), 1)])
    print(r_out, summary[r_out])

assert len({s["power_uW"] for s in summary.values()}) == len(R_OUTS), "power labels must differ"
json.dump(dict(r_in_um=R_IN, edge_um=EDGE_UM, plateau_uW_um2=PLATEAU, rings=summary),
          open(OUTD/"summary.json", "w"), indent=1)
