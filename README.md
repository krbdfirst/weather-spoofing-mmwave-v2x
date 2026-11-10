# Weather Spoofing Attacks on Rain-Adaptive mmWave Band Selection in V2X

Simulation and analysis code for the paper:

> R. Bello, I. Yusuf, J. Adjei Owusu, O. Aiyewunmi, G. Comert, J. Mwakalonge, E. Abuhdima,
> A. Mrebit, R. Ataai, V. Varghese, "Weather Data Spoofing Attacks on Rain-Adaptive
> Millimeter-Wave Frequency Selection in V2X Communication Networks," IEEE WiSEE 2026.

The simulation output used in the paper is archived on Zenodo:
[doi:10.5281/zenodo.24097635](https://doi.org/10.5281/zenodo.24097635).

## Overview

A weather-aware controller selects the sidelink carrier band (73, 60, 39 or 28 GHz, with a
5 GHz fallback) from reported rainfall. An attacker who falsifies the rain report controls
the selected band without transmitting on the radio channel. The scenario runs in ns-3
MilliCar (NR V2X sidelink, 3GPP V2V-Urban channel with beamforming and blockage), so the
selected band is the simulated carrier. Rain attenuation follows ITU-R P.838 and always acts
on the true rain rate; only the band selector sees the reported rate.

| Scenario | Rain report | Carrier |
|---|---|---|
| `baseline` | honest | controller-selected |
| `attackA` (Attack Type 1, force-up) | clear sky | 73 GHz |
| `attackB` (Attack Type 2, force-down) | heavy rain | 5 GHz |

Two bandwidth regimes are simulated: `fixwidth` (100 MHz for every band) and `realwidth`
(5 GHz: 20 MHz; 28/39/60/73 GHz: 100/200/400/400 MHz).

## Files

| File | Purpose | Paper output |
|---|---|---|
| `ns3/weather-v2x-millicar.cc` | ns-3 scenario: rain attenuation, band controller, attack scenarios, rain-cell field, CSV loggers | all simulation data |
| `run_millicar_matrix.sh` | Phase 1 sweep: scenario x rain {0,10,20,30,37.5,50} mm/h x regime x 10 seeds (360 runs) | `results/` |
| `run_rainfield_matrix.sh` | Phase 2 sweep: advecting EXCELL rain cell, 4 cell sizes x 3 rain rates x 10 seeds (360 runs) | `results_rainfield/` |
| `analyze_millicar.py` | PRR, reliable range, SINR, MCS, transport-block size and latency | Section V numbers |
| `make_paper_figures.py` | Paper figures at print size | PRR heatmap; range and long-range PRR vs rain |
| `defense_sinr_consistency.py` | SINR-consistency detector, ROC and mitigation | defense figure and detection numbers |
| `analyze_rainfield.py` | Effect of the rain cell on the attack and the detector | rain-field results |

## Requirements

- ns-3.36.1 with the MilliCar module (ns3-mmwave 6.0)
- Python 3 with numpy, pandas and matplotlib (`pip install -r requirements.txt`)

## Reproducing the results

### Option A: analysis only, from the archived data

1. Download `phase1_uniform_rain.zip` and `phase2_rain_cell.zip` from the
   [Zenodo record](https://doi.org/10.5281/zenodo.24097635).
2. Unzip both into this folder, which creates `results/` and `results_rainfield/`.
3. Run:

```bash
python3 analyze_millicar.py
python3 make_paper_figures.py
python3 defense_sinr_consistency.py
python3 analyze_rainfield.py
```

Figures are written to `results/figures/` and `paper/figures/` (PNG). Numerical results are
printed to the terminal.

### Option B: full rerun

1. Copy `ns3/weather-v2x-millicar.cc` into the `scratch/` folder of your ns3-mmwave tree and
   build it.
2. Point the sweep scripts at that tree and run them:

```bash
export NS3_DIR=/path/to/ns3-mmwave
JOBS=8 ./run_millicar_matrix.sh
JOBS=8 ./run_rainfield_matrix.sh
```

Both scripts skip runs whose output already exists, so an interrupted sweep can be resumed.
Then run the analysis as in Option A.

## Output format

Each run writes three CSV files, tagged `s<seed>_<regime>_<scenario>_<rain>` (Phase 2 adds
`_c<cellfactor>_<direction>`):

```
pkts : txrx,time,src,dst,seq,bytes,latency_ms,dist_m,x,y
phy  : time,rxnode,txnode,dist_m,sinr_db,numsym,tbsize,mcs
meta : regime,scenario,rain,band_ghz,bw_mhz,numveh,simtime
```

## Citation

If you use this code, please cite the paper above and the dataset
(doi:10.5281/zenodo.24097635). See `CITATION.cff`.

## License

MIT (see `LICENSE`). The scenario links against ns-3, which is licensed under GPLv2.
