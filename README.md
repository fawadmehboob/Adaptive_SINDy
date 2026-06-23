# Adaptive SINDy

Adaptive SINDy is a UAV disturbance-rejection and trajectory-tracking project built around sparse system identification of residual forces and adaptive control in windy environments.

This repository contains the simulation-side code for two SITL workflows:

- ArduPilot SITL with Gazebo Harmonic
- Crazyflie SITL with Gazebo Harmonic

It also contains an offline SINDy model-fitting notebook for identification experiments.

This work is based on the same framework as the paper:

- [Adaptive SINDy: Residual Force System Identification Based UAV Disturbance Rejection](https://arxiv.org/abs/2603.08863)

## Repository Layout

```text
Adaptive_SINDy/
├── Ardupilot_SITL/
├── Crazyflie_SITL/
├── SINDy/
│   └── Quad_Sindy_Windy.ipynb
├── DATA/
├── results/
│   ├── ardupilot/
│   └── crazyflie/
└── README.md
```

## What Each Folder Contains

- `Ardupilot_SITL/`: controller and wind/disturbance scripts for the ArduPilot-based simulation setup
- `Crazyflie_SITL/`: controller, wind model, and supporting files for the Crazyflie-based simulation setup
- `SINDy/`: offline identification work, including the notebook used to fit SINDy models from logged data
- `DATA/`: placeholder for CSV logs and processed datasets used by the offline identification workflow
- `results/`: output location for generated experiment logs and plots

## Prerequisites

The exact setup depends on which part of the repository you want to run, but in general you will need:

- Ubuntu/Linux environment
- Python 3
- Gazebo Harmonic
- Jupyter Notebook or JupyterLab for the offline notebook

For the ArduPilot workflow:

- ROS 2
- MAVROS
- ArduPilot SITL

For the Crazyflie workflow:

- CrazySim
- `cflib` / Crazyflie Python tools

For the offline SINDy notebook:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `pysindy`


