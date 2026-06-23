# Crazyflie SITL

Origin workspace: `/home/dzmitry/CrazySim/scripts`

Files currently kept in this repo:

- `cf_sindy_adaptive_tracker.py`: Adaptive SINDy controller for Crazyflie SITL
- `apply_wind_gz_transport.py`: Gazebo transport wind applier
- `wind_model.py`: local wind model used by the Gazebo transport script
- `thrustfit_latest.json`: thrust calibration used by the controller

Generated outputs should go under `../results/crazyflie/` and stay out of Git.

Useful optional files that still live only in the original workspace:

- `compare_controller_results.py`
- `plot_adaptive_sindy_best_trajectories.py`
- `motor_model_params.json`
- calibration and fitting scripts if you want the repo to include model-identification steps
