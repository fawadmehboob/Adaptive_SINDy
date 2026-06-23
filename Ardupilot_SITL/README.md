# ArduPilot SITL

Origin workspace: `/home/dzmitry/ros2_ws`

Files currently kept in this repo:

- `adaptive_sindy_tracker.py`: Adaptive SINDy controller for the ArduPilot SITL workflow
- `wind_wrench_cli.py`: Gazebo wind/disturbance injector for the ArduPilot setup

Keep in the external workspace, not in this repo:

- `build/`, `install/`, `log/`
- rosbags and `.db3` files
- result CSVs and plots
- terrain files and other simulator runtime artifacts

Optional file to copy later if your launch setup depends on it:

- `apm_pluginlists_att.yaml`
