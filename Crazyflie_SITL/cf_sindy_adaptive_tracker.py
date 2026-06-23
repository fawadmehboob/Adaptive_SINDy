#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.high_level_commander import HighLevelCommander
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie


def clamp(value, lower, upper):
    return max(lower, min(upper, value))


def clamp_norm(vector: np.ndarray, max_norm: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if max_norm > 0.0 and norm > max_norm:
        return vector * (max_norm / (norm + 1e-12))
    return vector


def wrap_pi(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def rate_limit(prev: float, target: float, max_step: float) -> float:
    delta = target - prev
    if delta > max_step:
        return prev + max_step
    if delta < -max_step:
        return prev - max_step
    return target


def rot_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_matrix = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=float)
    return rotation_matrix


def compute_desired_attitude_from_force(force_world: np.ndarray, yaw_desired: float) -> tuple[np.ndarray, float]:
    thrust_newton = float(np.linalg.norm(force_world))
    if thrust_newton < 1e-6:
        b3 = np.array([0.0, 0.0, 1.0], dtype=float)
        thrust_newton = 1e-6
    else:
        b3 = force_world / thrust_newton

    b1_heading = np.array([math.cos(yaw_desired), math.sin(yaw_desired), 0.0], dtype=float)
    b2 = np.cross(b3, b1_heading)
    norm_b2 = float(np.linalg.norm(b2))
    if norm_b2 < 1e-8:
        b2 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        b2 = b2 / norm_b2
    b1 = np.cross(b2, b3)

    rotation_matrix = np.column_stack((b1, b2, b3))
    return rotation_matrix, thrust_newton


def rotation_to_rpy(rotation_matrix: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(clamp(-float(rotation_matrix[2, 0]), -1.0, 1.0))
    roll = math.atan2(float(rotation_matrix[2, 1]), float(rotation_matrix[2, 2]))
    yaw = math.atan2(float(rotation_matrix[1, 0]), float(rotation_matrix[0, 0]))
    return roll, pitch, yaw


def theta_from_T_theta_phi(thrust_newton: float, theta: float, phi: float) -> np.ndarray:
    return np.array([
        [
            float(theta),
            float(phi),
            float(thrust_newton) * math.sin(theta),
            float(thrust_newton) * math.cos(theta),
            float(thrust_newton) * math.sin(phi),
            float(thrust_newton) * math.cos(phi),
            float(thrust_newton),
        ]
    ], dtype=float)


def reset_kalman(cf):
    cf.param.set_value("kalman.resetEstimation", "1")
    time.sleep(0.1)
    cf.param.set_value("kalman.resetEstimation", "2")
    time.sleep(1.0)


def logvar_exists(cf, full_name: str) -> bool:
    if "." not in full_name:
        return False
    group_name, var_name = full_name.split(".", 1)
    toc = cf.log.toc.toc
    return (group_name in toc) and (var_name in toc[group_name])


def choose_motor_vars(cf):
    candidates = [
        ["motor.m1", "motor.m2", "motor.m3", "motor.m4"],
        ["pwm.m1", "pwm.m2", "pwm.m3", "pwm.m4"],
        ["motorPower.m1", "motorPower.m2", "motorPower.m3", "motorPower.m4"],
        ["motors.m1", "motors.m2", "motors.m3", "motors.m4"],
    ]
    for motor_vars in candidates:
        if all(logvar_exists(cf, variable_name) for variable_name in motor_vars):
            return motor_vars
    return None


class CFStateLogger:
    def __init__(self, cf, log_period_ms: int, motor_vars):
        self.cf = cf
        self.log_period_ms = int(log_period_ms)
        self.motor_vars = motor_vars
        self.latest = {
            "timestamp": None,
            "x": float("nan"), "y": float("nan"), "z": float("nan"),
            "vx": float("nan"), "vy": float("nan"), "vz": float("nan"),
            "roll": float("nan"), "pitch": float("nan"), "yaw": float("nan"),
            "m1": float("nan"), "m2": float("nan"), "m3": float("nan"), "m4": float("nan"),
        }
        self._confs = []

    def _add(self, log_config, callback):
        self.cf.log.add_config(log_config)
        log_config.data_received_cb.add_callback(callback)
        log_config.start()
        self._confs.append(log_config)

    def _cb_state(self, timestamp, data, _logconf):
        self.latest["timestamp"] = timestamp
        for key in ["stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
                    "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
                    "stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw"]:
            if key in data:
                out_key = key.split(".")[1]
                self.latest[out_key] = float(data[key])

    def _cb_mot(self, timestamp, data, _logconf):
        self.latest["timestamp"] = timestamp
        if self.motor_vars is None:
            return
        self.latest["m1"] = float(data.get(self.motor_vars[0], float("nan")))
        self.latest["m2"] = float(data.get(self.motor_vars[1], float("nan")))
        self.latest["m3"] = float(data.get(self.motor_vars[2], float("nan")))
        self.latest["m4"] = float(data.get(self.motor_vars[3], float("nan")))

    def start(self):
        state_log_a = LogConfig("state_a", period_in_ms=self.log_period_ms)
        for variable_name in [
            "stateEstimate.x", "stateEstimate.y", "stateEstimate.z",
            "stateEstimate.vx", "stateEstimate.vy", "stateEstimate.vz",
        ]:
            state_log_a.add_variable(variable_name, "float")
        self._add(state_log_a, self._cb_state)

        state_log_b = LogConfig("state_b", period_in_ms=self.log_period_ms)
        for variable_name in ["stabilizer.roll", "stabilizer.pitch", "stabilizer.yaw"]:
            state_log_b.add_variable(variable_name, "float")
        self._add(state_log_b, self._cb_state)

        if self.motor_vars is not None:
            motor_log = LogConfig("mot", period_in_ms=self.log_period_ms)
            for variable_name in self.motor_vars:
                motor_log.add_variable(variable_name, "uint16_t")
            self._add(motor_log, self._cb_mot)

    def wait_first(self, timeout_s: float = 2.0):
        start_time = time.time()
        while self.latest["timestamp"] is None:
            if time.time() - start_time > timeout_s:
                raise RuntimeError("No log samples received (timeout).")
            time.sleep(0.01)

    def stop(self):
        for log_config in self._confs:
            try:
                log_config.stop()
            except Exception:
                pass
        self._confs = []


@dataclass
class ThrustMap:
    a_n_per_u2: float
    cmd_min: float
    cmd_max: float

    @classmethod
    def from_json(cls, path: str, cmd_min: float, cmd_max: float):
        with open(path, "r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        coefficient = float(payload["a_N_per_u2"])
        return cls(a_n_per_u2=coefficient, cmd_min=cmd_min, cmd_max=cmd_max)

    def thrust_to_cmd(self, thrust_newton: float) -> int:
        if self.a_n_per_u2 <= 0.0:
            return int(self.cmd_min)
        per_motor = math.sqrt(max(thrust_newton, 0.0) / max(4.0 * self.a_n_per_u2, 1e-12))
        per_motor = clamp(per_motor, self.cmd_min, self.cmd_max)
        return int(per_motor)

    def motors_to_thrust(self, motor_values: np.ndarray) -> float:
        return float(self.a_n_per_u2 * np.sum(np.square(motor_values)))


def desired_trajectory(traj: str, t: float, radius: float, period: float, center_x: float, center_y: float, z_ref: float):
    w = 2.0 * math.pi / max(period, 1e-6)
    if traj == "circle":
        x = center_x + radius * math.cos(w * t)
        y = center_y + radius * math.sin(w * t)
        vx = -radius * w * math.sin(w * t)
        vy = radius * w * math.cos(w * t)
        ax = -radius * w * w * math.cos(w * t)
        ay = -radius * w * w * math.sin(w * t)
    elif traj == "infinity":
        x = center_x + radius * math.sin(w * t)
        y = center_y + 0.5 * radius * math.sin(2.0 * w * t)
        vx = radius * w * math.cos(w * t)
        vy = radius * w * math.cos(2.0 * w * t)
        ax = -radius * w * w * math.sin(w * t)
        ay = -2.0 * radius * w * w * math.sin(2.0 * w * t)
    else:
        tau = max(period, 1e-6)
        exp_term = math.exp(-t / tau)
        r = radius * (1.0 - exp_term)
        dr = radius * exp_term / tau
        ddr = -radius * exp_term / (tau * tau)
        theta = w * t
        c = math.cos(theta)
        s = math.sin(theta)
        x = center_x + r * c
        y = center_y + r * s
        vx = dr * c - r * w * s
        vy = dr * s + r * w * c
        ax = ddr * c - 2.0 * dr * w * s - r * w * w * c
        ay = ddr * s + 2.0 * dr * w * c - r * w * w * s

    position_des = np.array([x, y, z_ref], dtype=float)
    velocity_des = np.array([vx, vy, 0.0], dtype=float)
    accel_des = np.array([ax, ay, 0.0], dtype=float)
    return position_des, velocity_des, accel_des


def safe_token(value: str) -> str:
    token = "".join(ch if (ch.isalnum() or ch in "-_") else "-" for ch in str(value)).strip("-_")
    return token or "x"


def make_csv_path(results_dir: str, results_subdir: str, prefix: str, traj: str, run_id: int, disturbance: str, adaptive: str):
    explicit_path = str(prefix).strip()
    if explicit_path.lower().endswith(".csv"):
        out_path = explicit_path
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        return out_path

    run_dir = os.path.join(results_dir, results_subdir)
    os.makedirs(run_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_num = max(int(run_id), 0)
    mode = f"dist_{safe_token(disturbance)}_adapt_{safe_token(adaptive)}"
    return os.path.join(
        run_dir,
        f"{safe_token(prefix)}_{safe_token(traj)}_run{run_num:02d}_{mode}_{stamp}.csv",
    )


def main():
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent
    default_results_dir = repo_root / "results" / "crazyflie"
    default_thrust_map = script_dir / "thrustfit_latest.json"

    parser = argparse.ArgumentParser(description="SINDy adaptive attitude+thrust tracker for Crazyflie SITL")
    parser.add_argument("--uri", default="udp://0.0.0.0:19850")
    parser.add_argument("--results-dir", default=str(default_results_dir))
    parser.add_argument("--results-subdir", default="controller_comparison_runs")
    parser.add_argument("--csv-prefix", default="cf_sindy")
    parser.add_argument("--run-id", type=int, default=1, help="Manual run index used in CSV naming.")

    parser.add_argument("--controller-id", type=int, default=1)
    parser.add_argument("--takeoff-alt", type=float, default=0.5)
    parser.add_argument("--takeoff-time", type=float, default=1.0)
    parser.add_argument("--takeoff-settle", type=float, default=1.5)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)

    parser.add_argument("--traj", choices=["circle", "infinity", "spiral"], default="circle")
    parser.add_argument("--z", type=float, default=0.5)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--period", type=float, default=10.0)
    parser.add_argument("--yaw", type=float, default=0.0, help="Yaw reference [rad].")

    parser.add_argument("--mass", type=float, default=0.0282)
    parser.add_argument("--kp-pos", type=float, default=1.35)
    parser.add_argument("--kd-vel", type=float, default=1.20)
    parser.add_argument("--Lambda", type=float, default=1.0)
    parser.add_argument("--a-cmd-clip", type=float, default=1.8)
    parser.add_argument("--a-dist-clip", type=float, default=5.0)
    parser.add_argument("--max-tilt-deg", type=float, default=20.0)
    parser.add_argument("--yaw-kp", type=float, default=0.3)
    parser.add_argument("--max-yawrate", type=float, default=30.0)
    parser.add_argument("--hover-yawrate-scale", type=float, default=0.0, help="Scale yawrate command in hover interface.")
    parser.add_argument("--command-interface", choices=["hover", "attitude_thrust"], default="hover")
    parser.add_argument("--max-vxy", type=float, default=0.8)
    parser.add_argument("--z-min", type=float, default=0.1)
    parser.add_argument("--z-max", type=float, default=1.5)
    parser.add_argument("--kz-pos", type=float, default=1.0)
    parser.add_argument("--kv-adist-xy", type=float, default=0.0)
    parser.add_argument("--kz-adist", type=float, default=0.0)
    parser.add_argument("--cmd-lpf-alpha", type=float, default=0.25, help="LPF alpha for hover commands in [0,1].")
    parser.add_argument("--max-dvxy-step", type=float, default=0.03, help="Max per-step change for vx/vy command [m/s].")
    parser.add_argument("--max-dz-step", type=float, default=0.005, help="Max per-step change for z command [m].")

    parser.add_argument("--disturbance", choices=["on", "off"], default="on")
    parser.add_argument("--adaptive", choices=["on", "off"], default="on")
    parser.add_argument("--lambda-l", type=float, default=0.01)
    parser.add_argument("--P0", type=float, default=1e5)
    parser.add_argument("--Q", type=float, default=1e-4)
    parser.add_argument("--R", type=float, default=3e-2)
    parser.add_argument("--ahat-clip", type=float, default=8e-3)
    parser.add_argument("--f-dist-clip", type=float, default=2.0)
    parser.add_argument("--adist-scale", type=float, default=0.55)
    parser.add_argument("--adist-scale-xy", type=float, default=0.80)
    parser.add_argument("--adist-scale-z", type=float, default=0.70)
    parser.add_argument("--dist-ramp-s", type=float, default=3.0)
    parser.add_argument("--a-meas-alpha", type=float, default=0.85)

    parser.add_argument("--pwm-min", type=float, default=10000.0)
    parser.add_argument("--pwm-max", type=float, default=60000.0)
    parser.add_argument("--thrust-map", default=str(default_thrust_map))
    parser.add_argument("--thrust-cmd-override", type=int, default=-1, help="If >=0, send this raw thrust command directly.")
    parser.add_argument("--thrust-cmd-scale", type=float, default=1.0, help="Scale applied to model-based thrust command.")
    parser.add_argument("--thrust-cmd-bias", type=int, default=0, help="Bias added to model-based thrust command.")
    parser.add_argument("--thrust-boost-cmd", type=int, default=-1, help="If >=0 and --thrust-hover-cmd>=0, use this thrust during initial boost phase.")
    parser.add_argument("--thrust-hover-cmd", type=int, default=-1, help="If >=0 and --thrust-boost-cmd>=0, hold this thrust after boost/transition.")
    parser.add_argument("--thrust-boost-s", type=float, default=0.6, help="Duration [s] to hold --thrust-boost-cmd.")
    parser.add_argument("--thrust-transition-s", type=float, default=0.4, help="Ramp time [s] from --thrust-boost-cmd to --thrust-hover-cmd.")
    parser.add_argument("--att-thrust-vz-pi", choices=["on", "off"], default="off", help="Enable vz->thrust PI trim in attitude_thrust mode.")
    parser.add_argument("--att-thrust-vz-target", type=float, default=0.0, help="Target vertical velocity [m/s] for PI trim.")
    parser.add_argument("--att-thrust-vz-kp", type=float, default=0.0, help="PI proportional gain [cmd per (m/s)].")
    parser.add_argument("--att-thrust-vz-ki", type=float, default=0.0, help="PI integral gain [cmd per m].")
    parser.add_argument("--att-thrust-vz-i-clip", type=float, default=1.0, help="Integrator clamp on vz error integral [m].")
    parser.add_argument("--att-thrust-vz-trim-clip", type=float, default=1500.0, help="Clamp on PI trim contribution [cmd].")
    parser.add_argument("--att-thrust-vz-lpf-alpha", type=float, default=0.5, help="LPF alpha for vz used by PI (0..1, higher=faster).")
    parser.add_argument("--att-thrust-vz-start-s", type=float, default=-1.0, help="PI start time [s]. If <0, starts after boost+transition.")

    parser.add_argument("--a-init-json", default="")
    parser.add_argument("--roll-sign", type=float, default=1.0)
    parser.add_argument("--pitch-sign", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.a_init_json:
        with open(args.a_init_json, "r", encoding="utf-8") as fh:
            loaded = np.array(json.load(fh), dtype=float)
    else:
        loaded = np.array([
            [-1.2067, 0.7159, 6.6525],
            [-5.5967, -2.4812, 0.7740],
            [-14.0690, -9.3969, 28.4207],
            [2.0308, 0.2608, 0.1418],
            [0.0600, -4.7287, 0.5801],
            [1.2474, 1.2628, -2.6191],
            [0.0514, 4.6624, -0.9564],
        ], dtype=float)

    if loaded.shape != (7, 3):
        raise RuntimeError(f"SINDy A_init shape must be (7,3), got {loaded.shape}")

    a_hat = loaded.copy()
    covariance = np.eye(7) * float(args.P0)
    proc_noise = np.eye(7) * float(args.Q)
    meas_noise = np.eye(3) * float(args.R)

    thrust_map = ThrustMap.from_json(args.thrust_map, cmd_min=args.pwm_min, cmd_max=args.pwm_max)
    csv_path = make_csv_path(
        args.results_dir,
        args.results_subdir,
        args.csv_prefix,
        args.traj,
        args.run_id,
        args.disturbance,
        args.adaptive,
    )

    cflib.crtp.init_drivers(enable_serial_driver=False)
    with open(csv_path, "w", newline="") as csv_handle:
        writer = csv.writer(csv_handle)
        writer.writerow([
            "t", "phase", "log_timestamp",
            "px", "py", "pz", "vx", "vy", "vz",
            "px_des", "py_des", "pz_des",
            "roll_deg", "pitch_deg", "yaw_deg",
            "cmd_mode", "vx_cmd", "vy_cmd", "z_cmd",
            "roll_cmd_deg", "pitch_cmd_deg", "yawrate_cmd_deg_s", "thrust_cmd", "thrust_trim_cmd",
            "y_force_x", "y_force_y", "y_force_z",
            "fhat_x", "fhat_y", "fhat_z",
            "ep_norm", "ev_norm", "m1", "m2", "m3", "m4",
        ])

        with SyncCrazyflie(args.uri, cf=Crazyflie(rw_cache="./cache")) as scf:
            cf = scf.cf
            print(f"Connected to {args.uri}")

            motor_vars = choose_motor_vars(cf)
            print(f"Motor vars: {motor_vars}")

            logger = CFStateLogger(cf, log_period_ms=int(1000.0 / max(args.rate_hz, 1e-6)), motor_vars=motor_vars)
            logger.start()
            logger.wait_first()
            print("Logger healthy: first sample received.")

            if not args.dry_run:
                cf.param.set_value("system.arm", "1")
                time.sleep(0.1)
                cf.param.set_value("stabilizer.controller", str(args.controller_id))
                time.sleep(0.1)
                cf.param.set_value("commander.enHighLevel", "1")
                time.sleep(0.1)
                reset_kalman(cf)

            hlc: HighLevelCommander = cf.high_level_commander
            commander = cf.commander
            if args.command_interface == "hover" and not hasattr(commander, "send_hover_setpoint"):
                raise RuntimeError("send_hover_setpoint is required for --command-interface hover")
            if not args.dry_run and args.command_interface == "attitude_thrust":
                for _ in range(5):
                    commander.send_setpoint(0.0, 0.0, 0.0, 0)
                    time.sleep(0.02)

            if not args.dry_run:
                hlc.takeoff(float(args.takeoff_alt), float(args.takeoff_time))
                time.sleep(float(args.takeoff_time) + float(args.takeoff_settle))

            a_filt = np.zeros(3)
            velocity_prev = None
            track_start_wall = None
            vx_cmd_prev = 0.0
            vy_cmd_prev = 0.0
            z_cmd_prev = float(args.takeoff_alt)
            thrust_trim_cmd = 0.0
            vz_error_int = 0.0
            vz_filt = 0.0
            thrust_profile_enabled = int(args.thrust_boost_cmd) >= 0 and int(args.thrust_hover_cmd) >= 0
            if float(args.att_thrust_vz_start_s) >= 0.0:
                att_thrust_vz_pi_start_s = float(args.att_thrust_vz_start_s)
            elif thrust_profile_enabled:
                att_thrust_vz_pi_start_s = max(float(args.thrust_boost_s), 0.0) + max(float(args.thrust_transition_s), 0.0)
            else:
                att_thrust_vz_pi_start_s = 0.0

            loop_dt = 1.0 / max(float(args.rate_hz), 1e-6)
            t0 = time.time()
            t_prev = t0

            try:
                while True:
                    now = time.time()
                    t = now - t0
                    if t >= float(args.duration):
                        break
                    dt = max(now - t_prev, 1e-3)

                    state = logger.latest
                    position = np.array([state["x"], state["y"], state["z"]], dtype=float)
                    velocity = np.array([state["vx"], state["vy"], state["vz"]], dtype=float)
                    roll = math.radians(float(state["roll"]))
                    pitch = math.radians(float(state["pitch"]))
                    yaw = math.radians(float(state["yaw"]))
                    rot_wb = rot_from_rpy(roll, pitch, yaw)

                    pos_des, vel_des, acc_des = desired_trajectory(
                        args.traj, t, args.radius, args.period, args.center_x, args.center_y, args.z,
                    )
                    e_pos = position - pos_des
                    e_vel = velocity - vel_des
                    acc_cmd = acc_des - float(args.kp_pos) * e_pos - float(args.kd_vel) * e_vel
                    acc_cmd = clamp_norm(acc_cmd, float(args.a_cmd_clip))

                    if velocity_prev is None:
                        a_meas = np.zeros(3)
                    else:
                        a_meas = (velocity - velocity_prev) / max(dt, 1e-3)
                    a_filt = float(args.a_meas_alpha) * a_filt + (1.0 - float(args.a_meas_alpha)) * a_meas

                    motors = np.array([state["m1"], state["m2"], state["m3"], state["m4"]], dtype=float)
                    have_motors = np.all(np.isfinite(motors))
                    if have_motors:
                        thrust_total = thrust_map.motors_to_thrust(motors)
                    else:
                        thrust_total = float(args.mass) * 9.80665

                    body_force = np.array([0.0, 0.0, thrust_total], dtype=float)
                    u_prev = (rot_wb @ body_force.reshape(3, 1)).reshape(3,)
                    y_force = float(args.mass) * a_filt + np.array([0.0, 0.0, float(args.mass) * 9.80665]) - u_prev

                    theta = math.asin(clamp(-float(rot_wb[2, 0]), -1.0, 1.0))
                    phi = math.atan2(float(rot_wb[2, 1]), float(rot_wb[2, 2]))
                    phi_feat = theta_from_T_theta_phi(thrust_total, theta, phi)

                    if args.disturbance == "on" and args.adaptive == "on":
                        sliding = e_vel + float(args.Lambda) * e_pos
                        phi_vec = phi_feat.reshape(-1)
                        y_hat = (phi_feat @ a_hat).reshape(3,)
                        err = y_hat - y_force
                        p_phi = covariance @ phi_vec
                        for axis in range(3):
                            rjj = max(float(meas_noise[axis, axis]), 1e-12)
                            a_hat[:, axis] += dt * (
                                -float(args.lambda_l) * a_hat[:, axis]
                                - p_phi * (err[axis] / rjj)
                                + p_phi * float(sliding[axis])
                            )
                        rbar = max(float(np.mean(np.diag(meas_noise))), 1e-12)
                        p_dot = (-2.0 * float(args.lambda_l)) * covariance + proc_noise - (1.0 / rbar) * np.outer(p_phi, p_phi)
                        covariance += dt * p_dot
                        covariance = 0.5 * (covariance + covariance.T)
                        if not np.all(np.isfinite(covariance)):
                            covariance = np.eye(covariance.shape[0], dtype=float) * float(args.P0)
                        else:
                            try:
                                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                                covariance = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-8)) @ eigenvectors.T
                            except np.linalg.LinAlgError:
                                covariance = np.eye(covariance.shape[0], dtype=float) * float(args.P0)
                        if float(args.ahat_clip) > 0.0:
                            a_hat = np.clip(a_hat, -float(args.ahat_clip), float(args.ahat_clip))

                    if args.disturbance == "on":
                        f_hat = (phi_feat @ a_hat).reshape(3,)
                        if track_start_wall is None:
                            track_start_wall = time.time()
                        ramp = 1.0
                        if float(args.dist_ramp_s) > 1e-6:
                            ramp = clamp((time.time() - track_start_wall) / float(args.dist_ramp_s), 0.0, 1.0)
                        f_hat = f_hat * ramp
                        adist_hat = f_hat / max(float(args.mass), 1e-6)
                        adist_hat *= float(args.adist_scale)
                        adist_hat[0] *= float(args.adist_scale_xy)
                        adist_hat[1] *= float(args.adist_scale_xy)
                        adist_hat[2] *= float(args.adist_scale_z)
                        f_hat = adist_hat * float(args.mass)
                        f_hat = clamp_norm(f_hat, float(args.f_dist_clip))
                        f_hat = clamp_norm(f_hat, float(args.a_dist_clip) * float(args.mass))
                    else:
                        f_hat = np.zeros(3)

                    force_des = float(args.mass) * (acc_cmd + np.array([0.0, 0.0, 9.80665])) - f_hat
                    adist = f_hat / max(float(args.mass), 1e-6)
                    v_cmd_world = vel_des - float(args.Lambda) * e_pos

                    max_tilt = math.radians(float(args.max_tilt_deg))
                    horiz = float(np.linalg.norm(force_des[:2]))
                    if horiz > 1e-6:
                        max_horiz = float(force_des[2]) * math.tan(max_tilt)
                        if horiz > max_horiz > 0.0:
                            force_des[:2] = force_des[:2] * (max_horiz / horiz)

                    rotation_des, thrust_des_newton = compute_desired_attitude_from_force(force_des, yaw_desired=float(args.yaw))
                    roll_des, pitch_des, _ = rotation_to_rpy(rotation_des)

                    yaw_error = wrap_pi(float(args.yaw) - yaw)
                    yawrate_cmd = clamp(
                        math.degrees(float(args.yaw_kp) * yaw_error),
                        -float(args.max_yawrate),
                        float(args.max_yawrate),
                    )

                    roll_cmd_deg = clamp(float(args.roll_sign) * math.degrees(roll_des), -float(args.max_tilt_deg), float(args.max_tilt_deg))
                    pitch_cmd_deg = clamp(float(args.pitch_sign) * math.degrees(pitch_des), -float(args.max_tilt_deg), float(args.max_tilt_deg))
                    thrust_cmd_base = thrust_map.thrust_to_cmd(thrust_des_newton)
                    if thrust_profile_enabled:
                        boost_cmd = float(args.thrust_boost_cmd)
                        hover_cmd = float(args.thrust_hover_cmd)
                        boost_s = max(float(args.thrust_boost_s), 0.0)
                        transition_s = max(float(args.thrust_transition_s), 1e-6)
                        if t < boost_s:
                            thrust_profile_cmd = boost_cmd
                        elif t < (boost_s + transition_s):
                            alpha = clamp((t - boost_s) / transition_s, 0.0, 1.0)
                            thrust_profile_cmd = (1.0 - alpha) * boost_cmd + alpha * hover_cmd
                        else:
                            thrust_profile_cmd = hover_cmd
                        thrust_cmd = int(
                            clamp(
                                thrust_profile_cmd,
                                float(args.pwm_min),
                                float(args.pwm_max),
                            )
                        )
                    elif int(args.thrust_cmd_override) >= 0:
                        thrust_cmd = int(clamp(int(args.thrust_cmd_override), int(args.pwm_min), int(args.pwm_max)))
                    else:
                        thrust_cmd = int(
                            clamp(
                                float(args.thrust_cmd_scale) * float(thrust_cmd_base) + float(args.thrust_cmd_bias),
                                float(args.pwm_min),
                                float(args.pwm_max),
                            )
                        )
                    thrust_trim_cmd = 0.0
                    if (
                        args.command_interface == "attitude_thrust"
                        and args.att_thrust_vz_pi == "on"
                        and t >= att_thrust_vz_pi_start_s
                    ):
                        lpf_alpha = clamp(float(args.att_thrust_vz_lpf_alpha), 0.0, 1.0)
                        vz_filt = lpf_alpha * float(velocity[2]) + (1.0 - lpf_alpha) * vz_filt
                        vz_err = float(args.att_thrust_vz_target) - vz_filt
                        vz_error_int += vz_err * dt
                        vz_error_int = clamp(vz_error_int, -float(args.att_thrust_vz_i_clip), float(args.att_thrust_vz_i_clip))
                        thrust_trim_cmd = (
                            float(args.att_thrust_vz_kp) * vz_err
                            + float(args.att_thrust_vz_ki) * vz_error_int
                        )
                        thrust_trim_cmd = clamp(
                            thrust_trim_cmd,
                            -float(args.att_thrust_vz_trim_clip),
                            float(args.att_thrust_vz_trim_clip),
                        )
                        thrust_cmd = int(
                            clamp(
                                float(thrust_cmd) + float(thrust_trim_cmd),
                                float(args.pwm_min),
                                float(args.pwm_max),
                            )
                        )
                    vx_cmd_world = float(v_cmd_world[0] - float(args.kv_adist_xy) * adist[0])
                    vy_cmd_world = float(v_cmd_world[1] - float(args.kv_adist_xy) * adist[1])
                    cos_yaw = math.cos(yaw)
                    sin_yaw = math.sin(yaw)
                    vx_cmd_body = cos_yaw * vx_cmd_world + sin_yaw * vy_cmd_world
                    vy_cmd_body = -sin_yaw * vx_cmd_world + cos_yaw * vy_cmd_world
                    vx_cmd_raw = float(clamp(vx_cmd_body, -float(args.max_vxy), float(args.max_vxy)))
                    vy_cmd_raw = float(clamp(vy_cmd_body, -float(args.max_vxy), float(args.max_vxy)))
                    z_cmd_raw = float(clamp(pos_des[2] - float(args.kz_pos) * e_pos[2] - float(args.kz_adist) * adist[2], float(args.z_min), float(args.z_max)))
                    vx_cmd_rl = rate_limit(vx_cmd_prev, vx_cmd_raw, float(args.max_dvxy_step))
                    vy_cmd_rl = rate_limit(vy_cmd_prev, vy_cmd_raw, float(args.max_dvxy_step))
                    z_cmd_rl = rate_limit(z_cmd_prev, z_cmd_raw, float(args.max_dz_step))
                    alpha = float(clamp(args.cmd_lpf_alpha, 0.0, 1.0))
                    vx_cmd = alpha * vx_cmd_rl + (1.0 - alpha) * vx_cmd_prev
                    vy_cmd = alpha * vy_cmd_rl + (1.0 - alpha) * vy_cmd_prev
                    z_cmd = alpha * z_cmd_rl + (1.0 - alpha) * z_cmd_prev
                    vx_cmd_prev = vx_cmd
                    vy_cmd_prev = vy_cmd
                    z_cmd_prev = z_cmd
                    cmd_mode = str(args.command_interface)

                    if not args.dry_run:
                        if args.command_interface == "hover":
                            commander.send_hover_setpoint(vx_cmd, vy_cmd, yawrate_cmd * float(args.hover_yawrate_scale), z_cmd)
                        else:
                            commander.send_setpoint(roll_cmd_deg, pitch_cmd_deg, yawrate_cmd, thrust_cmd)

                    writer.writerow([
                        t, "track", state["timestamp"],
                        position[0], position[1], position[2], velocity[0], velocity[1], velocity[2],
                        pos_des[0], pos_des[1], pos_des[2],
                        math.degrees(roll), math.degrees(pitch), math.degrees(yaw),
                        cmd_mode, vx_cmd, vy_cmd, z_cmd,
                        roll_cmd_deg, pitch_cmd_deg, yawrate_cmd, thrust_cmd, thrust_trim_cmd,
                        y_force[0], y_force[1], y_force[2],
                        f_hat[0], f_hat[1], f_hat[2],
                        float(np.linalg.norm(e_pos)), float(np.linalg.norm(e_vel)),
                        state["m1"], state["m2"], state["m3"], state["m4"],
                    ])

                    if int(t * 10) % 10 == 0:
                        csv_handle.flush()

                    velocity_prev = velocity
                    t_prev = now
                    time.sleep(loop_dt)

            finally:
                logger.stop()
                if not args.dry_run:
                    try:
                        if args.command_interface == "hover":
                            commander.send_hover_setpoint(0.0, 0.0, 0.0, float(args.takeoff_alt))
                            time.sleep(0.1)
                        else:
                            hold_cmd = int(args.thrust_hover_cmd) if int(args.thrust_hover_cmd) >= 0 else int(args.pwm_min)
                            commander.send_setpoint(0.0, 0.0, 0.0, hold_cmd)
                            time.sleep(0.1)
                        commander.send_notify_setpoint_stop()
                        time.sleep(0.05)
                    except Exception:
                        pass
                    try:
                        z_now = float(logger.latest.get("z", float(args.takeoff_alt)))
                        land_duration = clamp(z_now / 0.25 + 1.0, 2.0, 6.0)
                        hlc.land(0.0, float(land_duration))
                        time.sleep(float(land_duration) + 0.5)
                    except Exception:
                        pass

    print(f"Saved adaptive log: {csv_path}")


if __name__ == "__main__":
    main()
