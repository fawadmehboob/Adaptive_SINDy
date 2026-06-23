#!/usr/bin/env python3
import argparse
import csv
import math
import os
import time
from dataclasses import dataclass

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy, QoSHistoryPolicy
from rcl_interfaces.srv import GetParameters
from rcl_interfaces.msg import ParameterType

from geometry_msgs.msg import PoseStamped, TwistStamped
from mavros_msgs.msg import State, RCOut, Thrust, AttitudeTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL, ParamGet


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def clamp_norm(v: np.ndarray, max_norm: float) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if max_norm > 0 and n > max_norm:
        return v * (max_norm / (n + 1e-12))
    return v


def quat_xyzw_to_Rwb(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw) + 1e-12
    x, y, z, w = qx / n, qy / n, qz / n, qw / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def quat_from_R(R: np.ndarray) -> np.ndarray:
    tr = np.trace(R)
    if tr > 0:
        S = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * S
        x = (R[2, 1] - R[1, 2]) / S
        y = (R[0, 2] - R[2, 0]) / S
        z = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / S
        x = 0.25 * S
        y = (R[0, 1] + R[1, 0]) / S
        z = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / S
        x = (R[0, 1] + R[1, 0]) / S
        y = 0.25 * S
        z = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / S
        x = (R[0, 2] + R[2, 0]) / S
        y = (R[1, 2] + R[2, 1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z], dtype=float)


def compute_desired_attitude_from_force(F_d_enu: np.ndarray, yaw_des: float = 0.0):
    T_d = float(np.linalg.norm(F_d_enu))
    if T_d < 1e-6:
        b3 = np.array([0.0, 0.0, 1.0], dtype=float)
        T_d = 1e-6
    else:
        b3 = F_d_enu / T_d

    b1_h = np.array([math.cos(yaw_des), math.sin(yaw_des), 0.0], dtype=float)
    b2 = np.cross(b3, b1_h)
    nb2 = float(np.linalg.norm(b2))
    if nb2 < 1e-8:
        b2 = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        b2 = b2 / nb2
    b1 = np.cross(b2, b3)

    R_d = np.column_stack((b1, b2, b3))
    return R_d, T_d


def theta_from_T_theta_phi(T: float, theta: float, phi: float) -> np.ndarray:
    return np.array(
        [[
            1.0,
            float(theta),
            float(phi),
            float(T) * math.sin(theta),
            float(T) * math.cos(theta),
            float(T) * math.sin(phi),
            float(T) * math.cos(phi),
        ]],
        dtype=float,
    )


def configure_log_path(args, controller_name: str) -> None:
    if args.log_csv.strip():
        return
    mode_tag = "adapt" if args.adaptive == "on" else "noadapt"
    dist_tag = "diston" if args.disturbance == "on" else "distoff"
    run_id = max(1, int(args.run_id))
    out_dir = os.path.join(args.results_dir, controller_name, args.traj)
    args.log_csv = os.path.join(out_dir, f"run_{run_id:02d}_{mode_tag}_{dist_tag}.csv")


@dataclass
class AdaptParams:
    lambda_l: float
    P0: float
    Q: float
    R: float


@dataclass
class Limits:
    a_cmd_clip: float
    a_dist_clip: float
    tilt_deg: float


class SINDYAttitudeThrustController(Node):
    def __init__(self, args, A_init: np.ndarray):
        super().__init__("sindy_attitude_thrust_controller")
        self.args = args
        self.m = float(args.mass)
        self.g = 9.80665

        self.use_dist = (args.disturbance == "on")
        self.use_adapt = (args.adaptive == "on")

        self.Kp = float(args.kp_pos)
        self.Kd = float(args.kd_vel)

        self.dim_a = 7
        self.dim_y = 3
        if A_init.shape != (self.dim_a, self.dim_y):
            raise RuntimeError(f"SINDy A_init must be ({self.dim_a}, {self.dim_y}), got {A_init.shape}")

        self.A_hat = A_init.copy()
        self.adapt = AdaptParams(
            lambda_l=float(args.lambda_l),
            P0=float(args.P0),
            Q=float(args.Q),
            R=float(args.R),
        )
        self.P = np.eye(self.dim_a) * self.adapt.P0
        self.Q = np.eye(self.dim_a) * self.adapt.Q
        self.R_meas = np.eye(self.dim_y) * self.adapt.R

        self.lim = Limits(
            a_cmd_clip=float(args.a_cmd_clip),
            a_dist_clip=float(args.a_dist_clip),
            tilt_deg=float(args.max_tilt_deg),
        )

        self.a_alpha = float(args.a_meas_alpha)
        self.a_f = np.zeros(3)
        self.v_prev = None
        self.t_track0 = None

        self.pwm_min = float(args.pwm_min)
        self.pwm_max = float(args.pwm_max)
        self.pwm_hover = float(args.pwm_hover)

        self.state = State()
        self.pose = None
        self.vel = None
        self.rc = None
        self.attitude_transport = str(args.attitude_transport)
        self.pose_transport_available = None
        self._warned_no_pose_transport = False
        self._warned_no_raw_transport = False

        qos_be = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            durability=QoSDurabilityPolicy.VOLATILE,
        )

        self.create_subscription(State, "/mavros/state", self.state_cb, qos_be)
        self.create_subscription(PoseStamped, "/mavros/local_position/pose", self.pose_cb, qos_be)
        self.create_subscription(TwistStamped, "/mavros/local_position/velocity_local", self.vel_cb, qos_be)
        self.create_subscription(RCOut, "/mavros/rc/out", self.rc_cb, qos_be)

        self.att_pub = self.create_publisher(PoseStamped, "/mavros/setpoint_attitude/attitude", 10)
        self.thrust_pub = self.create_publisher(Thrust, "/mavros/setpoint_attitude/thrust", 10)
        self.raw_att_pub = self.create_publisher(AttitudeTarget, "/mavros/setpoint_raw/attitude", 10)

        self.arm_client = self.create_client(CommandBool, "/mavros/cmd/arming")
        self.mode_client = self.create_client(SetMode, "/mavros/set_mode")
        self.takeoff_client = self.create_client(CommandTOL, "/mavros/cmd/takeoff")
        self.param_get_client = self.create_client(ParamGet, "/mavros/param/get")
        self.sp_att_param_client = self.create_client(GetParameters, "/mavros/setpoint_attitude/get_parameters")
        self.sp_vel_param_client = self.create_client(GetParameters, "/mavros/setpoint_velocity/get_parameters")

        self.param_get_available = False
        self.wait_services()

        self.rows = []
        self.log_enabled = (args.log_csv.strip() != "")
        self.get_logger().info("SINDy attitude+thrust controller ready")

    def state_cb(self, msg):
        self.state = msg

    def pose_cb(self, msg):
        self.pose = msg

    def vel_cb(self, msg):
        self.vel = msg

    def rc_cb(self, msg):
        self.rc = msg

    def wait_services(self):
        for client, name in [
            (self.arm_client, "arming"),
            (self.mode_client, "set_mode"),
            (self.takeoff_client, "takeoff"),
        ]:
            while not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().info(f"Waiting for {name} service...")

        if not self.param_get_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("param/get service not available; GUID_OPTIONS check will be skipped.")
            self.param_get_available = False
        else:
            self.param_get_available = True

    def wait_for_connection_and_telemetry(self):
        self.get_logger().info("Waiting for FCU + pose + velocity...")
        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.state.connected and (self.pose is not None) and (self.vel is not None):
                break
        self.get_logger().info("FCU + telemetry ready!")

    def check_frame_convention(self):
        pose_transport_available = None
        if self.sp_att_param_client.wait_for_service(timeout_sec=1.0):
            req = GetParameters.Request()
            req.names = ["use_quaternion"]
            fut = self.sp_att_param_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            res = fut.result()
            if res and res.values and res.values[0].type == ParameterType.PARAMETER_BOOL:
                if bool(res.values[0].bool_value):
                    pose_transport_available = True
                    self.get_logger().info("Frame check: setpoint_attitude.use_quaternion=true (OK).")
                else:
                    pose_transport_available = False
                    self.get_logger().warn(
                        "Frame check: setpoint_attitude.use_quaternion=false; "
                        "PoseStamped attitude setpoints will be ignored."
                    )
            else:
                self.get_logger().warn("Frame check: could not read setpoint_attitude.use_quaternion.")
        else:
            self.get_logger().warn("/mavros/setpoint_attitude/get_parameters unavailable; cannot verify PoseStamped attitude transport.")

        self.pose_transport_available = pose_transport_available
        if self.attitude_transport == "auto":
            if pose_transport_available:
                self.attitude_transport = "pose_thrust"
            else:
                self.attitude_transport = "raw"
                self.get_logger().warn(
                    "Falling back to /mavros/setpoint_raw/attitude so tracking commands reach MAVROS."
                )
        elif self.attitude_transport == "pose_thrust" and pose_transport_available is False:
            self.get_logger().warn(
                "--attitude-transport=pose_thrust requested, but MAVROS quaternion transport is disabled."
            )

        if self.sp_vel_param_client.wait_for_service(timeout_sec=1.0):
            req = GetParameters.Request()
            req.names = ["mav_frame"]
            fut = self.sp_vel_param_client.call_async(req)
            rclpy.spin_until_future_complete(self, fut, timeout_sec=2.0)
            res = fut.result()
            if res and res.values:
                p = res.values[0]
                if p.type == ParameterType.PARAMETER_STRING:
                    self.get_logger().info(f"Frame check: setpoint_velocity.mav_frame='{p.string_value}'.")
                elif p.type == ParameterType.PARAMETER_INTEGER:
                    self.get_logger().info(f"Frame check: setpoint_velocity.mav_frame={p.integer_value}.")

        self.get_logger().info("Frame check summary: controller math uses ENU; MAVROS performs FCU frame conversion.")
        self.get_logger().info(f"Attitude command transport: {self.attitude_transport}")

    def check_guid_options(self):
        if not self.param_get_available:
            return
        req = ParamGet.Request()
        req.param_id = "GUID_OPTIONS"
        fut = self.param_get_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        res = fut.result()
        if res is None or not res.success:
            self.get_logger().warn("Could not read GUID_OPTIONS (param/get failed).")
            return
        try:
            guid_opts = int(res.value.integer)
        except Exception:
            try:
                guid_opts = int(res.value.real)
            except Exception:
                self.get_logger().warn("GUID_OPTIONS value unreadable.")
                return
        want_bits = [3, 4, 5]
        missing = [b for b in want_bits if ((guid_opts >> b) & 1) == 0]
        if missing:
            self.get_logger().warn(f"GUID_OPTIONS={guid_opts} missing bits {missing}. For ATTITUDE_TARGET use 56.")
        else:
            self.get_logger().info(f"GUID_OPTIONS={guid_opts} looks OK for thrust+tilt.")

    def set_mode(self, mode: str) -> bool:
        req = SetMode.Request()
        req.custom_mode = mode
        fut = self.mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        ok = bool(fut.result() and fut.result().mode_sent)
        self.get_logger().info(f"Set mode {mode}: {ok}")
        return ok

    def arm(self) -> bool:
        req = CommandBool.Request()
        req.value = True
        fut = self.arm_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        ok = bool(fut.result() and fut.result().success)
        self.get_logger().info(f"Armed: {ok}")
        return ok

    def takeoff(self, altitude: float) -> bool:
        req = CommandTOL.Request()
        req.altitude = float(altitude)
        req.latitude = 0.0
        req.longitude = 0.0
        req.min_pitch = 0.0
        req.yaw = float(self.args.yaw)
        fut = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        ok = bool(fut.result() and fut.result().success)
        self.get_logger().info(f"Takeoff {altitude}m: {ok}")
        return ok

    def wait_during_takeoff_window(self, target_altitude: float, duration: float) -> bool:
        if duration <= 0.0:
            return False

        target_z = max(0.5, 0.6 * float(target_altitude))
        reached = False
        last_log_t = 0.0
        t0 = time.time()

        while rclpy.ok() and (time.time() - t0) < duration:
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.time()
            if self.pose is not None:
                z = float(self.pose.pose.position.z)
                if (not reached) and z >= target_z:
                    reached = True
                    self.get_logger().info(
                        f"Takeoff climb detected: z={z:.2f} m reached during stabilization window."
                    )
                if (now - last_log_t) >= 2.0:
                    last_log_t = now
                    self.get_logger().info(f"Takeoff monitor: current z={z:.2f} m")
            time.sleep(0.05)

        if not reached:
            z_now = float(self.pose.pose.position.z) if self.pose is not None else float("nan")
            self.get_logger().warn(
                f"Takeoff command was accepted, but climb to z>={target_z:.2f} m was not observed "
                f"within {duration:.1f}s (current z={z_now:.2f} m)."
            )
        return reached

    @staticmethod
    def pwm_to_thrust(
        pwm,
        air_density=1.225,
        velocity_in=0.0,
        voltage=16.0,
        voltage_max=16.8,
        pwm_min=1000,
        pwm_max=2000,
        spin_min=0.1,
        spin_max=0.95,
        mot_expo=0.65,
        effective_prop_area=0.02,
        max_outflow_velocity=25.0,
    ):
        pwm = np.asarray(pwm, dtype=float)
        pwm_thrust_min = pwm_min + spin_min * (pwm_max - pwm_min)
        pwm_thrust_max = pwm_min + spin_max * (pwm_max - pwm_min)
        command = (pwm - pwm_thrust_min) / (pwm_thrust_max - pwm_thrust_min)
        command = np.clip(command, 0.0, 1.0)
        voltage_scale = voltage / voltage_max
        if voltage_scale < 0.1:
            return np.zeros_like(pwm)
        velocity_out = voltage_scale * max_outflow_velocity * np.sqrt((1 - mot_expo) * command + mot_expo * command ** 2)
        thrust = 0.5 * air_density * effective_prop_area * (velocity_out ** 2 - velocity_in ** 2)
        return np.maximum(thrust, 0.0)

    def desired_trajectory_enu(self, t: float):
        Rr = float(self.args.radius)
        z = float(self.args.z)
        period = float(self.args.period)
        w = 2.0 * math.pi / max(period, 1e-6)
        cx = float(self.args.center_x)
        cy = float(self.args.center_y)

        if self.args.traj == "circle":
            x = cx + Rr * math.cos(w * t)
            y = cy + Rr * math.sin(w * t)
            vx = -Rr * w * math.sin(w * t)
            vy = Rr * w * math.cos(w * t)
            ax = -Rr * w * w * math.cos(w * t)
            ay = -Rr * w * w * math.sin(w * t)
        elif self.args.traj == "infinity":
            x = cx + Rr * math.sin(w * t)
            y = cy + 0.5 * Rr * math.sin(2.0 * w * t)
            vx = Rr * w * math.cos(w * t)
            vy = 0.5 * Rr * 2.0 * w * math.cos(2.0 * w * t)
            ax = -Rr * w * w * math.sin(w * t)
            ay = -0.5 * Rr * (2.0 * w) ** 2 * math.sin(2.0 * w * t)
        else:  # spiral
            T = max(float(self.args.duration), 1e-6)
            s = min(max(t / T, 0.0), 1.0)
            r = Rr * s
            r_dot = (Rr / T) if s < 1.0 else 0.0
            th = w * t
            c = math.cos(th)
            si = math.sin(th)
            x = cx + r * c
            y = cy + r * si
            vx = r_dot * c - r * w * si
            vy = r_dot * si + r * w * c
            ax = -2.0 * r_dot * w * si - r * w * w * c
            ay = 2.0 * r_dot * w * c - r * w * w * si

        return np.array([x, y, z], dtype=float), np.array([vx, vy, 0.0], dtype=float), np.array([ax, ay, 0.0], dtype=float)

    def publish_attitude_thrust(self, R_d: np.ndarray, thrust: float):
        q = quat_from_R(R_d)
        thrust = float(clamp(thrust, 0.0, 1.0))

        # Raw attitude transport does not depend on MAVROS setpoint_attitude.use_quaternion.
        if self.attitude_transport == "raw":
            if self.raw_att_pub.get_subscription_count() == 0 and not self._warned_no_raw_transport:
                self.get_logger().warn(
                    "No subscribers on /mavros/setpoint_raw/attitude. "
                    "MAVROS raw attitude plugin may be disabled."
                )
                self._warned_no_raw_transport = True
            msg = AttitudeTarget()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.orientation.w = float(q[0])
            msg.orientation.x = float(q[1])
            msg.orientation.y = float(q[2])
            msg.orientation.z = float(q[3])
            msg.thrust = thrust
            msg.type_mask = (
                AttitudeTarget.IGNORE_ROLL_RATE
                | AttitudeTarget.IGNORE_PITCH_RATE
                | AttitudeTarget.IGNORE_YAW_RATE
            )
            self.raw_att_pub.publish(msg)
            return

        if self.att_pub.get_subscription_count() == 0 and not self._warned_no_pose_transport:
            self.get_logger().warn(
                "No subscribers on /mavros/setpoint_attitude/attitude. "
                "MAVROS quaternion attitude transport may be disabled."
            )
            self._warned_no_pose_transport = True

        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.pose.orientation.w = float(q[0])
        msg.pose.orientation.x = float(q[1])
        msg.pose.orientation.y = float(q[2])
        msg.pose.orientation.z = float(q[3])
        self.att_pub.publish(msg)

        tmsg = Thrust()
        tmsg.header.stamp = self.get_clock().now().to_msg()
        tmsg.thrust = thrust
        self.thrust_pub.publish(tmsg)

    def publish_level_hover(self):
        self.publish_attitude_thrust(np.eye(3, dtype=float), float(self.args.hover_thrust))

    def adapt_step(self, phi: np.ndarray, y_force: np.ndarray, s: np.ndarray, dt: float):
        lam = self.adapt.lambda_l
        phi_vec = phi.reshape(-1)
        y_hat = (phi @ self.A_hat).reshape(3,)
        err = y_hat - y_force

        Pphi = self.P @ phi_vec
        for j in range(3):
            rjj = max(float(self.R_meas[j, j]), 1e-12)
            self.A_hat[:, j] += dt * (-lam * self.A_hat[:, j] - Pphi * (err[j] / rjj) + Pphi * float(s[j]))

        rbar = max(float(np.mean(np.diag(self.R_meas))), 1e-12)
        Pdot = (-2.0 * lam) * self.P + self.Q - (1.0 / rbar) * np.outer(Pphi, Pphi)
        self.P += dt * Pdot
        self.P = 0.5 * (self.P + self.P.T)
        w, V = np.linalg.eigh(self.P)
        self.P = V @ np.diag(np.maximum(w, 1e-8)) @ V.T

        if self.args.ahat_clip > 0:
            self.A_hat = np.clip(self.A_hat, -self.args.ahat_clip, self.args.ahat_clip)

    def log_row(self, t, p, p_des, v_cmd, yenu, adist, pos_err_norm, T_N):
        if not self.log_enabled:
            return
        t_epoch = time.time()
        t_sync = self.get_clock().now().nanoseconds * 1e-9
        self.rows.append([
            t, t_epoch, t_sync,
            p[0], p[1], p[2],
            p_des[0], p_des[1], p_des[2],
            v_cmd[0], v_cmd[1], v_cmd[2],
            yenu[0], yenu[1], yenu[2],
            adist[0], adist[1], adist[2],
            float(pos_err_norm),
            float(T_N),
        ])

    def write_csv(self):
        if not self.log_enabled:
            return
        out_dir = os.path.dirname(self.args.log_csv)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(self.args.log_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "t", "t_epoch", "t_sync",
                "px", "py", "pz",
                "px_des", "py_des", "pz_des",
                "vx_cmd", "vy_cmd", "vz_cmd",
                "yenu_x", "yenu_y", "yenu_z",
                "adist_x", "adist_y", "adist_z",
                "pos_err_norm",
                "T_N",
            ])
            w.writerows(self.rows)
        self.get_logger().info(f"Wrote log to {self.args.log_csv}")

    def step(self, t: float, dt: float):
        p = np.array([self.pose.pose.position.x, self.pose.pose.position.y, self.pose.pose.position.z], dtype=float)
        v = np.array([self.vel.twist.linear.x, self.vel.twist.linear.y, self.vel.twist.linear.z], dtype=float)

        q = self.pose.pose.orientation
        Rwb = quat_xyzw_to_Rwb(q.x, q.y, q.z, q.w)

        p_des, v_des, a_des = self.desired_trajectory_enu(t)
        e_p = p - p_des
        e_v = v - v_des
        v_cmd = v_des - float(self.args.Lambda) * e_p

        a_cmd = a_des - self.Kp * e_p - self.Kd * e_v
        a_cmd = clamp_norm(a_cmd, self.lim.a_cmd_clip)

        if self.v_prev is None:
            a_meas = np.zeros(3)
        else:
            a_meas = (v - self.v_prev) / max(dt, 1e-3)
        self.a_f = self.a_alpha * self.a_f + (1.0 - self.a_alpha) * a_meas
        a_use = self.a_f

        g_vec = np.array([0.0, 0.0, self.g], dtype=float)
        T_N = 0.0
        if self.rc is not None and hasattr(self.rc, "channels") and len(self.rc.channels) >= 4:
            pwm = np.array(self.rc.channels[:4], dtype=float)
            T_N = float(np.sum(self.pwm_to_thrust(
                pwm.reshape(1, 4),
                pwm_min=self.pwm_min,
                pwm_max=self.pwm_max,
            )))
            F_body = np.array([0.0, 0.0, T_N], dtype=float)
            u_prev = (Rwb @ F_body.reshape(3, 1)).reshape(3,)
        else:
            T_N = float(self.m * self.g)
            u_prev = np.array([0.0, 0.0, self.m * self.g], dtype=float)

        y_force = self.m * a_use + self.m * g_vec - u_prev

        theta = math.asin(-Rwb[2, 0])
        phi = math.atan2(Rwb[2, 1], Rwb[2, 2])
        phi_feat = theta_from_T_theta_phi(T_N, theta, phi)

        if self.use_dist and self.use_adapt:
            s = e_v + float(self.args.Lambda) * e_p
            self.adapt_step(phi_feat, y_force, s, dt)

        if self.use_dist:
            f_hat = (phi_feat @ self.A_hat).reshape(3,)
            if self.t_track0 is None:
                self.t_track0 = time.time()
            ramp = 1.0
            if self.args.dist_ramp_s > 1e-6:
                ramp = min(1.0, max(0.0, (time.time() - self.t_track0) / self.args.dist_ramp_s))
            f_hat *= ramp
            adist_hat = f_hat / max(self.m, 1e-6)
            adist_hat *= float(self.args.adist_scale)
            adist_hat[:2] *= float(self.args.adist_scale_xy)
            adist_hat[2] *= float(self.args.adist_scale_z)
            f_hat = adist_hat * self.m
            f_hat = clamp_norm(f_hat, float(self.args.f_dist_clip))
            f_hat = clamp_norm(f_hat, self.lim.a_dist_clip * self.m)
        else:
            f_hat = np.zeros(3)

        F_d = self.m * (a_cmd + g_vec) - f_hat

        max_tilt = math.radians(self.lim.tilt_deg)
        horiz = np.linalg.norm(F_d[:2])
        if horiz > 1e-6:
            max_horiz = float(F_d[2]) * math.tan(max_tilt)
            if horiz > max_horiz > 0:
                F_d[:2] = F_d[:2] * (max_horiz / horiz)

        R_d, T_d = compute_desired_attitude_from_force(F_d, yaw_des=float(self.args.yaw))
        thrust = (T_d / max(self.m * self.g, 1e-6)) * float(self.args.hover_thrust)
        self.publish_attitude_thrust(R_d, thrust)

        yenu = y_force / max(self.m, 1e-6)
        adist = f_hat / max(self.m, 1e-6)
        self.log_row(t, p, p_des, v_cmd, yenu, adist, float(np.linalg.norm(e_p)), T_N)
        self.v_prev = v.copy()


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", choices=["circle", "infinity", "spiral"], default="spiral")
    parser.add_argument("--disturbance", choices=["on", "off"], default="on")
    parser.add_argument("--adaptive", choices=["on", "off"], default="on")
    parser.add_argument("--log-csv", default="",
                        help="Optional explicit CSV log path. If omitted, path is auto-generated from run/traj.")
    parser.add_argument("--run-id", type=int, default=1,
                        help="Run index used in auto-generated CSV naming (e.g., 1..10).")
    parser.add_argument("--results-dir", default="results",
                        help="Base directory for auto-generated logs.")

    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--period", type=float, default=20.0)
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--rate-hz", type=float, default=50.0)

    parser.add_argument("--takeoff-alt", type=float, default=3.0)
    parser.add_argument("--takeoff-wait", type=float, default=15.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--attitude-transport", choices=["auto", "pose_thrust", "raw"], default="auto")

    parser.add_argument("--mass", type=float, default=1.50)
    parser.add_argument("--hover-thrust", type=float, default=0.5)

    parser.add_argument("--kp-pos", type=float, default=1.35)
    parser.add_argument("--kd-vel", type=float, default=1.20)
    parser.add_argument("--a-cmd-clip", type=float, default=1.8)
    parser.add_argument("--max-tilt-deg", type=float, default=25.0)
    parser.add_argument("--Lambda", type=float, default=1.0)

    parser.add_argument("--lambda-l", type=float, default=0.01)
    parser.add_argument("--P0", type=float, default=1e5)
    parser.add_argument("--Q", type=float, default=1e-4)
    parser.add_argument("--R", type=float, default=3e-2)
    parser.add_argument("--ahat-clip", type=float, default=8e-3)
    parser.add_argument("--a-dist-clip", type=float, default=5.0)
    parser.add_argument("--f-dist-clip", type=float, default=10.0)
    parser.add_argument("--adist-scale", type=float, default=0.55)
    parser.add_argument("--adist-scale-xy", type=float, default=0.80)
    parser.add_argument("--adist-scale-z", type=float, default=0.70)
    parser.add_argument("--dist-ramp-s", type=float, default=3.0)
    parser.add_argument("--a-meas-alpha", type=float, default=0.85)

    parser.add_argument("--pwm-min", type=float, default=1000.0)
    parser.add_argument("--pwm-max", type=float, default=2000.0)
    parser.add_argument("--pwm-hover", type=float, default=1500.0)

    cli = parser.parse_args(args=args)
    configure_log_path(cli, controller_name="adaptive_sindy_tracker")

    A_acc_init = np.array([
        [-1.2067, 0.7159, 6.6525],
        [-5.5967, -2.4812, 0.7740],
        [-14.069, -9.3969, 28.4207],
        [2.0308, 0.2608, 0.1418],
        [0.06, -4.7287, 0.5801],
        [1.2474, 1.2628, -2.6191],
        [0.0514, 4.6624, -0.9564],
    ], dtype=float)

    rclpy.init(args=args)
    node = SINDYAttitudeThrustController(cli, A_acc_init)
    node.wait_for_connection_and_telemetry()
    node.check_frame_convention()
    node.check_guid_options()

    dt_cmd = 1.0 / max(cli.rate_hz, 1e-6)

    if node.attitude_transport == "pose_thrust":
        node.get_logger().info("Pre-streaming neutral attitude targets...")
        t0 = time.time()
        while rclpy.ok() and (time.time() - t0) < 2.0:
            rclpy.spin_once(node, timeout_sec=0.0)
            node.publish_level_hover()
            time.sleep(dt_cmd)
    else:
        node.get_logger().info(
            "Skipping pre-stream before takeoff because raw attitude setpoints can interfere with CommandTOL climb."
        )

    node.set_mode("GUIDED")
    node.arm()
    node.takeoff(cli.takeoff_alt)

    node.get_logger().info(f"Waiting {cli.takeoff_wait}s to stabilize...")
    node.wait_during_takeoff_window(cli.takeoff_alt, cli.takeoff_wait)

    start = time.time()
    t_prev = start
    node.get_logger().info(f"Tracking trajectory: {cli.traj} (dist={cli.disturbance}, adapt={cli.adaptive})")
    node.get_logger().info(f"CSV log path: {cli.log_csv}")
    while rclpy.ok() and (time.time() - start) < cli.duration:
        now = time.time()
        t = now - start
        dt = max(now - t_prev, 1e-3)
        rclpy.spin_once(node, timeout_sec=0.0)
        if node.pose is not None and node.vel is not None:
            node.step(t, dt)
        time.sleep(dt_cmd)
        t_prev = now

    node.get_logger().info("Landing...")
    node.set_mode("LAND")
    node.write_csv()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
