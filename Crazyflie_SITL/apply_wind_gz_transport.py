#!/usr/bin/env python3
import argparse
import math
import time
import numpy as np

# Gazebo Transport Python
from gz.transport13 import Node  # per Gazebo transport python docs :contentReference[oaicite:1]{index=1}

# Try common gz-msgs Python module versions (depends on your distro)
def _import_gz_msg(msg_module_base: str, cls: str):
    for v in (10, 11, 9, 8):
        try:
            mod = __import__(f"gz.msgs{v}.{msg_module_base}_pb2", fromlist=[cls])
            return getattr(mod, cls)
        except Exception:
            continue
    raise ImportError(
        f"Could not import {cls} from gz.msgsXX.{msg_module_base}_pb2. "
        "You may be missing the Python gz-msgs package for your system."
    )

EntityWrench = _import_gz_msg("entity_wrench", "EntityWrench")
Entity       = _import_gz_msg("entity", "Entity")
OdometryWithCovariance = _import_gz_msg("odometry_with_covariance", "OdometryWithCovariance")

# Your wind model (sim-agnostic parts)
from wind_model import FanJetWindModel, FanJetParams


def clamp_vec(force_vec, fmax):
    norm = float(np.linalg.norm(force_vec))
    if norm > fmax:
        force_vec = force_vec * (fmax / (norm + 1e-12))
    return force_vec


def clamp_components(force_vec, fx_max, fy_max, fz_max):
    force_vec[0] = float(np.clip(force_vec[0], -fx_max, fx_max))
    force_vec[1] = float(np.clip(force_vec[1], -fy_max, fy_max))
    force_vec[2] = float(np.clip(force_vec[2], -fz_max, fz_max))
    return force_vec


def clamp_slew(force_desired, force_prev, df_max):
    if df_max <= 0.0:
        return force_desired
    delta = force_desired - force_prev
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm > df_max:
        delta = delta * (df_max / (delta_norm + 1e-12))
    return force_prev + delta


def quat_to_rot(qx, qy, qz, qw):
    xx, yy, zz = qx*qx, qy*qy, qz*qz
    xy, xz, yz = qx*qy, qx*qz, qy*qz
    wx, wy, wz = qw*qx, qw*qy, qw*qz
    return np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=float)


class WindApplierGZ:
    def __init__(self, args):
        self.args = args
        self.node = Node()

        self.odom_msg = None
        self.odom_stamp_ns = None
        self.last_pub_stamp_ns = None
        self.last_wall = None
        self.last_sim_t = None
        self.t0 = time.monotonic()
        self.rng = np.random.default_rng(int(args.seed))

        self.ou_state = np.zeros(3, dtype=float)
        self.ou_mu = np.array([args.gust_mu_x, args.gust_mu_y, args.gust_mu_z], dtype=float)
        self.prev_force = np.zeros(3, dtype=float)

        self.wind = None
        if args.wind_mode == "fanjet":
            params = FanJetParams(
                origin_w=np.array(args.fan_origin, dtype=float),
                axis_w=np.array(args.fan_axis, dtype=float),
                u0=args.u0,
                x0=args.x0,
                sigma0=args.sigma0,
                spread_k=args.spread_k,
                turb_sigma=args.turb_sigma,
                turb_tau=args.turb_tau,
                rho=args.rho,
                cdA=args.cdA,
                max_wind_speed=args.max_wind_speed,
                max_force=args.max_force,
            )
            self.wind = FanJetWindModel(params=params, seed=args.seed)

        # Publisher for wrench
        self.pub = self.node.advertise(args.wrench_topic, EntityWrench)

        # Subscriber for odom
        ok = self.node.subscribe(OdometryWithCovariance, args.odom_topic, self._odom_cb)
        if not ok:
            raise RuntimeError(f"Failed subscribing to {args.odom_topic}")

        print("[wind] Subscribed:", args.odom_topic)
        print("[wind] Publishing : ", args.wrench_topic)
        print("[wind] Target link: ", args.link_name)
        print("[wind] Mode      : ", args.wind_mode)

    def _odom_cb(self, msg: OdometryWithCovariance):
        self.odom_msg = msg
        stamp_ns = None
        try:
            stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nsec)
        except Exception:
            try:
                stamp_ns = int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)
            except Exception:
                stamp_ns = None
        self.odom_stamp_ns = stamp_ns

    def _extract_state(self):
        """Extract pos, quat, vel from gz.msgs.OdometryWithCovariance."""
        od = self.odom_msg
        # pose
        pose = od.pose_with_covariance.pose
        pos = pose.position
        ori = pose.orientation

        pos_w = np.array([pos.x, pos.y, pos.z], dtype=float)
        quat = np.array([ori.x, ori.y, ori.z, ori.w], dtype=float)

        # twist
        tw = od.twist_with_covariance.twist
        v = tw.linear
        vel_raw = np.array([v.x, v.y, v.z], dtype=float)

        if self.args.twist_is_body_frame:
            R = quat_to_rot(quat[0], quat[1], quat[2], quat[3])
            vel_w = R @ vel_raw
        else:
            vel_w = vel_raw

        return pos_w, quat, vel_w

    def _extract_sim_time(self):
        try:
            return float(self.odom_msg.header.stamp.sec) + 1e-9 * float(self.odom_msg.header.stamp.nsec)
        except Exception:
            try:
                return float(self.odom_msg.header.stamp.sec) + 1e-9 * float(self.odom_msg.header.stamp.nanosec)
            except Exception:
                return None

    def _make_entity_wrench(self, force_w):
        ew = EntityWrench()
        ew.entity.name = self.args.link_name

        # Set entity type to LINK. Depending on msg version, LINK may be an enum.
        # If enum symbol isn't available, fallback to numeric 2 (commonly LINK).
        try:
            ew.entity.type = Entity.LINK
        except Exception:
            ew.entity.type = 2

        ew.wrench.force.x = float(force_w[0])
        ew.wrench.force.y = float(force_w[1])
        ew.wrench.force.z = float(force_w[2])
        ew.wrench.torque.x = 0.0
        ew.wrench.torque.y = 0.0
        ew.wrench.torque.z = 0.0
        return ew

    def _ou_force_world(self, t, dt):
        cycle = max(1e-6, float(self.args.gust_on_sec) + float(self.args.gust_off_sec))
        phase = float(t % cycle)
        gust_active = phase < float(self.args.gust_on_sec)

        theta = float(self.args.gust_theta)
        sigma = float(self.args.gust_sigma)
        if gust_active:
            noise = self.rng.standard_normal(3)
            self.ou_state = self.ou_state + theta * (self.ou_mu - self.ou_state) * dt + sigma * math.sqrt(dt) * noise
        else:
            self.ou_state = self.ou_state + theta * (self.ou_mu - self.ou_state) * dt

        fx = float(self.args.fx_mean) + float(self.args.fx_amp) * math.sin(2.0 * math.pi * float(self.args.freq_hz) * t) + self.ou_state[0]
        fy = float(self.args.fy_mean) + float(self.args.fy_amp) * math.sin(2.0 * math.pi * float(self.args.freq_hz) * t + 1.3) + self.ou_state[1]
        fz = float(self.args.fz_mean) + self.ou_state[2]

        force_raw = np.array([fx, fy, fz], dtype=float)
        force_cmd = clamp_vec(force_raw.copy(), float(self.args.fmax))
        force_cmd = clamp_components(force_cmd, float(self.args.fx_max), float(self.args.fy_max), float(self.args.fz_max))
        force_cmd = clamp_slew(force_cmd, self.prev_force, float(self.args.df_max_per_step))
        force_cmd = clamp_vec(force_cmd, float(self.args.fmax))
        self.prev_force = force_cmd.copy()
        return force_cmd

    def run(self):
        dt_target = 1.0 / self.args.rate_hz
        try:
            while True:
                if self.odom_msg is None:
                    time.sleep(0.01)
                    continue

                if self.odom_stamp_ns is not None and self.odom_stamp_ns == self.last_pub_stamp_ns:
                    time.sleep(0.001)
                    continue
                self.last_pub_stamp_ns = self.odom_stamp_ns

                now = time.monotonic()
                sim_t = self._extract_sim_time()
                if sim_t is not None:
                    if self.last_sim_t is None:
                        dt = dt_target
                    else:
                        dt = max(1e-6, sim_t - self.last_sim_t)
                    self.last_sim_t = sim_t
                    t = sim_t
                elif self.last_wall is None:
                    dt = dt_target
                    t = now - self.t0
                else:
                    dt = max(1e-6, now - self.last_wall)
                    t = now - self.t0
                self.last_wall = now

                pos_w, quat, vel_w = self._extract_state()

                if self.args.wind_mode == "fanjet":
                    wind_v = self.wind.wind_velocity_world(pos_w, t=t, dt=dt)
                    force_world = self.wind.aero_force_world(wind_v, vel_w)
                else:
                    force_world = self._ou_force_world(t=t, dt=dt)

                msg = self._make_entity_wrench(force_world)
                self.pub.publish(msg)

                time.sleep(dt_target)
        except KeyboardInterrupt:
            print("\n[wind] stopping, sending zero wrench once...")
            try:
                msg0 = self._make_entity_wrench(np.zeros(3))
                self.pub.publish(msg0)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--odom_topic", default="/model/crazyflie_0/odometry_with_covariance")
    ap.add_argument("--wrench_topic", default="/world/crazysim_default/wrench")
    ap.add_argument("--link_name", default="crazyflie_0::base_link")

    ap.add_argument("--rate_hz", type=float, default=50.0)
    ap.add_argument("--twist_is_body_frame", action="store_true", default=False,
                    help="If set, rotate twist.linear from body->world using pose quaternion")
    ap.add_argument("--wind_mode", choices=["fanjet", "random_ou"], default="fanjet")

    # wind params
    ap.add_argument("--fan_origin", nargs=3, type=float, default=[0.0, 0.0, 0.5])
    ap.add_argument("--fan_axis",   nargs=3, type=float, default=[1.0, 0.0, 0.0])
    ap.add_argument("--u0", type=float, default=2.0)
    ap.add_argument("--x0", type=float, default=0.20)
    ap.add_argument("--sigma0", type=float, default=0.10)
    ap.add_argument("--spread_k", type=float, default=0.18)
    ap.add_argument("--turb_sigma", type=float, default=0.8)
    ap.add_argument("--turb_tau", type=float, default=0.35)
    ap.add_argument("--rho", type=float, default=1.225)
    ap.add_argument("--cdA", type=float, default=0.0012)
    ap.add_argument("--max_wind_speed", type=float, default=2.0)
    ap.add_argument("--max_force", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=0)

    # position-independent random force wind (same logic style as wind_wrench_cli.py)
    ap.add_argument("--fx-mean", type=float, default=0.0)
    ap.add_argument("--fy-mean", type=float, default=0.0)
    ap.add_argument("--fz-mean", type=float, default=0.0)
    ap.add_argument("--fx-amp",  type=float, default=0.0)
    ap.add_argument("--fy-amp",  type=float, default=0.0)
    ap.add_argument("--freq-hz", type=float, default=0.1)
    ap.add_argument("--gust-sigma", type=float, default=2.0)
    ap.add_argument("--gust-theta", type=float, default=0.8)
    ap.add_argument("--gust-mu-x", type=float, default=0.0)
    ap.add_argument("--gust-mu-y", type=float, default=0.0)
    ap.add_argument("--gust-mu-z", type=float, default=0.0)
    ap.add_argument("--gust-on-sec", type=float, default=3.0)
    ap.add_argument("--gust-off-sec", type=float, default=3.0)
    ap.add_argument("--fmax", type=float, default=8.0)
    ap.add_argument("--fx-max", type=float, default=4.0)
    ap.add_argument("--fy-max", type=float, default=4.0)
    ap.add_argument("--fz-max", type=float, default=4.0)
    ap.add_argument("--df-max-per-step", type=float, default=1.0)

    args = ap.parse_args()
    WindApplierGZ(args).run()


if __name__ == "__main__":
    main()
