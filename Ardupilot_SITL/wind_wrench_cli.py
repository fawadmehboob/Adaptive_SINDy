
#!/usr/bin/env python3
import argparse
import math
import signal
import subprocess
import time
import numpy as np
import csv
import threading
import re

running = True

def handle_sig(sig, frame):
    global running
    running = False

# -----------------------
# Gazebo clock reader (gz topic -e ...)
# -----------------------
class GzClockReader:
    """
    Streams Gazebo clock messages and keeps the latest sim time.
    Works with either:
      sim { sec: .. nsec: .. }    (gz.msgs.Clock)
    or
      sim_time { sec: .. nsec: .. } (some stats msgs)
    """
    def __init__(self, clock_topic: str):
        self.clock_topic = clock_topic
        self.proc = None
        self._lock = threading.Lock()
        self.sim_sec = None
        self.sim_nsec = None
        self.sim_time = None
        self._thr = None

        # regex supports multiline blocks
        self._re_clock_1 = re.compile(r"sim\s*\{\s*sec:\s*(\d+)\s*nsec:\s*(\d+)\s*\}", re.S)
        self._re_clock_2 = re.compile(r"sim_time\s*\{\s*sec:\s*(\d+)\s*nsec:\s*(\d+)\s*\}", re.S)

    def start(self):
        # -e streams. We parse stdout.
        self.proc = subprocess.Popen(
            ["gz", "topic", "-e", "-t", self.clock_topic],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _run(self):
        buf = ""
        while running and self.proc and self.proc.stdout:
            line = self.proc.stdout.readline()
            if not line:
                break
            buf += line
            # keep buffer bounded
            if len(buf) > 8192:
                buf = buf[-4096:]

            m = self._re_clock_1.search(buf)
            if not m:
                m = self._re_clock_2.search(buf)

            if m:
                sec = int(m.group(1))
                nsec = int(m.group(2))
                with self._lock:
                    self.sim_sec = sec
                    self.sim_nsec = nsec
                    self.sim_time = sec + 1e-9 * nsec

    def latest(self):
        with self._lock:
            return self.sim_time, self.sim_sec, self.sim_nsec

    def stop(self):
        if self.proc:
            try:
                self.proc.terminate()
            except Exception:
                pass

# -----------------------
# Wrench helpers
# -----------------------
def _entity_payload(entity_id=None, entity_name=None, entity_type=None):
    if entity_name:
        if entity_type:
            return f'entity: {{name: "{entity_name}" type: {entity_type}}}'
        return f'entity: {{name: "{entity_name}"}}'
    if entity_id is None:
        raise ValueError("Either entity_id or entity_name must be provided.")
    return f"entity: {{id: {int(entity_id)}}}"


def gz_pub_wrench(topic, entity_id, fx, fy, fz, entity_name=None, entity_type=None):
    ent = _entity_payload(entity_id=entity_id, entity_name=entity_name, entity_type=entity_type)
    payload = (
        f"{ent} "
        f"wrench: {{force: {{x: {fx:.6f}, y: {fy:.6f}, z: {fz:.6f}}}}}"
    )
    subprocess.run(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.EntityWrench", "-p", payload],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def gz_clear_wrench(world, entity_id=None, entity_name=None, entity_type=None):
    topic = f"/world/{world}/wrench/clear"
    if entity_name:
        if entity_type:
            payload = f'name: "{entity_name}" type: {entity_type}'
        else:
            payload = f'name: "{entity_name}"'
    else:
        payload = f"id: {entity_id}"
    subprocess.run(
        ["gz", "topic", "-t", topic, "-m", "gz.msgs.Entity", "-p", payload],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def clamp_vec(f, fmax):
    n = float(np.linalg.norm(f))
    if n > fmax:
        f = f * (fmax / (n + 1e-12))
    return f


def clamp_components(f, fx_max, fy_max, fz_max):
    f[0] = float(np.clip(f[0], -fx_max, fx_max))
    f[1] = float(np.clip(f[1], -fy_max, fy_max))
    f[2] = float(np.clip(f[2], -fz_max, fz_max))
    return f


def clamp_slew(f_des, f_prev, df_max):
    if df_max <= 0:
        return f_des
    delta = f_des - f_prev
    dn = float(np.linalg.norm(delta))
    if dn > df_max:
        delta = delta * (df_max / (dn + 1e-12))
    return f_prev + delta

# Optional: same desired trajectory as your neural script (ENU)
def desired_trajectory_enu(args, t):
    Rr = float(args.radius)
    z = float(args.z)
    period = float(args.period)
    w = 2.0 * math.pi / max(period, 1e-6)
    cx = float(args.center_x)
    cy = float(args.center_y)

    if args.traj == "circle":
        x = cx + Rr * math.cos(w * t)
        y = cy + Rr * math.sin(w * t)
        vx = -Rr * w * math.sin(w * t)
        vy =  Rr * w * math.cos(w * t)
        ax = -Rr * w * w * math.cos(w * t)
        ay = -Rr * w * w * math.sin(w * t)
    else:
        x = cx + Rr * math.sin(w * t)
        y = cy + 0.5 * Rr * math.sin(2.0 * w * t)
        vx = Rr * w * math.cos(w * t)
        vy = 0.5 * Rr * 2.0 * w * math.cos(2.0 * w * t)
        ax = -Rr * w * w * math.sin(w * t)
        ay = -0.5 * Rr * (2.0*w)**2 * math.sin(2.0 * w * t)

    p = np.array([x, y, z], dtype=float)
    v = np.array([vx, vy, 0.0], dtype=float)
    a = np.array([ax, ay, 0.0], dtype=float)
    return p, v, a

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="iris_runway")
    parser.add_argument("--entity-id", type=int, default=None)
    parser.add_argument("--entity-name", type=str, default="",
                        help='Stable Gazebo entity name, e.g. "iris_with_gimbal::base_link"')
    parser.add_argument("--entity-type", type=str, default="",
                        help='Optional gz.msgs.Entity type token, e.g. LINK or MODEL')
    parser.add_argument("--rate-hz", type=float, default=20.0)

    parser.add_argument("--mode", choices=["persistent", "oneshot"], default="persistent")
    parser.add_argument("--clear-when-off", choices=["on","off"], default="on")

    # Wind stats
    parser.add_argument("--fx-mean", type=float, default=0.0)
    parser.add_argument("--fy-mean", type=float, default=0.0)
    parser.add_argument("--fz-mean", type=float, default=0.0)
    parser.add_argument("--fx-amp",  type=float, default=0.0)
    parser.add_argument("--fy-amp",  type=float, default=0.0)
    parser.add_argument("--freq-hz", type=float, default=0.1)

    parser.add_argument("--gust-sigma", type=float, default=2.0)
    parser.add_argument("--gust-theta", type=float, default=0.8)
    parser.add_argument("--gust-mu-x", type=float, default=0.0)
    parser.add_argument("--gust-mu-y", type=float, default=0.0)
    parser.add_argument("--gust-mu-z", type=float, default=0.0)
    parser.add_argument("--gust-on-sec", type=float, default=3.0)
    parser.add_argument("--gust-off-sec", type=float, default=3.0)
    parser.add_argument("--fmax", type=float, default=8.0)
    parser.add_argument("--fx-max", type=float, default=4.0,
                        help="Hard per-axis cap on published Fx (N)")
    parser.add_argument("--fy-max", type=float, default=4.0,
                        help="Hard per-axis cap on published Fy (N)")
    parser.add_argument("--fz-max", type=float, default=4.0,
                        help="Hard per-axis cap on published Fz (N)")
    parser.add_argument("--df-max-per-step", type=float, default=1.0,
                        help="Max force vector change per loop step (N). 0 disables.")

    # Sync / time logging
    parser.add_argument("--t0-epoch", type=float, default=None,
                        help="Shared wall-clock epoch (seconds) to sync multiple logs. If omitted, uses current time.")
    parser.add_argument("--clock-topic", default=None,
                        help="Gazebo clock topic. Default: /world/<world>/clock")
    parser.add_argument("--log-sim-time", choices=["on","off"], default="on")

    # Mass to convert force->accel
    parser.add_argument("--mass-kg", type=float, default=1.5,
                        help="Mass used to compute a_wind = F/m. Set this to your model mass.")
    # (Iris is commonly ~1.5kg in ArduPilot’s iris model, but confirm in your SDF.) :contentReference[oaicite:1]{index=1}

    # Optional: log same desired trajectory columns as neural script
    parser.add_argument("--log-traj", choices=["on","off"], default="on")
    parser.add_argument("--traj", choices=["circle", "infinity"], default="circle")
    parser.add_argument("--z", type=float, default=3.0)
    parser.add_argument("--center-x", type=float, default=0.0)
    parser.add_argument("--center-y", type=float, default=0.0)
    parser.add_argument("--radius", type=float, default=2.0)
    parser.add_argument("--period", type=float, default=20.0)

    parser.add_argument("--log-csv", default="wind_log.csv")

    args = parser.parse_args()
    if args.entity_id is None and not args.entity_name.strip():
        raise ValueError("Provide either --entity-id or --entity-name.")

    global running
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    dt = 1.0 / max(args.rate_hz, 1e-6)
    rng = np.random.default_rng(0)
    entity_name = args.entity_name.strip() if args.entity_name else None
    entity_type = args.entity_type.strip() if args.entity_type else None

    topic = f"/world/{args.world}/wrench/persistent" if args.mode == "persistent" else f"/world/{args.world}/wrench"
    target_desc = f"id={args.entity_id}" if entity_name is None else f'name="{entity_name}"'
    if entity_type:
        target_desc += f" type={entity_type}"
    print(f"[wind] topic={topic} target={target_desc} rate={args.rate_hz}Hz fmax={args.fmax}N")

    # setup sync epoch
    t0_epoch = float(args.t0_epoch) if args.t0_epoch is not None else time.time()

    # setup Gazebo clock reader
    clock_reader = None
    if args.log_sim_time == "on":
        clock_topic = args.clock_topic or f"/world/{args.world}/clock"
        clock_reader = GzClockReader(clock_topic)
        try:
            clock_reader.start()
            print(f"[wind] reading sim time from {clock_topic}")
        except Exception as e:
            print(f"[wind] WARN: could not start clock reader: {e}")
            clock_reader = None

    # CSV log
    log_enabled = (args.log_csv is not None and args.log_csv.strip() != "")
    csv_f = None
    writer = None
    if log_enabled:
        csv_f = open(args.log_csv, "w", newline="")
        writer = csv.writer(csv_f)
        header = [
            # wall-clock sync
            "t_epoch", "t_sync",
            # sim time (Gazebo)
            "t_sim", "t_sim_sec", "t_sim_nsec",
            # internal loop time
            "t_rel",
            "gust_active", "did_clear",
            "gx","gy","gz",
            "fx_pub","fy_pub","fz_pub",
            "ax_pub","ay_pub","az_pub",
        ]
        if args.log_traj == "on":
            header += [
                "px_des","py_des","pz_des",
                "vx_des","vy_des","vz_des",
                "ax_des","ay_des","az_des",
            ]
        writer.writerow(header)
        csv_f.flush()

    # OU gust state
    g = np.zeros(3, dtype=float)
    mu = np.array([args.gust_mu_x, args.gust_mu_y, args.gust_mu_z], dtype=float)
    f_prev = np.zeros(3, dtype=float)

    t_start = time.time()
    last_phase_active = None

    try:
        while running:
            t_epoch = time.time()
            t_sync = t_epoch - t0_epoch
            t_rel = t_epoch - t_start

            # gust on/off schedule
            cycle = args.gust_on_sec + args.gust_off_sec
            phase = (t_rel % cycle)
            gust_active = phase < args.gust_on_sec

            # OU update
            if gust_active:
                noise = rng.standard_normal(3)
                g = g + args.gust_theta * (mu - g) * dt + args.gust_sigma * math.sqrt(dt) * noise
            else:
                g = g + args.gust_theta * (mu - g) * dt

            fx = args.fx_mean + args.fx_amp * math.sin(2.0 * math.pi * args.freq_hz * t_rel) + g[0]
            fy = args.fy_mean + args.fy_amp * math.sin(2.0 * math.pi * args.freq_hz * t_rel + 1.3) + g[1]
            fz = args.fz_mean + g[2]

            F_raw = np.array([fx, fy, fz], dtype=float)
            F_cmd = clamp_vec(F_raw.copy(), float(args.fmax))
            F_cmd = clamp_components(F_cmd, float(args.fx_max), float(args.fy_max), float(args.fz_max))
            F_cmd = clamp_slew(F_cmd, f_prev, float(args.df_max_per_step))
            F_clamp = clamp_vec(F_cmd, float(args.fmax))

            did_clear = 0
            if (not gust_active) and args.mode == "persistent" and args.clear_when_off == "on":
                if last_phase_active is True or last_phase_active is None:
                    gz_clear_wrench(args.world, args.entity_id, entity_name=entity_name, entity_type=entity_type)
                    did_clear = 1
                Fx_pub, Fy_pub, Fz_pub = 0.0, 0.0, 0.0
                gz_pub_wrench(
                    topic, args.entity_id, Fx_pub, Fy_pub, Fz_pub,
                    entity_name=entity_name, entity_type=entity_type
                )
            else:
                Fx_pub, Fy_pub, Fz_pub = float(F_clamp[0]), float(F_clamp[1]), float(F_clamp[2])
                gz_pub_wrench(
                    topic, args.entity_id, Fx_pub, Fy_pub, Fz_pub,
                    entity_name=entity_name, entity_type=entity_type
                )

            # accel equivalent (ENU/world)
            m = max(float(args.mass_kg), 1e-9)
            ax_pub, ay_pub, az_pub = Fx_pub / m, Fy_pub / m, Fz_pub / m

            # sim time
            t_sim, t_sim_sec, t_sim_nsec = (None, None, None)
            if clock_reader is not None:
                t_sim, t_sim_sec, t_sim_nsec = clock_reader.latest()

            # desired traj (optional)
            traj_cols = []
            if args.log_traj == "on":
                p_des, v_des, a_des = desired_trajectory_enu(args, t_sync)  # uses synced time base
                traj_cols = [
                    float(p_des[0]), float(p_des[1]), float(p_des[2]),
                    float(v_des[0]), float(v_des[1]), float(v_des[2]),
                    float(a_des[0]), float(a_des[1]), float(a_des[2]),
                ]

            if log_enabled:
                writer.writerow([
                    float(t_epoch), float(t_sync),
                    ("" if t_sim is None else float(t_sim)),
                    ("" if t_sim_sec is None else int(t_sim_sec)),
                    ("" if t_sim_nsec is None else int(t_sim_nsec)),
                    float(t_rel),
                    int(gust_active), int(did_clear),
                    float(g[0]), float(g[1]), float(g[2]),
                    float(Fx_pub), float(Fy_pub), float(Fz_pub),
                    float(ax_pub), float(ay_pub), float(az_pub),
                    *traj_cols
                ])
                csv_f.flush()

            last_phase_active = gust_active
            f_prev = np.array([Fx_pub, Fy_pub, Fz_pub], dtype=float)
            time.sleep(dt)

    finally:
        print("[wind] clearing persistent wrench...")
        gz_clear_wrench(args.world, args.entity_id, entity_name=entity_name, entity_type=entity_type)
        if clock_reader is not None:
            clock_reader.stop()
        time.sleep(0.2)
        if log_enabled and csv_f is not None:
            csv_f.close()

if __name__ == "__main__":
    main()



# #!/usr/bin/env python3
# import argparse
# import math
# import signal
# import subprocess
# import time
# import numpy as np
# import csv


# running = True

# def handle_sig(sig, frame):
#     global running
#     running = False

# def gz_pub_wrench(topic, entity_id, fx, fy, fz):
#     payload = (
#         f"entity: {{id: {entity_id}}} "
#         f"wrench: {{force: {{x: {fx:.6f}, y: {fy:.6f}, z: {fz:.6f}}}}}"
#     )
#     subprocess.run(
#         ["gz", "topic", "-t", topic, "-m", "gz.msgs.EntityWrench", "-p", payload],
#         check=False,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

# def gz_clear_wrench(world, entity_id):
#     topic = f"/world/{world}/wrench/clear"
#     payload = f"id: {entity_id}"
#     subprocess.run(
#         ["gz", "topic", "-t", topic, "-m", "gz.msgs.Entity", "-p", payload],
#         check=False,
#         stdout=subprocess.DEVNULL,
#         stderr=subprocess.DEVNULL,
#     )

# def clamp_vec(f, fmax):
#     n = float(np.linalg.norm(f))
#     if n > fmax:
#         f = f * (fmax / (n + 1e-12))
#     return f

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--world", default="iris_runway")
#     parser.add_argument("--entity-id", type=int, required=True)
#     parser.add_argument("--rate-hz", type=float, default=20.0)

#     # Choose topic behavior
#     parser.add_argument("--mode", choices=["persistent", "oneshot"], default="persistent",
#                         help="persistent uses /wrench/persistent (recommended). oneshot uses /wrench (1 sim step).")
#     parser.add_argument("--clear-when-off", choices=["on","off"], default="on",
#                         help="when gust is off: clear persistent wrench (on) or publish zero force (off).")

#     # Mean bias (keep near 0 unless you intentionally want drift)
#     parser.add_argument("--fx-mean", type=float, default=0.0)
#     parser.add_argument("--fy-mean", type=float, default=0.0)
#     parser.add_argument("--fz-mean", type=float, default=0.0)

#     # Slow sinusoid (optional, can be 0)
#     parser.add_argument("--fx-amp",  type=float, default=0.0)
#     parser.add_argument("--fy-amp",  type=float, default=0.0)
#     parser.add_argument("--freq-hz", type=float, default=0.1)

#     # Mean-reverting gust (OU process)
#     parser.add_argument("--gust-sigma", type=float, default=2.0, help="noise intensity (N)")
#     parser.add_argument("--gust-theta", type=float, default=0.8, help="reversion rate (1/s), higher = faster decay")
#     parser.add_argument("--gust-mu-x", type=float, default=0.0)
#     parser.add_argument("--gust-mu-y", type=float, default=0.0)
#     parser.add_argument("--gust-mu-z", type=float, default=0.0)

#     # Intermittency (gust bursts)
#     parser.add_argument("--gust-on-sec", type=float, default=3.0)
#     parser.add_argument("--gust-off-sec", type=float, default=3.0)

#     # Safety bound
#     parser.add_argument("--fmax", type=float, default=8.0, help="max |force| magnitude (N)")
#     parser.add_argument("--log-csv", default="wind_log.csv",
#                     help="CSV path for wind logs. Use '' to disable.")


#     args = parser.parse_args()

#     log_enabled = (args.log_csv is not None and args.log_csv.strip() != "")
#     csv_f = None
#     writer = None

#     if log_enabled:
#         csv_f = open(args.log_csv, "w", newline="")
#         writer = csv.writer(csv_f)
#         writer.writerow([
#             "t",
#             "gust_active",
#             "did_clear",
#             "gx","gy","gz",
#             "fx_raw","fy_raw","fz_raw",
#             "fx_clamp","fy_clamp","fz_clamp",
#             "fx_pub","fy_pub","fz_pub",
#         ])
#         csv_f.flush()


#     signal.signal(signal.SIGINT, handle_sig)
#     signal.signal(signal.SIGTERM, handle_sig)

#     dt = 1.0 / max(args.rate_hz, 1e-6)
#     rng = np.random.default_rng(0)

#     topic = f"/world/{args.world}/wrench/persistent" if args.mode == "persistent" else f"/world/{args.world}/wrench"
#     print(f"[wind] topic={topic} entity_id={args.entity_id} rate={args.rate_hz}Hz fmax={args.fmax}N")
#     if args.mode == "oneshot":
#         print("[wind] NOTE: oneshot applies force for ONE sim step; needs very high publish rate to be noticeable.")

#     # OU gust state
#     g = np.zeros(3, dtype=float)
#     mu = np.array([args.gust_mu_x, args.gust_mu_y, args.gust_mu_z], dtype=float)

#     t0 = time.time()
#     last_phase_active = None

#     try:
#         while running:
#             t = time.time() - t0

#             cycle = args.gust_on_sec + args.gust_off_sec
#             phase = (t % cycle)
#             gust_active = phase < args.gust_on_sec

#             # OU update:
#             # dg = theta*(mu - g)*dt + sigma*sqrt(dt)*N(0,1)
#             if gust_active:
#                 noise = rng.standard_normal(3)
#                 g = g + args.gust_theta * (mu - g) * dt + args.gust_sigma * math.sqrt(dt) * noise
#             else:
#                 # decay toward mu with no noise
#                 g = g + args.gust_theta * (mu - g) * dt

#             # optional sin components (still zero-mean if mean=0)
#             fx = args.fx_mean + args.fx_amp * math.sin(2.0 * math.pi * args.freq_hz * t) + g[0]
#             fy = args.fy_mean + args.fy_amp * math.sin(2.0 * math.pi * args.freq_hz * t + 1.3) + g[1]
#             fz = args.fz_mean + g[2]

#             # F = clamp_vec(np.array([fx, fy, fz], dtype=float), float(args.fmax))

#             # # If gust is OFF, either clear or explicitly publish 0 to avoid “stuck pushing”
#             # if (not gust_active) and args.mode == "persistent" and args.clear_when_off == "on":
#             #     if last_phase_active is True or last_phase_active is None:
#             #         gz_clear_wrench(args.world, args.entity_id)
#             #     # still publish zeros occasionally to keep things consistent
#             #     gz_pub_wrench(topic, args.entity_id, 0.0, 0.0, 0.0)
#             # else:
#             #     gz_pub_wrench(topic, args.entity_id, float(F[0]), float(F[1]), float(F[2]))
#             F_raw = np.array([fx, fy, fz], dtype=float)
#             F_clamp = clamp_vec(F_raw.copy(), float(args.fmax))

#             did_clear = 0
#             # Decide what we actually publish this step
#             if (not gust_active) and args.mode == "persistent" and args.clear_when_off == "on":
#                 if last_phase_active is True or last_phase_active is None:
#                     gz_clear_wrench(args.world, args.entity_id)
#                     did_clear = 1
#                 Fx_pub, Fy_pub, Fz_pub = 0.0, 0.0, 0.0
#                 gz_pub_wrench(topic, args.entity_id, Fx_pub, Fy_pub, Fz_pub)
#             else:
#                 Fx_pub, Fy_pub, Fz_pub = float(F_clamp[0]), float(F_clamp[1]), float(F_clamp[2])
#                 gz_pub_wrench(topic, args.entity_id, Fx_pub, Fy_pub, Fz_pub)

#             # ---- CSV log ----
#             if log_enabled:
#                 writer.writerow([
#                     float(t),
#                     int(gust_active),
#                     int(did_clear),
#                     float(g[0]), float(g[1]), float(g[2]),
#                     float(F_raw[0]), float(F_raw[1]), float(F_raw[2]),
#                     float(F_clamp[0]), float(F_clamp[1]), float(F_clamp[2]),
#                     float(Fx_pub), float(Fy_pub), float(Fz_pub),
#                 ])
#                 csv_f.flush()

#             last_phase_active = gust_active
#             time.sleep(dt)

#     finally:
#         print("[wind] clearing persistent wrench...")
#         gz_clear_wrench(args.world, args.entity_id)
#         time.sleep(0.2)
#         if log_enabled and csv_f is not None:
#             csv_f.close()


# if __name__ == "__main__":
#     main()
