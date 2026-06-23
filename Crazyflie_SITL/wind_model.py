

"""wind_model.py

Fan-jet wind model usable in any simulator (Gazebo / PyBullet / etc).

- wind_velocity_world(...) and aero_force_world(...) are simulator-agnostic.
- apply_to_body(...) is PyBullet-only convenience and is guarded so it won't
  crash if pybullet isn't installed.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

# -------------------- Optional PyBullet dependency --------------------
# We keep the module importable even if pybullet is not installed.
try:
    import pybullet as p  # type: ignore
    _HAVE_PYBULLET = True
except Exception:
    p = None  # type: ignore
    _HAVE_PYBULLET = False
# ---------------------------------------------------------------------


def _unit(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


@dataclass
class FanJetParams:
    # Fan pose (world frame)
    origin_w: np.ndarray               # (3,) fan outlet origin position (m)
    axis_w: np.ndarray                 # (3,) unit vector pointing downstream

    # Mean jet parameters
    u0: float = 2.0                   # (m/s) nominal centerline speed at outlet (x=0)
    x0: float = 0.20                   # (m) virtual origin
    sigma0: float = 0.10               # (m) jet radius (1-sigma) at outlet
    spread_k: float = 0.18             # (-) sigma(x) = sigma0 + spread_k * x

    # Turbulence (additive velocity noise)
    turb_sigma: float = 0.8            # (m/s)
    turb_tau: float = 0.35             # (s) OU correlation time

    # Optional periodic modulation
    amp_sine: float = 0.10             # (-) ±10%
    freq_sine: float = 1.2             # (Hz)

    # Aerodynamics
    rho: float = 1.225                 # (kg/m^3)
    cdA: float = 0.003                 # (m^2) lumped Cd*A

    # Safety clamps
    max_wind_speed: float = 12.0       # (m/s)
    max_force: float = 0.02             # (N)

    # Spatial gating
    r_cut_mult: float = 3.0            # r_cut(x) = r_cut_mult * sigma(x)


class FanJetWindModel:
    """Localized fan-jet wind field with temporally correlated turbulence."""

    def __init__(self, params: Optional[FanJetParams] = None, seed: Optional[int] = None):
        if params is None:
            params = FanJetParams(
                origin_w=np.array([0.0, 0.0, 0.5], dtype=float),
                axis_w=np.array([1.0, 0.0, 0.0], dtype=float),
            )
        params.origin_w = np.array(params.origin_w, dtype=float).reshape(3)
        params.axis_w = _unit(np.array(params.axis_w, dtype=float).reshape(3))
        self.p = params

        self._rng = np.random.default_rng(seed)
        self._ou_v = np.zeros(3, dtype=float)
        self._gust_time_left = 0.0
        self._gust_v = np.zeros(3, dtype=float)

    def set_pose(self, origin_w: np.ndarray, axis_w: np.ndarray) -> None:
        self.p.origin_w = np.array(origin_w, dtype=float).reshape(3)
        self.p.axis_w = _unit(np.array(axis_w, dtype=float).reshape(3))

    def reset_episode(self, seed: Optional[int] = None) -> None:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._ou_v[:] = 0.0
        self._gust_time_left = 0.0
        self._gust_v[:] = 0.0

    # -------------------------- wind field --------------------------

    def _sample_ou(self, dt: float) -> np.ndarray:
        tau = max(self.p.turb_tau, 1e-6)
        sigma = max(self.p.turb_sigma, 0.0)
        dW = self._rng.standard_normal(3) * np.sqrt(max(dt, 0.0))
        self._ou_v += (-self._ou_v / tau) * dt + sigma * dW
        return self._ou_v

    def _maybe_trigger_gust(self, dt: float, rate_hz: float = 0.15) -> None:
        if self._gust_time_left > 0.0:
            self._gust_time_left = max(0.0, self._gust_time_left - dt)
            if self._gust_time_left == 0.0:
                self._gust_v[:] = 0.0
            return

        if self._rng.random() < 1.0 - np.exp(-rate_hz * dt):
            dur = self._rng.uniform(0.10, 0.35)
            mag = self._rng.uniform(0.6, 1.8)
            lateral = _unit(self._rng.standard_normal(3))
            axis = self.p.axis_w
            lateral -= np.dot(lateral, axis) * axis
            lateral = _unit(lateral)
            self._gust_v = mag * (0.85 * axis + 0.15 * lateral)
            self._gust_time_left = dur

    def wind_velocity_world(self, pos_w: np.ndarray, t: float, dt: float) -> np.ndarray:
        pos_w = np.array(pos_w, dtype=float).reshape(3)
        o = self.p.origin_w
        a = self.p.axis_w

        rel = pos_w - o
        x = float(np.dot(rel, a))
        if x <= 0.0:
            return np.zeros(3, dtype=float)

        rel_perp = rel - x * a
        r = float(np.linalg.norm(rel_perp))

        sigma = max(self.p.sigma0 + self.p.spread_k * x, 1e-6)
        if r > self.p.r_cut_mult * sigma:
            return np.zeros(3, dtype=float)

        u_c = self.p.u0 * (self.p.x0 / (self.p.x0 + x))
        u_mean = u_c * np.exp(-(r * r) / (2.0 * sigma * sigma))

        if self.p.amp_sine > 0.0 and self.p.freq_sine > 0.0:
            u_mean *= (1.0 + self.p.amp_sine * np.sin(2.0 * np.pi * self.p.freq_sine * t))

        v_turb = self._sample_ou(dt)
        self._maybe_trigger_gust(dt)

        v = u_mean * a + v_turb + self._gust_v

        spd = float(np.linalg.norm(v))
        if spd > self.p.max_wind_speed:
            v = v * (self.p.max_wind_speed / spd)

        return v

    # -------------------------- force model --------------------------

    def aero_force_world(self, wind_v_w: np.ndarray, body_v_w: np.ndarray) -> np.ndarray:
        wind_v_w = np.array(wind_v_w, dtype=float).reshape(3)
        body_v_w = np.array(body_v_w, dtype=float).reshape(3)

        v_rel = wind_v_w - body_v_w
        s = float(np.linalg.norm(v_rel))
        if s < 1e-6:
            return np.zeros(3, dtype=float)

        f = 0.5 * self.p.rho * self.p.cdA * s * v_rel

        fn = float(np.linalg.norm(f))
        if fn > self.p.max_force:
            f = f * (self.p.max_force / fn)

        return f

    # -------------------------- PyBullet helper --------------------------

    def apply_to_body(
        self,
        client_id: int,
        body_id: int,
        t: float,
        dt: float,
        link_index: int = -1,
        body_pos_w: Optional[np.ndarray] = None,
        body_vel_w: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """PyBullet-only: apply wind disturbance to a rigid body.

        Returns (wind_velocity_world, applied_force_world).

        For Gazebo usage, do NOT call this; instead call:
          wind_velocity_world(...) and aero_force_world(...)
        and publish the resulting force as an EntityWrench.
        """
        if not _HAVE_PYBULLET or p is None:
            raise RuntimeError(
                "apply_to_body() is PyBullet-only, but pybullet is not available. "
                "Install pybullet or use wind_velocity_world/aero_force_world for Gazebo."
            )

        if body_pos_w is None:
            body_pos_w, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=client_id)
        if body_vel_w is None:
            body_vel_w, _ = p.getBaseVelocity(body_id, physicsClientId=client_id)

        wind_v = self.wind_velocity_world(body_pos_w, t=t, dt=dt)
        f_w = self.aero_force_world(wind_v, body_vel_w)

        p.applyExternalForce(
            objectUniqueId=body_id,
            linkIndex=link_index,
            forceObj=f_w.tolist(),
            posObj=list(body_pos_w),
            flags=p.WORLD_FRAME,
            physicsClientId=client_id,
        )

        return wind_v, f_w
