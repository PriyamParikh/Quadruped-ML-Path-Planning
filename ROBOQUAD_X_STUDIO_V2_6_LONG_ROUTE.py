"""
ROBOQUAD FK V1.9 — Analysis Analysis Studio + Path Planning Algorithms
================================================================

Purpose
-------
A focused machine-learning GUI for a 12-DOF robot dog. The user defines a
2D path using waypoints, trains several supervised regression algorithms, and
then lets the learned controller generate a sequence of forward/backward gait
commands while the 12 joints animate through equation-based trot cycles.

Important modelling note
------------------------
This is a kinematic ML controller. The floating body translation is prescribed
by the learned command sequence; it is not derived from foot-ground forces,
friction, rigid-body dynamics, or inverse kinematics.

Coordinate convention
---------------------
World/body when yaw = 0:
    +X = forward
    +Y = left
    +Z = up

Leg joints:
    q1 = hip ab/ad, local X
    q2 = hip flex/ext, local Y
    q3 = knee flex/ext, local Y

The ML controller observes the target waypoint relative to the moving body.
A geometric front/rear supervisor chooses Forward vs Backward based on the
sign of longitudinal body-frame error. The ML regressor predicts:
    1) travel magnitude per gait cycle [m]
    2) yaw correction per gait cycle [deg]
    3) q2 fore-aft gait amplitude [deg]
    4) q3 knee-lift amplitude [deg]

Algorithms trained:
    Random Forest, Extra Trees, KNN, MLP

V1.8 analysis modules:
    Workspace, trajectory, trajectory planning, live singularity, gait,
    ML algorithm comparison and error-analysis figures. All are exportable.
"""

import csv
import json
import math
import tkinter as tk
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tkinter import ttk, filedialog, messagebox

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import joblib
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except Exception:
    joblib = None
    SKLEARN_AVAILABLE = False


APP_TITLE = "ROBOQUAD-X Studio — Intelligent Quadruped Navigation & Motion Lab"
LEG_ORDER = ("FL", "FR", "RL", "RR")
LEG_NAMES = {
    "FL": "Front Left",
    "FR": "Front Right",
    "RL": "Rear Left",
    "RR": "Rear Right",
}
JOINTS = ("q1", "q2", "q3")
JOINT_LIMITS = {
    "q1": (-45.0, 45.0),
    "q2": (-100.0, 80.0),
    "q3": (-150.0, 10.0),
}
STAND = {leg: {"q1": 0.0, "q2": 25.0, "q3": -55.0} for leg in LEG_ORDER}


# ---------------------------------------------------------------------------
# Mathematics / FK
# ---------------------------------------------------------------------------

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0, 0],
        [0, c, -s, 0],
        [0, s, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0, s, 0],
        [0, 1, 0, 0],
        [-s, 0, c, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0, 0],
        [s, c, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)


def trans(x, y, z):
    T = np.eye(4, dtype=float)
    T[:3, 3] = [x, y, z]
    return T


def wrap_angle(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def quintic(u):
    u = min(max(float(u), 0.0), 1.0)
    return 10*u**3 - 15*u**4 + 6*u**5


@dataclass
class Geometry:
    # Compact dog-like torso plus independently editable leg link lengths.
    body_length: float = 0.56
    body_width: float = 0.25
    body_height: float = 0.12
    FL_upper: float = 0.115
    FL_lower: float = 0.125
    FR_upper: float = 0.115
    FR_lower: float = 0.125
    RL_upper: float = 0.115
    RL_lower: float = 0.125
    RR_upper: float = 0.115
    RR_lower: float = 0.125

    def upper(self, leg):
        return float(getattr(self, f"{leg}_upper"))

    def lower(self, leg):
        return float(getattr(self, f"{leg}_lower"))


class QuadrupedFK:
    def __init__(self, geometry=None):
        self.g = geometry or Geometry()

    def hip_anchor(self, leg):
        x = self.g.body_length/2 if leg.startswith("F") else -self.g.body_length/2
        y = self.g.body_width/2 if leg.endswith("L") else -self.g.body_width/2
        return np.array([x, y, 0.0], dtype=float)

    def leg_points(self, leg, q):
        q1, q2, q3 = [math.radians(float(q[j])) for j in JOINTS]
        a = self.hip_anchor(leg)
        T0 = trans(*a)
        T1 = T0 @ rot_x(q1)
        T2 = T1 @ rot_y(q2)
        Tk = T2 @ trans(0, 0, -self.g.upper(leg))
        T3 = Tk @ rot_y(q3)
        Tf = T3 @ trans(0, 0, -self.g.lower(leg))
        return T0[:3, 3].copy(), Tk[:3, 3].copy(), Tf[:3, 3].copy()

    def jacobian_numeric(self, leg, q, eps_rad=1e-5):
        """Numerical 3x3 foot-position Jacobian [m/rad]."""
        J = np.zeros((3, 3), dtype=float)
        eps_deg = math.degrees(float(eps_rad))
        for col, joint in enumerate(JOINTS):
            qp = dict(q); qm = dict(q)
            qp[joint] = float(qp[joint]) + eps_deg
            qm[joint] = float(qm[joint]) - eps_deg
            pp = self.leg_points(leg, qp)[2]
            pm = self.leg_points(leg, qm)[2]
            J[:, col] = (pp - pm) / (2.0*float(eps_rad))
        return J

    def singularity_metrics(self, leg, q):
        J = self.jacobian_numeric(leg, q)
        sv = np.linalg.svd(J, compute_uv=False)
        sigma_min = float(np.min(sv))
        sigma_max = float(np.max(sv))
        cond = float('inf') if sigma_min < 1e-12 else sigma_max/sigma_min
        return {
            'J': J,
            'det': float(np.linalg.det(J)),
            'rank': int(np.linalg.matrix_rank(J, tol=1e-8)),
            'sigma_min': sigma_min,
            'sigma_max': sigma_max,
            'condition': cond,
        }

    def all_singularity_metrics(self, q_all):
        return {leg: self.singularity_metrics(leg, q_all[leg]) for leg in LEG_ORDER}

    def all_points(self, q_all):
        return {leg: self.leg_points(leg, q_all[leg]) for leg in LEG_ORDER}


# ---------------------------------------------------------------------------
# ML controller core — separately testable from GUI
# ---------------------------------------------------------------------------

def generate_training_data(n=4000, max_step=0.08, max_yaw_deg=12.0, seed=42):
    """
    Synthetic teacher dataset for supervised imitation of a kinematic path
    controller. Direction itself is supervised geometrically by ex_body sign;
    the ML model learns the continuous command magnitudes.

    Features:
        abs_ex_body, effective_ey_body, distance, effective_yaw_error, dir_sign

    Targets:
        travel_magnitude, yaw_step_deg, q2_amplitude_deg, q3_lift_deg
    """
    n = max(500, int(n))
    rng = np.random.default_rng(int(seed))
    ex = rng.uniform(-1.5, 1.5, n)
    ey = rng.uniform(-1.0, 1.0, n)
    dist = np.hypot(ex, ey)
    raw = np.arctan2(ey, ex)
    sign = np.where(ex >= 0.0, 1.0, -1.0)

    eff_yaw = raw.copy()
    back_idx = np.where(sign < 0.0)[0]
    eff_yaw[back_idx] = np.array([wrap_angle(v - math.pi) for v in raw[back_idx]])

    # Rear-view lateral error is mirrored into the effective travel frame.
    eff_ey = np.where(sign > 0.0, ey, -ey)

    alignment = np.clip(np.cos(eff_yaw), 0.15, 1.0)
    travel = np.minimum(
        max_step,
        np.maximum(0.005, 0.55 * dist * alignment)
    )

    yaw_step = np.clip(np.degrees(eff_yaw) * 0.65, -max_yaw_deg, max_yaw_deg)
    effort = np.clip(travel / max_step, 0.0, 1.0)
    lateral = np.clip(np.abs(eff_ey) / np.maximum(dist, 1e-6), 0.0, 1.0)

    q2_amp = np.clip(10.0 + 12.0*effort + 5.0*lateral, 8.0, 30.0)
    q3_lift = np.clip(20.0 + 8.0*effort + 5.0*lateral, 18.0, 36.0)

    X = np.column_stack([np.abs(ex), eff_ey, dist, eff_yaw, sign])
    y = np.column_stack([travel, yaw_step, q2_amp, q3_lift])
    return X, y


def train_ml_models(X, y, test_fraction=0.20, seed=42):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is not installed")

    test_fraction = min(max(float(test_fraction), 0.10), 0.40)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_fraction, random_state=int(seed)
    )

    models = {
        "Random Forest": RandomForestRegressor(
            n_estimators=160, random_state=int(seed), n_jobs=-1, min_samples_leaf=1
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=180, random_state=int(seed), n_jobs=-1, min_samples_leaf=1
        ),
        "KNN": Pipeline([
            ("scale", StandardScaler()),
            ("reg", KNeighborsRegressor(n_neighbors=7, weights="distance")),
        ]),
        "MLP": Pipeline([
            ("scale", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=(64, 32), activation="relu",
                max_iter=900, early_stopping=True, random_state=int(seed)
            )),
        ]),
    }

    results = {}
    trained = {}
    target_std = np.std(y_test, axis=0) + 1e-12

    for name, model in models.items():
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        rmse_each = np.sqrt(np.mean((y_test - p)**2, axis=0))
        r2_each = np.array([
            r2_score(y_test[:, i], p[:, i]) for i in range(y_test.shape[1])
        ])
        normalized_rmse = float(np.mean(rmse_each / target_std))
        results[name] = {
            "travel_rmse": float(rmse_each[0]),
            "yaw_rmse": float(rmse_each[1]),
            "q2_rmse": float(rmse_each[2]),
            "q3_rmse": float(rmse_each[3]),
            "mean_r2": float(np.mean(r2_each)),
            "nrmse": normalized_rmse,
        }
        trained[name] = model

    best_name = min(results, key=lambda k: results[k]["nrmse"])
    return trained, results, best_name, trained[best_name], len(X_train), len(X_test)


def state_features(x, y, yaw, tx, ty):
    """
    Legacy ML feature mapping retained for model compatibility.
    """
    dx = float(tx) - float(x)
    dy = float(ty) - float(y)
    c, s = math.cos(yaw), math.sin(yaw)
    ex = c*dx + s*dy
    ey = -s*dx + c*dy
    dist = math.hypot(dx, dy)
    sign = 1.0 if ex >= 0.0 else -1.0
    raw = math.atan2(ey, ex)
    eff_yaw = raw if sign > 0 else wrap_angle(raw - math.pi)
    eff_ey = ey if sign > 0 else -ey
    f = np.array([abs(ex), eff_ey, dist, eff_yaw, sign], dtype=float)
    return f, sign, dist


def forward_only_features(x, y, yaw, tx, ty):
    """
    Features for the turn-first policy.

    Direction is deliberately fixed to +1 because normal waypoint tracking
    must turn to face the next target and then move FORWARD.
    """
    dx = float(tx) - float(x)
    dy = float(ty) - float(y)
    c, s = math.cos(yaw), math.sin(yaw)
    ex = c*dx + s*dy
    ey = -s*dx + c*dy
    dist = math.hypot(dx, dy)
    yaw_error = wrap_angle(math.atan2(dy, dx) - yaw)
    f = np.array([abs(ex), ey, dist, yaw_error, 1.0], dtype=float)
    return f, dist, yaw_error


def head_xy_from_body(x, y, yaw, head_offset):
    """World XY of the tracked nose/head point."""
    return (
        float(x) + math.cos(float(yaw))*float(head_offset),
        float(y) + math.sin(float(yaw))*float(head_offset),
    )


def plan_ml_waypoints(model, waypoints, start=(0.0, 0.0, 0.0),
                      max_step=0.08, max_yaw_deg=12.0,
                      tolerance=0.035, max_commands=160,
                      head_offset=0.40,
                      turn_threshold_deg=5.0,
                      allow_reverse=False):
    """
    Turn-first / head-tracked path planner.

    Recommended policy (allow_reverse=False):
      1. A waypoint is evaluated against the HEAD/Nose point.
      2. If the next target is not aligned, rotate first.
      3. Turning pivots around the current head point, so the head remains at
         the current destination while the body changes orientation.
      4. After alignment, move forward only.

    Automatic reverse remains optional for experiments, but is disabled by
    default because it can cause the dog to back into the next destination.
    """
    x, y, yaw = map(float, start)
    commands = []
    body_path = [(x, y)]
    hx, hy = head_xy_from_body(x, y, yaw, head_offset)
    head_path = [(hx, hy)]
    wp_index = 0

    max_step = max(0.01, float(max_step))
    max_yaw_deg = max(1.0, float(max_yaw_deg))
    tolerance = max(0.005, float(tolerance))
    turn_threshold = math.radians(max(0.5, float(turn_threshold_deg)))
    head_offset = max(0.01, float(head_offset))

    # Treat the first waypoint as the path start anchor when it coincides with
    # the initial BODY centre. This preserves the GUI's existing waypoint style
    # where the first point is commonly (0, 0).
    if waypoints:
        if math.hypot(
            float(waypoints[0][0])-x,
            float(waypoints[0][1])-y
        ) <= tolerance:
            wp_index = 1

    while wp_index < len(waypoints) and len(commands) < int(max_commands):
        tx, ty = map(float, waypoints[wp_index])

        hx, hy = head_xy_from_body(x, y, yaw, head_offset)
        head_error = math.hypot(tx-hx, ty-hy)

        if head_error <= tolerance:
            wp_index += 1
            continue

        # Desired orientation is based on the line from the CURRENT HEAD to the
        # next destination. The turn itself is performed around the head pivot.
        desired_yaw = math.atan2(ty-hy, tx-hx)
        yaw_error = wrap_angle(desired_yaw - yaw)

        # Get ML amplitudes / travel magnitude. For the normal policy, use a
        # forward-only feature vector; geometric logic handles turn direction.
        f, _, _ = forward_only_features(x, y, yaw, tx, ty)
        pred = np.asarray(model.predict(f.reshape(1, -1))).reshape(-1)
        ml_travel = float(np.clip(pred[0], 0.005, max_step))
        q2_amp = float(np.clip(pred[2], 8.0, 32.0))
        q3_lift = float(np.clip(pred[3], 16.0, 40.0))

        if allow_reverse:
            # Optional experimental mode. Still tracks the HEAD rather than the
            # body centre, but can choose reverse when the target is behind.
            c, s = math.cos(yaw), math.sin(yaw)
            ex_h = c*(tx-hx) + s*(ty-hy)
            if ex_h < 0.0 and abs(yaw_error) > math.pi/2:
                direction = "Backward"
                travel = min(ml_travel, head_error)
                nx = x - math.cos(yaw)*travel
                ny = y - math.sin(yaw)*travel
                nyaw = yaw
                commands.append({
                    "target_index": wp_index,
                    "direction": direction,
                    "travel_m": travel,
                    "yaw_step_deg": 0.0,
                    "q2_amp_deg": q2_amp,
                    "q3_lift_deg": q3_lift,
                    "start_x": x, "start_y": y, "start_yaw": yaw,
                    "end_x": nx, "end_y": ny, "end_yaw": nyaw,
                    "target_x": tx, "target_y": ty,
                    "error_before_m": head_error,
                    "head_start_x": hx, "head_start_y": hy,
                    "pivot_x": hx, "pivot_y": hy,
                })
                x, y, yaw = nx, ny, nyaw
                body_path.append((x, y))
                head_path.append(head_xy_from_body(x, y, yaw, head_offset))
                continue

        # TURN FIRST. No translational shortcut is allowed.
        if abs(yaw_error) > turn_threshold:
            yaw_step_deg = float(np.clip(
                math.degrees(yaw_error),
                -max_yaw_deg, +max_yaw_deg
            ))
            nyaw = wrap_angle(yaw + math.radians(yaw_step_deg))

            # Pivot around the current HEAD point. This keeps the nose at the
            # previous destination while the body swings around to face the
            # next destination.
            nx = hx - math.cos(nyaw)*head_offset
            ny = hy - math.sin(nyaw)*head_offset

            direction = "Turn Left" if yaw_step_deg > 0.0 else "Turn Right"

            commands.append({
                "target_index": wp_index,
                "direction": direction,
                "travel_m": 0.0,
                "yaw_step_deg": yaw_step_deg,
                "q2_amp_deg": max(12.0, q2_amp*0.80),
                "q3_lift_deg": q3_lift,
                "start_x": x, "start_y": y, "start_yaw": yaw,
                "end_x": nx, "end_y": ny, "end_yaw": nyaw,
                "target_x": tx, "target_y": ty,
                "error_before_m": head_error,
                "head_start_x": hx, "head_start_y": hy,
                "pivot_x": hx, "pivot_y": hy,
            })

            x, y, yaw = nx, ny, nyaw
            body_path.append((x, y))
            head_path.append((hx, hy))  # head remains fixed during the turn
            continue

        # ALIGNED: move FORWARD only. Head and body translate together.
        travel = min(ml_travel, head_error)
        nx = x + math.cos(yaw)*travel
        ny = y + math.sin(yaw)*travel
        nyaw = yaw

        commands.append({
            "target_index": wp_index,
            "direction": "Forward",
            "travel_m": travel,
            "yaw_step_deg": 0.0,
            "q2_amp_deg": q2_amp,
            "q3_lift_deg": q3_lift,
            "start_x": x, "start_y": y, "start_yaw": yaw,
            "end_x": nx, "end_y": ny, "end_yaw": nyaw,
            "target_x": tx, "target_y": ty,
            "error_before_m": head_error,
            "head_start_x": hx, "head_start_y": hy,
            "pivot_x": hx, "pivot_y": hy,
        })

        x, y, yaw = nx, ny, nyaw
        body_path.append((x, y))
        head_path.append(head_xy_from_body(x, y, yaw, head_offset))

    reached = wp_index >= len(waypoints)
    hx, hy = head_xy_from_body(x, y, yaw, head_offset)
    final_error = 0.0
    if waypoints:
        final_error = math.hypot(
            float(waypoints[-1][0])-hx,
            float(waypoints[-1][1])-hy
        )

    return {
        "commands": commands,
        "path": head_path,          # backwards-compatible: planned HEAD path
        "head_path": head_path,
        "body_path": body_path,
        "reached": reached,
        "final_error_m": final_error,
        "final_pose": (x, y, yaw),
        "final_head": (hx, hy),
        "waypoints_reached": wp_index,
        "head_offset_m": head_offset,
    }


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scrollable control container
# ---------------------------------------------------------------------------

class ScrollableFrame(ttk.Frame):
    def __init__(self, parent, width=None):
        super().__init__(parent)
        self.canvas = tk.Canvas(self, highlightthickness=0, width=width)
        self.vbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.inner = ttk.Frame(self.canvas)
        self.window = self.canvas.create_window((0, 0), window=self.inner, anchor='nw')
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.inner.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all')))
        self.canvas.bind('<Configure>', lambda e: self.canvas.itemconfigure(self.window, width=e.width))
        self.canvas.pack(side='left', fill='both', expand=True)
        self.vbar.pack(side='right', fill='y')
        self.canvas.bind('<Enter>', self._bind_wheel)
        self.canvas.bind('<Leave>', self._unbind_wheel)

    def _bind_wheel(self, _event=None):
        self.canvas.bind_all('<MouseWheel>', self._wheel)

    def _unbind_wheel(self, _event=None):
        self.canvas.unbind_all('<MouseWheel>')

    def _wheel(self, event):
        delta = -1 if event.delta > 0 else 1
        self.canvas.yview_scroll(delta*3, 'units')

class MLRobotDogApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1580x940")
        self.minsize(1220, 760)

        self.fk = QuadrupedFK()
        self.joint_angles = {leg: dict(STAND[leg]) for leg in LEG_ORDER}
        self.prev_joint_angles = {leg: dict(STAND[leg]) for leg in LEG_ORDER}
        self.joint_vel = {leg: {j: 0.0 for j in JOINTS} for leg in LEG_ORDER}
        self.joint_acc = {leg: {j: 0.0 for j in JOINTS} for leg in LEG_ORDER}
        self.prev_joint_vel = {leg: {j: 0.0 for j in JOINTS} for leg in LEG_ORDER}

        self.waypoints = [(0.0, 0.0), (0.5, 0.0), (1.0, 0.0), (0.45, 0.0), (-0.15, 0.0)]
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.plan = None

        self.base_x = 0.0
        self.base_y = 0.0
        self.base_yaw = 0.0
        self.base_z = 0.0
        self.ground_z = -0.32
        self.body_trail = []
        self.head_trail = []
        self.sim_log = []

        self.sim_running = False
        self.sim_paused = False
        self.sim_job = None
        self.command_index = 0
        self.command_elapsed = 0.0
        self.sim_time = 0.0
        self.current_command = None
        self.command_start_pose = None

        # GUI vars
        self.status_var = tk.StringVar(value="Ready. Train a model, plan a path, then simulate.")
        self.best_var = tk.StringVar(value="Best ML model: —")
        self.path_var = tk.StringVar(value="Path status: —")
        self.live_dir_var = tk.StringVar(value="Direction: —")
        self.live_pose_var = tk.StringVar(value="Body: X=0.000 m, Y=0.000 m, yaw=0.0°")
        self.live_target_var = tk.StringVar(value="Target: —")
        self.live_error_var = tk.StringVar(value="Waypoint error: —")
        self.live_command_var = tk.StringVar(value="ML command: —")

        self.dataset_var = tk.IntVar(value=5000)
        self.test_var = tk.DoubleVar(value=0.20)
        self.seed_var = tk.IntVar(value=42)
        self.max_step_var = tk.DoubleVar(value=0.08)
        self.max_yaw_var = tk.DoubleVar(value=12.0)
        self.tolerance_var = tk.DoubleVar(value=0.035)
        self.max_commands_var = tk.IntVar(value=160)
        self.cycle_time_var = tk.DoubleVar(value=0.65)
        self.duty_var = tk.DoubleVar(value=0.55)
        self.plane_size_var = tk.DoubleVar(value=4.0)
        self.playback_speed_var = tk.DoubleVar(value=3.0)
        self.turn_threshold_var = tk.DoubleVar(value=5.0)
        self.path_policy_var = tk.StringVar(value="Turn First + Forward")
        self.start_x_var = tk.DoubleVar(value=0.0)
        self.start_y_var = tk.DoubleVar(value=0.0)
        self.start_yaw_var = tk.DoubleVar(value=0.0)
        self.wp_x_var = tk.DoubleVar(value=1.25)
        self.wp_y_var = tk.DoubleVar(value=0.0)

        # Geometry editing: each leg has independent upper/lower link lengths.
        g = self.fk.g
        self.body_length_var = tk.DoubleVar(value=g.body_length)
        self.body_width_var = tk.DoubleVar(value=g.body_width)
        self.body_height_var = tk.DoubleVar(value=g.body_height)
        self.leg_upper_vars = {leg: tk.DoubleVar(value=g.upper(leg)) for leg in LEG_ORDER}
        self.leg_lower_vars = {leg: tk.DoubleVar(value=g.lower(leg)) for leg in LEG_ORDER}

        # Workspace / analysis graph controls.
        self.workspace_leg_var = tk.StringVar(value='ALL')
        self.workspace_mode_var = tk.StringVar(value='Full leg workspace')
        self.workspace_samples_var = tk.IntVar(value=11)
        self.workspace_cache = []

        self.gait_plot_direction_var = tk.StringVar(value='Forward')
        self.live_singularity_var = tk.StringVar(value='Singularity: —')

        # 12-DOF analysis graph controls.
        self.joint_plot_leg_var = tk.StringVar(value='FL')
        self.joint_plot_joint_var = tk.StringVar(value='q2')
        self.joint_plot_live_var = tk.BooleanVar(value=True)
        self.sim_initial_joint_angles = {
            leg: dict(STAND[leg]) for leg in LEG_ORDER
        }

        self.graph_frame_counter = 0
        self._last_singularity_metrics = None

        self._build_style()
        self._build_ui()
        self._refresh_waypoints()
        self._update_3d()
        self._update_path_plot()
        self.update_singularity_plot()
        self.update_gait_plot()
        self.update_error_plot()

    def _build_style(self):
        style = ttk.Style(self)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 10, "bold"))

    def _build_ui(self):
        self.geometry("1710x980")
        self.minsize(1280, 760)

        banner = ttk.Frame(self, padding=(10, 7))
        banner.pack(fill='x')
        ttk.Label(banner, text='ROBOQUAD-X Studio — Intelligent Quadruped Navigation & Motion Lab', style='Header.TLabel').pack(side='left')
        ttk.Label(banner, text='  |  Developed by Dr. Priyam Parikh and Shaurya Shah', style='Sub.TLabel').pack(side='left', padx=(8, 0))
        ttk.Label(banner, text='  |  Head-tracked Turn → Align → Forward').pack(side='left', padx=(4, 0))
        ttk.Button(banner, text='Export All Analysis Figures', command=self.export_all_analysis_figures).pack(side='right', padx=4)
        ttk.Label(banner, textvariable=self.status_var).pack(side='right', padx=8)

        # All main functions are horizontal tabs; no nested analysis notebook.
        nb = ttk.Notebook(self)
        nb.pack(fill='both', expand=True, padx=7, pady=(0, 6))
        self.main_nb = nb

        tabs = {}
        for key, label in (
            ('control', 'ML Control'),
            ('sim', '3D Simulation'),
            ('geometry', 'Geometry'),
            ('workspace', 'Workspace'),
            ('trajectory', 'Trajectory'),
            ('planning', 'Planning'),
            ('singularity', 'Singularity'),
            ('gait', 'Gait'),
            ('ml', 'ML Algorithms'),
            ('error', 'Error Analysis'),
            ('joints', '12-DOF'),
        ):
            tabs[key] = ttk.Frame(nb)
            nb.add(tabs[key], text=label)

        self._build_control_tab(tabs['control'])
        self._build_simulation_tab(tabs['sim'])
        self._build_geometry_tab(tabs['geometry'])
        self._build_workspace_tab(tabs['workspace'])
        self._build_trajectory_tab(tabs['trajectory'])
        self._build_planning_tab(tabs['planning'])
        self._build_singularity_tab(tabs['singularity'])
        self._build_gait_tab(tabs['gait'])
        self._build_ml_algorithms_tab(tabs['ml'])
        self._build_error_tab(tabs['error'])
        self._build_joint_tab(tabs['joints'])

    def _build_control_tab(self, parent):
        # Scrollable options solve small-window / laptop navigation problems.
        scroll = ScrollableFrame(parent, width=620)
        scroll.pack(fill='both', expand=True)
        root = scroll.inner

        title = ttk.LabelFrame(root, text='Workflow', padding=8)
        title.pack(fill='x', padx=8, pady=(8, 4))
        ttk.Label(
            title,
            text='1) Define waypoints  →  2) Train ML  →  3) Plan  →  4) Simulate  →  5) Export analysis graphs',
            style='Sub.TLabel'
        ).pack(anchor='w')

        # Two horizontal columns inside the vertical scroll region.
        cols = ttk.Frame(root)
        cols.pack(fill='both', expand=True, padx=8, pady=4)
        left = ttk.Frame(cols)
        right = ttk.Frame(cols)
        left.grid(row=0, column=0, sticky='nsew', padx=(0, 6))
        right.grid(row=0, column=1, sticky='nsew', padx=(6, 0))
        cols.columnconfigure(0, weight=1)
        cols.columnconfigure(1, weight=1)

        path_box = ttk.LabelFrame(left, text='1. Define Head/Nose Path', padding=8)
        path_box.pack(fill='x', pady=3)
        self.wp_list = tk.Listbox(path_box, height=8, exportselection=False)
        self.wp_list.pack(fill='x')
        row = ttk.Frame(path_box); row.pack(fill='x', pady=(4, 0))
        ttk.Label(row, text='X').pack(side='left')
        ttk.Entry(row, textvariable=self.wp_x_var, width=9).pack(side='left', padx=2)
        ttk.Label(row, text='Y').pack(side='left')
        ttk.Entry(row, textvariable=self.wp_y_var, width=9).pack(side='left', padx=2)
        ttk.Button(row, text='Add', command=self.add_waypoint).pack(side='left', padx=2)
        ttk.Button(row, text='Remove', command=self.remove_waypoint).pack(side='left', padx=2)
        row2 = ttk.Frame(path_box); row2.pack(fill='x', pady=(4, 0))
        ttk.Button(row2, text='Forward Example', command=self.load_forward_example).pack(side='left', padx=2)
        ttk.Button(row2, text='Forward/Backward Example', command=self.load_fb_example).pack(side='left', padx=2)
        ttk.Button(row2, text='Clear', command=self.clear_waypoints).pack(side='left', padx=2)

        start_box = ttk.LabelFrame(left, text='Robot Start', padding=7)
        start_box.pack(fill='x', pady=3)
        for r, (lab, var) in enumerate((
            ('Body X (m)', self.start_x_var),
            ('Body Y (m)', self.start_y_var),
            ('Yaw (deg)', self.start_yaw_var),
        )):
            ttk.Label(start_box, text=lab).grid(row=r, column=0, sticky='w', pady=2)
            ttk.Entry(start_box, textvariable=var, width=10).grid(row=r, column=1, padx=4)

        train_box = ttk.LabelFrame(left, text='2. Machine-Learning Training', padding=8)
        train_box.pack(fill='x', pady=3)
        for r, (lab, var) in enumerate((
            ('Training samples', self.dataset_var),
            ('Test fraction', self.test_var),
            ('Random seed', self.seed_var),
            ('Max travel/cycle (m)', self.max_step_var),
            ('Max yaw/cycle (deg)', self.max_yaw_var),
        )):
            ttk.Label(train_box, text=lab).grid(row=r, column=0, sticky='w', pady=2)
            ttk.Entry(train_box, textvariable=var, width=10).grid(row=r, column=1, padx=4)
        ttk.Button(train_box, text='TRAIN 4 MODELS', command=self.train_models).grid(row=5, column=0, columnspan=2, sticky='ew', pady=(6, 2))
        ttk.Button(train_box, text='Save Best Model', command=self.save_model).grid(row=6, column=0, sticky='ew', pady=2)
        ttk.Button(train_box, text='Load Model', command=self.load_model).grid(row=6, column=1, sticky='ew', pady=2)
        train_box.columnconfigure(0, weight=1); train_box.columnconfigure(1, weight=1)

        sim_box = ttk.LabelFrame(right, text='3. Path Planning / Simulation', padding=8)
        sim_box.pack(fill='x', pady=3)
        ttk.Label(sim_box, text='Path policy').grid(row=0, column=0, sticky='w', pady=2)
        ttk.Combobox(
            sim_box, textvariable=self.path_policy_var, state='readonly',
            values=('Turn First + Forward', 'Allow Automatic Reverse'), width=22
        ).grid(row=0, column=1, padx=4, pady=2, sticky='ew')
        for r, (lab, var) in enumerate((
            ('Waypoint tolerance (m)', self.tolerance_var),
            ('Turn-align threshold (deg)', self.turn_threshold_var),
            ('Max ML commands', self.max_commands_var),
            ('Gait cycle time (s)', self.cycle_time_var),
            ('Simulation speed ×', self.playback_speed_var),
            ('Duty factor', self.duty_var),
            ('World plane size (m)', self.plane_size_var),
        ), start=1):
            ttk.Label(sim_box, text=lab).grid(row=r, column=0, sticky='w', pady=2)
            ttk.Entry(sim_box, textvariable=var, width=10).grid(row=r, column=1, padx=4)
        ttk.Button(sim_box, text='PLAN ML PATH', command=self.plan_path).grid(row=8, column=0, columnspan=2, sticky='ew', pady=(7, 2))
        ttk.Button(sim_box, text='▶ PLAN + SIMULATE', command=lambda: self.plan_path(True)).grid(row=9, column=0, columnspan=2, sticky='ew', pady=2)
        ttk.Button(sim_box, text='⏸ Pause / Resume', command=self.pause_resume).grid(row=10, column=0, sticky='ew', pady=2)
        ttk.Button(sim_box, text='■ Stop', command=self.stop_sim).grid(row=10, column=1, sticky='ew', pady=2)
        ttk.Label(
            sim_box,
            text='Recommended: Turn First + Forward. Waypoint arrival is measured at the HEAD/Nose, not the body centre.',
            wraplength=360, justify='left'
        ).grid(row=11, column=0, columnspan=2, sticky='w', pady=(5, 0))
        sim_box.columnconfigure(0, weight=1); sim_box.columnconfigure(1, weight=1)

        save_box = ttk.LabelFrame(right, text='4. Data / Animation', padding=8)
        save_box.pack(fill='x', pady=3)
        ttk.Button(save_box, text='Save Planned HEAD Path CSV', command=self.save_planned_path).pack(fill='x', pady=2)
        ttk.Button(save_box, text='Save Full Simulation CSV', command=self.save_sim_log).pack(fill='x', pady=2)
        ttk.Button(save_box, text='Save Animation GIF', command=self.save_animation_gif).pack(fill='x', pady=2)
        ttk.Button(save_box, text='Export All Analysis Figures (PNG + PDF)', command=self.export_all_analysis_figures).pack(fill='x', pady=(8, 2))

        state_box = ttk.LabelFrame(right, text='Current State', padding=8)
        state_box.pack(fill='x', pady=3)
        ttk.Label(state_box, textvariable=self.best_var, wraplength=500, justify='left').pack(anchor='w')
        ttk.Label(state_box, textvariable=self.path_var, wraplength=500, justify='left').pack(anchor='w', pady=(4, 0))
        ttk.Label(state_box, textvariable=self.live_singularity_var, style='Sub.TLabel').pack(anchor='w', pady=(4, 0))

    def _build_simulation_tab(self, parent):
        live = ttk.LabelFrame(parent, text='Live ML Path-Following Simulation', padding=5)
        live.pack(fill='x', padx=6, pady=(6, 2))
        ttk.Label(live, textvariable=self.live_dir_var, style='Sub.TLabel').grid(row=0, column=0, sticky='w', padx=5)
        ttk.Label(live, textvariable=self.live_pose_var).grid(row=0, column=1, sticky='w', padx=5)
        ttk.Label(live, textvariable=self.live_target_var).grid(row=1, column=0, sticky='w', padx=5)
        ttk.Label(live, textvariable=self.live_error_var).grid(row=1, column=1, sticky='w', padx=5)
        ttk.Label(live, textvariable=self.live_command_var).grid(row=2, column=0, columnspan=2, sticky='w', padx=5)
        ttk.Label(live, textvariable=self.live_singularity_var).grid(row=3, column=0, columnspan=2, sticky='w', padx=5)
        live.columnconfigure(0, weight=1); live.columnconfigure(1, weight=1)

        controls = ttk.Frame(parent)
        controls.pack(fill='x', padx=6, pady=2)
        ttk.Button(controls, text='▶ Plan + Simulate', command=lambda: self.plan_path(True)).pack(side='left', padx=2)
        ttk.Button(controls, text='⏸ Pause / Resume', command=self.pause_resume).pack(side='left', padx=2)
        ttk.Button(controls, text='■ Stop', command=self.stop_sim).pack(side='left', padx=2)
        ttk.Button(controls, text='Save Animation GIF', command=self.save_animation_gif).pack(side='left', padx=8)

        # Playback-speed control affects only wall-clock playback. The plotted
        # time, angular velocity and angular acceleration remain referenced to
        # the simulated gait time.
        speed_box = ttk.LabelFrame(controls, text='Simulation Speed', padding=(6, 2))
        speed_box.pack(side='right', padx=4)
        ttk.Label(speed_box, text='×').pack(side='left')
        ttk.Scale(
            speed_box,
            from_=0.25,
            to=8.0,
            variable=self.playback_speed_var,
            orient='horizontal',
            length=180
        ).pack(side='left', padx=4)
        ttk.Spinbox(
            speed_box,
            from_=0.25,
            to=8.0,
            increment=0.25,
            textvariable=self.playback_speed_var,
            width=6
        ).pack(side='left', padx=3)
        for label, value in (('1×',1.0),('2×',2.0),('3×',3.0),('5×',5.0),('8×',8.0)):
            ttk.Button(
                speed_box, text=label, width=4,
                command=lambda v=value: self.playback_speed_var.set(v)
            ).pack(side='left', padx=1)

        self.fig3d = Figure(figsize=(10, 7), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection='3d')
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=parent)
        toolbar = NavigationToolbar2Tk(self.canvas3d, parent, pack_toolbar=False)
        toolbar.update(); toolbar.pack(fill='x')
        self.canvas3d.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=(0, 4))

    def _build_geometry_tab(self, parent):
        scroll = ScrollableFrame(parent, width=650)
        scroll.pack(fill='both', expand=True)
        root = scroll.inner
        ttk.Label(root, text='Robot Geometry — Independent Leg Link Lengths', style='Header.TLabel').pack(anchor='w', padx=10, pady=(10, 2))
        ttk.Label(
            root,
            text='Each of the eight leg links can be entered independently. This lets the simulator represent symmetric or asymmetric/custom robot-dog designs.',
            wraplength=1100, justify='left'
        ).pack(anchor='w', padx=10, pady=(0, 8))

        body = ttk.LabelFrame(root, text='Body Dimensions (m)', padding=10)
        body.pack(fill='x', padx=10, pady=4)
        for r, (lab, var) in enumerate((
            ('Body length', self.body_length_var),
            ('Body width', self.body_width_var),
            ('Body height', self.body_height_var),
        )):
            ttk.Label(body, text=lab).grid(row=r, column=0, sticky='w', padx=4, pady=4)
            ttk.Entry(body, textvariable=var, width=12).grid(row=r, column=1, sticky='w', padx=4)

        legs = ttk.LabelFrame(root, text='Per-Leg Link Lengths (m)', padding=10)
        legs.pack(fill='x', padx=10, pady=4)
        ttk.Label(legs, text='Leg', style='Sub.TLabel').grid(row=0, column=0, padx=5, pady=4)
        ttk.Label(legs, text='Upper link L1', style='Sub.TLabel').grid(row=0, column=1, padx=5, pady=4)
        ttk.Label(legs, text='Lower link L2', style='Sub.TLabel').grid(row=0, column=2, padx=5, pady=4)
        ttk.Label(legs, text='Total', style='Sub.TLabel').grid(row=0, column=3, padx=5, pady=4)
        self.geometry_total_labels = {}
        for r, leg in enumerate(LEG_ORDER, start=1):
            ttk.Label(legs, text=LEG_NAMES[leg]).grid(row=r, column=0, sticky='w', padx=5, pady=4)
            ttk.Entry(legs, textvariable=self.leg_upper_vars[leg], width=12).grid(row=r, column=1, padx=5)
            ttk.Entry(legs, textvariable=self.leg_lower_vars[leg], width=12).grid(row=r, column=2, padx=5)
            lbl = ttk.Label(legs, text='—')
            lbl.grid(row=r, column=3, padx=5)
            self.geometry_total_labels[leg] = lbl

        buttons = ttk.Frame(root); buttons.pack(fill='x', padx=10, pady=8)
        ttk.Button(buttons, text='Apply Geometry', command=self.apply_geometry_from_gui).pack(side='left', padx=3)
        ttk.Button(buttons, text='Reset Compact Dog', command=self.reset_compact_geometry).pack(side='left', padx=3)
        ttk.Button(buttons, text='Regenerate Workspace', command=self.generate_workspace_graph).pack(side='left', padx=12)

        limits = ttk.LabelFrame(root, text='Joint Limits Used for Workspace / Singularity Analysis', padding=10)
        limits.pack(fill='x', padx=10, pady=(4, 12))
        for r, joint in enumerate(JOINTS):
            lo, hi = JOINT_LIMITS[joint]
            ttk.Label(limits, text=f'{joint}: {lo:.1f}° to {hi:.1f}°').grid(row=r, column=0, sticky='w', padx=4, pady=3)
        self._refresh_geometry_totals()

    def _build_workspace_tab(self, parent):
        controls = ttk.Frame(parent, padding=6)
        controls.pack(fill='x')
        ttk.Label(controls, text='Leg').pack(side='left')
        ttk.Combobox(controls, textvariable=self.workspace_leg_var, state='readonly', values=('ALL','FL','FR','RL','RR'), width=8).pack(side='left', padx=3)
        ttk.Label(controls, text='Mode').pack(side='left', padx=(10,0))
        ttk.Combobox(
            controls, textvariable=self.workspace_mode_var, state='readonly', width=20,
            values=('Full leg workspace','q1 sweep','q2 sweep','q3 sweep')
        ).pack(side='left', padx=3)
        ttk.Label(controls, text='Samples/axis').pack(side='left', padx=(10,0))
        ttk.Spinbox(controls, from_=7, to=21, increment=2, textvariable=self.workspace_samples_var, width=6).pack(side='left', padx=3)
        ttk.Button(controls, text='Generate', command=self.generate_workspace_graph).pack(side='left', padx=8)
        ttk.Button(controls, text='Save Figure', command=lambda: self.save_figure_dialog(self.workspace_fig, 'Fig_Workspace')).pack(side='left', padx=3)
        ttk.Button(controls, text='Save Workspace CSV', command=self.save_workspace_csv).pack(side='left', padx=3)

        self.workspace_fig = Figure(figsize=(11, 7), dpi=100)
        self.ws_ax3d = self.workspace_fig.add_subplot(221, projection='3d')
        self.ws_axxy = self.workspace_fig.add_subplot(222)
        self.ws_axxz = self.workspace_fig.add_subplot(223)
        self.ws_axyz = self.workspace_fig.add_subplot(224)
        self.workspace_canvas = FigureCanvasTkAgg(self.workspace_fig, master=parent)
        self.workspace_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.generate_workspace_graph()

    def _build_trajectory_tab(self, parent):
        controls = ttk.Frame(parent, padding=6); controls.pack(fill='x')
        ttk.Label(controls, text='analysis trajectory plots: desired, planned and executed HEAD path').pack(side='left')
        ttk.Button(controls, text='Refresh', command=self._update_path_plot).pack(side='right', padx=3)
        ttk.Button(controls, text='Save Figure', command=lambda: self.save_figure_dialog(self.path_fig, 'Fig_Trajectory')).pack(side='right', padx=3)
        self.path_fig = Figure(figsize=(11, 7), dpi=100)
        self.path_axes = [self.path_fig.add_subplot(221), self.path_fig.add_subplot(222), self.path_fig.add_subplot(223), self.path_fig.add_subplot(224)]
        self.path_ax = self.path_axes[0]  # compatibility
        self.path_canvas = FigureCanvasTkAgg(self.path_fig, master=parent)
        self.path_canvas.get_tk_widget().pack(fill='both', expand=True)

    def _build_planning_tab(self, parent):
        controls = ttk.Frame(parent, padding=6); controls.pack(fill='x')
        ttk.Label(controls, text='Trajectory Planning Options: Linear vs Cubic vs Quintic').pack(side='left')
        ttk.Button(controls, text='Refresh', command=self.update_planning_plot).pack(side='right', padx=3)
        ttk.Button(controls, text='Save Figure', command=lambda: self.save_figure_dialog(self.planning_fig, 'Fig_Trajectory_Planning')).pack(side='right', padx=3)
        self.planning_fig = Figure(figsize=(11, 7), dpi=100)
        self.plan_axes = [self.planning_fig.add_subplot(221), self.planning_fig.add_subplot(222), self.planning_fig.add_subplot(223), self.planning_fig.add_subplot(224)]
        self.planning_canvas = FigureCanvasTkAgg(self.planning_fig, master=parent)
        self.planning_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_planning_plot()

    def _build_singularity_tab(self, parent):
        controls = ttk.Frame(parent, padding=6); controls.pack(fill='x')
        ttk.Label(controls, textvariable=self.live_singularity_var, style='Sub.TLabel').pack(side='left')
        ttk.Button(controls, text='Refresh', command=self.update_singularity_plot).pack(side='right', padx=3)
        ttk.Button(controls, text='Save Figure', command=lambda: self.save_figure_dialog(self.singularity_fig, 'Fig_Live_Singularity')).pack(side='right', padx=3)
        self.singularity_fig = Figure(figsize=(11, 7), dpi=100)
        self.sing_axes = [self.singularity_fig.add_subplot(221), self.singularity_fig.add_subplot(222), self.singularity_fig.add_subplot(223), self.singularity_fig.add_subplot(224)]
        self.singularity_canvas = FigureCanvasTkAgg(self.singularity_fig, master=parent)
        self.singularity_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_singularity_plot()

    def _build_gait_tab(self, parent):
        controls = ttk.Frame(parent, padding=6); controls.pack(fill='x')
        ttk.Label(controls, text='Gait Direction').pack(side='left')
        ttk.Combobox(
            controls, textvariable=self.gait_plot_direction_var, state='readonly', width=12,
            values=('Forward','Backward','Turn Left','Turn Right')
        ).pack(side='left', padx=4)
        ttk.Button(controls, text='Refresh', command=self.update_gait_plot).pack(side='left', padx=5)
        ttk.Button(controls, text='Save Figure', command=lambda: self.save_figure_dialog(self.gait_fig, 'Fig_Gait_Analysis')).pack(side='right', padx=3)
        self.gait_fig = Figure(figsize=(11, 7), dpi=100)
        self.gait_axes = [self.gait_fig.add_subplot(221), self.gait_fig.add_subplot(222), self.gait_fig.add_subplot(223), self.gait_fig.add_subplot(224)]
        self.gait_canvas = FigureCanvasTkAgg(self.gait_fig, master=parent)
        self.gait_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_gait_plot()

    def _build_ml_algorithms_tab(self, parent):
        top = ttk.Frame(parent, padding=6); top.pack(fill='x')
        ttk.Label(top, textvariable=self.best_var, style='Header.TLabel').pack(side='left')
        ttk.Button(top, text='Save Figure', command=lambda: self.save_figure_dialog(self.score_fig, 'Fig_ML_Algorithms')).pack(side='right', padx=3)
        cols = ('model','travel','yaw','q2','q3','r2')
        self.result_tree = ttk.Treeview(parent, columns=cols, show='headings', height=5)
        heads = {'model':'Algorithm','travel':'Travel RMSE (m)','yaw':'Yaw RMSE (deg)','q2':'q2 RMSE (deg)','q3':'q3 RMSE (deg)','r2':'Mean R²'}
        widths = {'model':170,'travel':125,'yaw':130,'q2':125,'q3':125,'r2':100}
        for c in cols:
            self.result_tree.heading(c, text=heads[c]); self.result_tree.column(c, width=widths[c], anchor='center')
        self.result_tree.pack(fill='x', padx=6)
        self.score_fig = Figure(figsize=(11, 7), dpi=100)
        self.score_axes = [self.score_fig.add_subplot(221), self.score_fig.add_subplot(222), self.score_fig.add_subplot(223), self.score_fig.add_subplot(224)]
        self.score_ax = self.score_axes[0]  # compatibility
        self.score_canvas = FigureCanvasTkAgg(self.score_fig, master=parent)
        self.score_canvas.get_tk_widget().pack(fill='both', expand=True, padx=4, pady=4)
        self._update_score_plot()

    def _build_error_tab(self, parent):
        controls = ttk.Frame(parent, padding=6); controls.pack(fill='x')
        ttk.Label(controls, text='HEAD waypoint tracking error and statistical error metrics').pack(side='left')
        ttk.Button(controls, text='Refresh', command=self.update_error_plot).pack(side='right', padx=3)
        ttk.Button(controls, text='Save Figure', command=lambda: self.save_figure_dialog(self.error_fig, 'Fig_Error_Analysis')).pack(side='right', padx=3)
        self.error_fig = Figure(figsize=(11, 7), dpi=100)
        self.error_axes = [self.error_fig.add_subplot(221), self.error_fig.add_subplot(222), self.error_fig.add_subplot(223), self.error_fig.add_subplot(224)]
        self.error_canvas = FigureCanvasTkAgg(self.error_fig, master=parent)
        self.error_canvas.get_tk_widget().pack(fill='both', expand=True)
        self.update_error_plot()

    def _build_main_tab(self, parent):
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True)
        left = ttk.Frame(paned, width=440)
        right = ttk.Frame(paned)
        paned.add(left, weight=0)
        paned.add(right, weight=1)

        # 1 Path
        path_box = ttk.LabelFrame(left, text="1. Define Path", padding=8)
        path_box.pack(fill="x", padx=6, pady=(6, 3))
        self.wp_list = tk.Listbox(path_box, height=7, exportselection=False)
        self.wp_list.pack(fill="x")
        row = ttk.Frame(path_box)
        row.pack(fill="x", pady=(4, 0))
        ttk.Label(row, text="X").pack(side="left")
        ttk.Entry(row, textvariable=self.wp_x_var, width=8).pack(side="left", padx=2)
        ttk.Label(row, text="Y").pack(side="left")
        ttk.Entry(row, textvariable=self.wp_y_var, width=8).pack(side="left", padx=2)
        ttk.Button(row, text="Add", command=self.add_waypoint).pack(side="left", padx=2)
        ttk.Button(row, text="Remove", command=self.remove_waypoint).pack(side="left", padx=2)
        row2 = ttk.Frame(path_box)
        row2.pack(fill="x", pady=(4, 0))
        ttk.Button(row2, text="Forward Example", command=self.load_forward_example).pack(side="left", padx=2)
        ttk.Button(row2, text="Forward + Backward Example", command=self.load_fb_example).pack(side="left", padx=2)
        ttk.Button(row2, text="Clear", command=self.clear_waypoints).pack(side="left", padx=2)

        start_box = ttk.LabelFrame(left, text="Robot Start", padding=7)
        start_box.pack(fill="x", padx=6, pady=3)
        for r, (lab, var) in enumerate((
            ("X (m)", self.start_x_var),
            ("Y (m)", self.start_y_var),
            ("Yaw (deg)", self.start_yaw_var),
        )):
            ttk.Label(start_box, text=lab).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(start_box, textvariable=var, width=9).grid(row=r, column=1, padx=4)

        # 2 Train
        train_box = ttk.LabelFrame(left, text="2. Train Machine Learning", padding=8)
        train_box.pack(fill="x", padx=6, pady=3)
        for r, (lab, var) in enumerate((
            ("Training samples", self.dataset_var),
            ("Test fraction", self.test_var),
            ("Random seed", self.seed_var),
            ("Max travel/cycle (m)", self.max_step_var),
            ("Max yaw/cycle (deg)", self.max_yaw_var),
        )):
            ttk.Label(train_box, text=lab).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(train_box, textvariable=var, width=9).grid(row=r, column=1, padx=4)
        ttk.Button(
            train_box, text="TRAIN 4 MODELS",
            command=self.train_models
        ).grid(row=5, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(train_box, text="Save Best Model", command=self.save_model).grid(row=6, column=0, sticky="ew", pady=2)
        ttk.Button(train_box, text="Load Model", command=self.load_model).grid(row=6, column=1, sticky="ew", pady=2)
        train_box.columnconfigure(0, weight=1)
        train_box.columnconfigure(1, weight=1)

        # 3 Plan/simulate
        sim_box = ttk.LabelFrame(left, text="3. ML Path Following", padding=8)
        sim_box.pack(fill="x", padx=6, pady=3)
        ttk.Label(sim_box, text="Path policy").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(
            sim_box,
            textvariable=self.path_policy_var,
            state="readonly",
            values=("Turn First + Forward", "Allow Automatic Reverse"),
            width=22
        ).grid(row=0, column=1, padx=4, pady=2, sticky="ew")

        for r, (lab, var) in enumerate((
            ("Waypoint tolerance (m)", self.tolerance_var),
            ("Turn-align threshold (deg)", self.turn_threshold_var),
            ("Max ML commands", self.max_commands_var),
            ("Gait cycle time (s)", self.cycle_time_var),
            ("Simulation speed ×", self.playback_speed_var),
            ("Duty factor", self.duty_var),
            ("World plane size (m)", self.plane_size_var),
        ), start=1):
            ttk.Label(sim_box, text=lab).grid(row=r, column=0, sticky="w", pady=2)
            ttk.Entry(sim_box, textvariable=var, width=9).grid(row=r, column=1, padx=4)

        ttk.Button(sim_box, text="PLAN ML PATH", command=self.plan_path).grid(row=8, column=0, columnspan=2, sticky="ew", pady=(6, 2))
        ttk.Button(sim_box, text="▶ PLAN + SIMULATE", command=lambda: self.plan_path(True)).grid(row=9, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Button(sim_box, text="⏸ Pause / Resume", command=self.pause_resume).grid(row=10, column=0, sticky="ew", pady=2)
        ttk.Button(sim_box, text="■ Stop", command=self.stop_sim).grid(row=10, column=1, sticky="ew", pady=2)

        ttk.Label(
            sim_box,
            text=(
                "Recommended policy: Turn First + Forward. The head/nose is the tracked "
                "waypoint reference; automatic reverse is disabled."
            ),
            wraplength=300,
            justify="left"
        ).grid(row=11, column=0, columnspan=2, sticky="w", pady=(5, 0))
        sim_box.columnconfigure(0, weight=1)
        sim_box.columnconfigure(1, weight=1)

        save_box = ttk.LabelFrame(left, text="Save", padding=7)
        save_box.pack(fill="x", padx=6, pady=3)
        ttk.Button(save_box, text="Save Planned Path CSV", command=self.save_planned_path).pack(fill="x", pady=2)
        ttk.Button(save_box, text="Save Full Simulation CSV", command=self.save_sim_log).pack(fill="x", pady=2)
        ttk.Button(save_box, text="Save Animation GIF", command=self.save_animation_gif).pack(fill="x", pady=2)

        # Right 3D
        live = ttk.LabelFrame(right, text="Live ML Path-Following Simulation", padding=5)
        live.pack(fill="x", padx=4, pady=(6, 2))
        ttk.Label(live, textvariable=self.live_dir_var, style="Sub.TLabel").grid(row=0, column=0, sticky="w", padx=5)
        ttk.Label(live, textvariable=self.live_pose_var).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(live, textvariable=self.live_target_var).grid(row=1, column=0, sticky="w", padx=5)
        ttk.Label(live, textvariable=self.live_error_var).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(live, textvariable=self.live_command_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5)
        live.columnconfigure(0, weight=1)
        live.columnconfigure(1, weight=1)

        self.fig3d = Figure(figsize=(8, 7), dpi=100)
        self.ax3d = self.fig3d.add_subplot(111, projection="3d")
        self.canvas3d = FigureCanvasTkAgg(self.fig3d, master=right)
        toolbar = NavigationToolbar2Tk(self.canvas3d, right, pack_toolbar=False)
        toolbar.update()
        toolbar.pack(fill="x")
        self.canvas3d.get_tk_widget().pack(fill="both", expand=True)

    def _build_analytics_tab(self, parent):
        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, textvariable=self.best_var, style="Header.TLabel").pack(anchor="w")
        ttk.Label(outer, textvariable=self.path_var).pack(anchor="w", pady=(2, 6))

        cols = ("model", "travel", "yaw", "q2", "q3", "r2")
        self.result_tree = ttk.Treeview(outer, columns=cols, show="headings", height=5)
        heads = {
            "model": "Algorithm",
            "travel": "Travel RMSE (m)",
            "yaw": "Yaw RMSE (deg)",
            "q2": "q2 RMSE (deg)",
            "q3": "q3 RMSE (deg)",
            "r2": "Mean R²",
        }
        widths = {"model": 170, "travel": 125, "yaw": 130, "q2": 125, "q3": 125, "r2": 100}
        for c in cols:
            self.result_tree.heading(c, text=heads[c])
            self.result_tree.column(c, width=widths[c], anchor="center")
        self.result_tree.pack(fill="x")

        graphs = ttk.Panedwindow(outer, orient="horizontal")
        graphs.pack(fill="both", expand=True, pady=(8, 0))
        left = ttk.Frame(graphs)
        right = ttk.Frame(graphs)
        graphs.add(left, weight=1)
        graphs.add(right, weight=1)

        self.path_fig = Figure(figsize=(6, 5), dpi=100)
        self.path_ax = self.path_fig.add_subplot(111)
        self.path_canvas = FigureCanvasTkAgg(self.path_fig, master=left)
        self.path_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.score_fig = Figure(figsize=(6, 5), dpi=100)
        self.score_ax = self.score_fig.add_subplot(111)
        self.score_canvas = FigureCanvasTkAgg(self.score_fig, master=right)
        self.score_canvas.get_tk_widget().pack(fill="both", expand=True)

    def _build_joint_tab(self, parent):
        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")
        ttk.Label(
            top,
            text="12-DOF Joint Kinematics — Angle, Displacement, Velocity & Acceleration",
            style="Header.TLabel"
        ).pack(side="left")

        # Plot controls.
        ttk.Label(top, text="Leg").pack(side="left", padx=(18, 2))
        ttk.Combobox(
            top,
            textvariable=self.joint_plot_leg_var,
            state="readonly",
            values=LEG_ORDER,
            width=6
        ).pack(side="left", padx=2)

        ttk.Label(top, text="Joint").pack(side="left", padx=(8, 2))
        ttk.Combobox(
            top,
            textvariable=self.joint_plot_joint_var,
            state="readonly",
            values=JOINTS,
            width=6
        ).pack(side="left", padx=2)

        ttk.Checkbutton(
            top,
            text="Live graph",
            variable=self.joint_plot_live_var
        ).pack(side="left", padx=8)

        ttk.Button(
            top,
            text="Refresh",
            command=self.update_joint_kinematics_plot
        ).pack(side="left", padx=2)

        ttk.Button(
            top,
            text="Save Selected Joint Figure",
            command=lambda: self.save_figure_dialog(
                self.joint_kinematics_fig,
                f"Fig_Joint_{self.joint_plot_leg_var.get()}_{self.joint_plot_joint_var.get()}"
            )
        ).pack(side="right", padx=2)

        ttk.Button(
            top,
            text="Export All 12 Joint Figures",
            command=self.export_all_joint_kinematics_figures
        ).pack(side="right", padx=2)

        # Live table.
        cols = ("leg", "joint", "angle", "disp", "vel", "acc")
        self.joint_tree = ttk.Treeview(
            outer, columns=cols, show="headings", height=12
        )
        heads = {
            "leg": "Leg",
            "joint": "Joint",
            "angle": "Angle θ (deg)",
            "disp": "Displacement Δθ (deg)",
            "vel": "Angular velocity ω (deg/s)",
            "acc": "Angular acceleration α (deg/s²)",
        }
        widths = {
            "leg": 115,
            "joint": 75,
            "angle": 125,
            "disp": 155,
            "vel": 180,
            "acc": 190,
        }
        for c in cols:
            self.joint_tree.heading(c, text=heads[c])
            self.joint_tree.column(c, width=widths[c], anchor="center")
        self.joint_tree.pack(fill="x", pady=(6, 6))
        self._refresh_joint_table()

        # analysis-ready four-panel graph for one selected joint.
        self.joint_kinematics_fig = Figure(figsize=(11, 7), dpi=100)
        self.joint_kin_axes = [
            self.joint_kinematics_fig.add_subplot(221),
            self.joint_kinematics_fig.add_subplot(222),
            self.joint_kinematics_fig.add_subplot(223),
            self.joint_kinematics_fig.add_subplot(224),
        ]
        self.joint_kin_canvas = FigureCanvasTkAgg(
            self.joint_kinematics_fig, master=outer
        )
        self.joint_kin_canvas.get_tk_widget().pack(fill="both", expand=True)

        self.update_joint_kinematics_plot()


    # ------------------------------------------------------------------
    # Waypoints / training
    # ------------------------------------------------------------------
    def _refresh_waypoints(self):
        self.wp_list.delete(0, "end")
        for i, (x, y) in enumerate(self.waypoints):
            self.wp_list.insert("end", f"{chr(65+i)}   X={x:+.3f}   Y={y:+.3f}")

    def add_waypoint(self):
        try:
            self.waypoints.append((float(self.wp_x_var.get()), float(self.wp_y_var.get())))
            self._refresh_waypoints(); self._update_path_plot()
        except Exception:
            messagebox.showerror("Waypoint", "Enter numeric X and Y values.")

    def remove_waypoint(self):
        sel = self.wp_list.curselection()
        if sel:
            del self.waypoints[int(sel[0])]
            self._refresh_waypoints(); self._update_path_plot()

    def clear_waypoints(self):
        self.waypoints = []
        self._refresh_waypoints(); self._update_path_plot()

    def load_forward_example(self):
        self.waypoints = [(0, 0), (0.4, 0), (0.8, 0), (1.2, 0)]
        self._refresh_waypoints(); self._update_path_plot()

    def load_fb_example(self):
        self.waypoints = [(0, 0), (0.55, 0), (1.0, 0), (0.45, 0), (-0.15, 0)]
        self._refresh_waypoints(); self._update_path_plot()

    def train_models(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("ML", "Install scikit-learn and joblib first.")
            return
        try:
            self.status_var.set("Training ML models...")
            self.update_idletasks()
            X, y = generate_training_data(
                self.dataset_var.get(), self.max_step_var.get(),
                self.max_yaw_var.get(), self.seed_var.get()
            )
            (self.models, self.results, self.best_model_name, self.best_model,
             ntr, nte) = train_ml_models(
                X, y, self.test_var.get(), self.seed_var.get()
            )
            r = self.results[self.best_model_name]
            self.best_var.set(
                f"Best ML model: {self.best_model_name} | Mean R²={r['mean_r2']:.5f} | Normalized RMSE={r['nrmse']:.5f}"
            )
            self.status_var.set(f"Training complete: {ntr} train / {nte} test samples")
            self._refresh_result_table(); self._update_score_plot(); self.update_error_plot()
        except Exception as exc:
            messagebox.showerror("ML Training", str(exc))

    def _refresh_result_table(self):
        for item in self.result_tree.get_children():
            self.result_tree.delete(item)
        for name, r in sorted(self.results.items(), key=lambda kv: kv[1]["nrmse"]):
            self.result_tree.insert("", "end", values=(
                name,
                f"{r['travel_rmse']:.5f}",
                f"{r['yaw_rmse']:.4f}",
                f"{r['q2_rmse']:.4f}",
                f"{r['q3_rmse']:.4f}",
                f"{r['mean_r2']:.5f}",
            ))

    def _update_score_plot(self):
        axes = getattr(self, 'score_axes', [self.score_ax])
        for ax in axes:
            ax.clear()
        if not self.results:
            axes[0].text(0.5, 0.5, 'Train the four models first', ha='center', va='center', transform=axes[0].transAxes)
            axes[0].set_title('ML Model Test Comparison')
            if hasattr(self, 'score_canvas'):
                self.score_fig.tight_layout(); self.score_canvas.draw_idle()
            return

        names = list(self.results)
        x = np.arange(len(names))
        width = 0.18
        metrics = [('travel_rmse','Travel RMSE'),('yaw_rmse','Yaw RMSE'),('q2_rmse','q2 RMSE'),('q3_rmse','q3 RMSE')]
        for k, (key, label) in enumerate(metrics):
            vals = [self.results[n][key] for n in names]
            axes[0].bar(x + (k-1.5)*width, vals, width=width, label=label)
        axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=15)
        axes[0].set_title('Per-Output Test RMSE'); axes[0].set_ylabel('RMSE'); axes[0].legend(fontsize=8); axes[0].grid(True, axis='y', alpha=0.25)

        r2 = [self.results[n]['mean_r2'] for n in names]
        axes[1].bar(names, r2); axes[1].set_title('Mean R²'); axes[1].set_ylabel('R²'); axes[1].tick_params(axis='x', rotation=15); axes[1].grid(True, axis='y', alpha=0.25)

        nr = [self.results[n]['nrmse'] for n in names]
        axes[2].bar(names, nr); axes[2].set_title('Normalized RMSE (lower is better)'); axes[2].tick_params(axis='x', rotation=15); axes[2].grid(True, axis='y', alpha=0.25)

        order = sorted(names, key=lambda n: self.results[n]['nrmse'])
        rank = list(range(1, len(order)+1))
        axes[3].barh(order, rank); axes[3].invert_yaxis(); axes[3].set_title('Algorithm Rank'); axes[3].set_xlabel('Rank (1 = best)'); axes[3].grid(True, axis='x', alpha=0.25)
        self.score_fig.tight_layout(); self.score_canvas.draw_idle()

    def save_model(self):
        if self.best_model is None:
            messagebox.showinfo("Save Model", "Train a model first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".joblib", filetypes=[("Joblib", "*.joblib")])
        if path:
            joblib.dump({
                "format": "ROBOQUAD_ML_V1_8",
                "model_name": self.best_model_name,
                "model": self.best_model,
            }, path)
            self.status_var.set(f"Saved model: {Path(path).name}")

    def load_model(self):
        if not SKLEARN_AVAILABLE:
            messagebox.showerror("ML", "Install scikit-learn and joblib first.")
            return
        path = filedialog.askopenfilename(filetypes=[("Joblib", "*.joblib"), ("All files", "*.*")])
        if path:
            try:
                obj = joblib.load(path)
                if isinstance(obj, dict) and "model" in obj:
                    self.best_model = obj["model"]
                    self.best_model_name = obj.get("model_name", "Loaded model")
                else:
                    self.best_model = obj
                    self.best_model_name = "Loaded model"
                self.best_var.set(f"Best ML model: {self.best_model_name}")
                self.status_var.set(f"Loaded model: {Path(path).name}")
            except Exception as exc:
                messagebox.showerror("Load Model", str(exc))

    # ------------------------------------------------------------------
    # Path planning / simulation
    # ------------------------------------------------------------------
    def plan_path(self, simulate=False):
        if self.best_model is None:
            messagebox.showinfo("ML Path", "Train or load an ML model first.")
            return
        if not self.waypoints:
            messagebox.showerror("ML Path", "Add at least one waypoint.")
            return
        try:
            start = (
                float(self.start_x_var.get()),
                float(self.start_y_var.get()),
                math.radians(float(self.start_yaw_var.get())),
            )
            head_offset = self._head_offset()
            allow_reverse = (
                self.path_policy_var.get() == "Allow Automatic Reverse"
            )
            self.plan = plan_ml_waypoints(
                self.best_model, self.waypoints, start,
                self.max_step_var.get(), self.max_yaw_var.get(),
                self.tolerance_var.get(), self.max_commands_var.get(),
                head_offset=head_offset,
                turn_threshold_deg=self.turn_threshold_var.get(),
                allow_reverse=allow_reverse
            )
            cmds = self.plan["commands"]
            nf = sum(c["direction"] == "Forward" for c in cmds)
            nb = sum(c["direction"] == "Backward" for c in cmds)
            nt = sum(c["direction"].startswith("Turn") for c in cmds)
            self.path_var.set(
                f"Path status: {'ALL WAYPOINTS REACHED' if self.plan['reached'] else 'MAX COMMANDS REACHED'} | "
                f"commands={len(cmds)} | turn={nt} | forward={nf} | backward={nb} | "
                f"HEAD final error={self.plan['final_error_m']:.4f} m"
            )
            self.status_var.set("ML path planned successfully.")
            self._update_path_plot(); self._update_3d(); self.update_planning_plot()
            if simulate:
                self.start_sim()
        except Exception as exc:
            messagebox.showerror("ML Path Planning", str(exc))

    def gait_angles(self, phase, direction, q2_amp, q3_lift):
        offsets = {"FL": 0.0, "RR": 0.0, "FR": 0.5, "RL": 0.5}
        duty = min(max(float(self.duty_var.get()), 0.25), 0.90)
        q = {}
        for leg in LEG_ORDER:
            lp = (phase + offsets[leg]) % 1.0
            theta = 2*math.pi*lp
            if lp < duty:
                lift = 0.0
            else:
                u = (lp-duty) / max(1e-9, 1-duty)
                lift = max(0.0, math.sin(math.pi*u))
            if direction == "Forward":
                q2 = 25.0 + q2_amp*math.sin(theta)
            elif direction == "Backward":
                q2 = 25.0 - q2_amp*math.sin(theta)
            elif direction == "Turn Left":
                # Left legs reverse their sweep; right legs advance.
                side_sign = -1.0 if leg.endswith("L") else +1.0
                q2 = 25.0 + side_sign*q2_amp*math.sin(theta)
            elif direction == "Turn Right":
                # Mirror of Turn Left.
                side_sign = +1.0 if leg.endswith("L") else -1.0
                q2 = 25.0 + side_sign*q2_amp*math.sin(theta)
            else:
                q2 = 25.0

            q[leg] = {
                "q1": 0.0,
                "q2": q2,
                "q3": -55.0 - q3_lift*lift,
            }
        return q

    def start_sim(self):
        if not self.plan or not self.plan["commands"]:
            messagebox.showinfo("Simulation", "Plan a non-empty ML path first.")
            return
        self.stop_sim(silent=True)
        self.base_x = float(self.start_x_var.get())
        self.base_y = float(self.start_y_var.get())
        self.base_yaw = math.radians(float(self.start_yaw_var.get()))
        self.joint_angles = {leg: dict(STAND[leg]) for leg in LEG_ORDER}
        self.prev_joint_angles = {leg: dict(STAND[leg]) for leg in LEG_ORDER}
        self.sim_initial_joint_angles = {
            leg: dict(STAND[leg]) for leg in LEG_ORDER
        }
        self.prev_joint_vel = {leg: {j: 0.0 for j in JOINTS} for leg in LEG_ORDER}
        self.body_trail = [(self.base_x, self.base_y)]
        hx0, hy0 = self._head_xy()
        self.head_trail = [(hx0, hy0)]
        self.sim_log = []
        self.command_index = 0
        self.command_elapsed = 0.0
        self.sim_time = 0.0
        self.sim_running = True
        self.sim_paused = False
        self._prepare_command()
        self.status_var.set("ML path simulation running.")
        self._sim_frame()

    def _prepare_command(self):
        if self.command_index >= len(self.plan["commands"]):
            self._finish_sim(); return
        self.current_command = self.plan["commands"][self.command_index]
        self.command_elapsed = 0.0
        self.command_start_pose = (self.base_x, self.base_y, self.base_yaw)
        c = self.current_command
        self.live_dir_var.set(f"Direction: {c['direction']}")
        self.live_target_var.set(
            f"Target waypoint {c['target_index']+1}: X={c['target_x']:.3f}, Y={c['target_y']:.3f}"
        )
        self.live_command_var.set(
            f"Command: {c['direction']} | travel={c['travel_m']:.4f} m, "
            f"yaw={c['yaw_step_deg']:+.2f}°, A2={c['q2_amp_deg']:.2f}°, "
            f"A3={c['q3_lift_deg']:.2f}° | HEAD tracking active"
        )

    def _sim_frame(self):
        if not self.sim_running or self.sim_paused:
            return
        T = max(0.20, float(self.cycle_time_var.get()))
        fps = 30.0
        try:
            playback = min(max(float(self.playback_speed_var.get()), 0.25), 8.0)
        except Exception:
            playback = 3.0
        # Simulated time advanced per displayed frame.
        # Higher playback shortens wall-clock animation without changing
        # the underlying physical cycle duration used by velocity/acceleration.
        dt = playback / fps
        u = min(max(self.command_elapsed/T, 0.0), 1.0)
        s = quintic(u)
        c = self.current_command

        # Joint gait follows full phase; starts/ends at the neutral q2 phase.
        new_q = self.gait_angles(u % 1.0, c["direction"], c["q2_amp_deg"], c["q3_lift_deg"])

        # Numerical joint derivatives.
        for leg in LEG_ORDER:
            for j in JOINTS:
                angle = float(new_q[leg][j])
                vel = (angle - self.prev_joint_angles[leg][j]) / dt
                acc = (vel - self.prev_joint_vel[leg][j]) / dt
                self.joint_vel[leg][j] = vel
                self.joint_acc[leg][j] = acc
                self.prev_joint_angles[leg][j] = angle
                self.prev_joint_vel[leg][j] = vel
        self.joint_angles = new_q

        bx0, by0, byaw0 = self.command_start_pose
        yaw_delta = math.radians(c["yaw_step_deg"])

        if c["direction"] in ("Turn Left", "Turn Right"):
            # Rotate the BODY around the current HEAD pivot. The tracked head
            # therefore remains fixed at the previous destination during turn.
            self.base_yaw = wrap_angle(byaw0 + yaw_delta*s)
            pivot_x = float(c["pivot_x"])
            pivot_y = float(c["pivot_y"])
            h_off = self._head_offset()
            self.base_x = pivot_x - math.cos(self.base_yaw)*h_off
            self.base_y = pivot_y - math.sin(self.base_yaw)*h_off

        else:
            self.base_yaw = byaw0
            if c["direction"] == "Forward":
                signed_travel = +c["travel_m"]
            elif c["direction"] == "Backward":
                signed_travel = -c["travel_m"]
            else:
                signed_travel = 0.0

            self.base_x = bx0 + math.cos(byaw0)*signed_travel*s
            self.base_y = by0 + math.sin(byaw0)*signed_travel*s

        self.body_trail.append((self.base_x, self.base_y))
        hx, hy = self._head_xy()
        self.head_trail.append((hx, hy))

        err = math.hypot(c["target_x"]-hx, c["target_y"]-hy)
        self.live_pose_var.set(
            f"Body: X={self.base_x:.3f}, Y={self.base_y:.3f}, yaw={math.degrees(self.base_yaw):.1f}° | "
            f"HEAD: X={hx:.3f}, Y={hy:.3f}"
        )
        self.live_error_var.set(f"HEAD waypoint error: {err:.4f} m")

        self._log_frame(err)
        self._refresh_joint_table()
        self._update_3d()
        self.graph_frame_counter += 1
        # Keep simulation responsive: singularity remains live at ~2.5 Hz,
        # while heavier analysis/error plots refresh about once per second.
        if self.graph_frame_counter % 12 == 0:
            self.update_singularity_plot()
        if self.graph_frame_counter % 30 == 0:
            self.update_error_plot()
            self._update_path_plot()
            if (
                hasattr(self, "joint_plot_live_var")
                and self.joint_plot_live_var.get()
            ):
                self.update_joint_kinematics_plot()

        self.command_elapsed += dt
        self.sim_time += dt
        if self.command_elapsed >= T:
            # Force planner endpoint to avoid cumulative interpolation drift.
            self.base_x = c["end_x"]
            self.base_y = c["end_y"]
            self.base_yaw = c["end_yaw"]
            self.command_index += 1
            if self.command_index >= len(self.plan["commands"]):
                self._finish_sim(); return
            self._prepare_command()

        self.sim_job = self.after(int(1000/fps), self._sim_frame)

    def _log_frame(self, error):
        hx, hy = self._head_xy()
        T = max(0.20, float(self.cycle_time_var.get()))
        phase = (self.command_elapsed / T) % 1.0
        support_legs = self._support_legs(phase)
        if self._last_singularity_metrics is None or self.graph_frame_counter % 3 == 0:
            self._last_singularity_metrics = self.fk.all_singularity_metrics(self.joint_angles)
        metrics = self._last_singularity_metrics
        row = {
            'time_s': self.sim_time,
            'command_index': self.command_index,
            'direction': self.current_command['direction'],
            'target_index': self.current_command['target_index'],
            'body_x_m': self.base_x,
            'body_y_m': self.base_y,
            'body_yaw_deg': math.degrees(self.base_yaw),
            'head_x_m': hx,
            'head_y_m': hy,
            'head_waypoint_error_m': error,
            'waypoint_error_m': error,
            'travel_command_m': self.current_command['travel_m'],
            'yaw_command_deg': self.current_command['yaw_step_deg'],
            'q2_amplitude_deg': self.current_command['q2_amp_deg'],
            'q3_lift_deg': self.current_command['q3_lift_deg'],
            'gait_phase': phase,
            'support_count': len(support_legs),
        }
        worst_cond = 0.0
        min_sigma = float('inf')
        for leg in LEG_ORDER:
            m = metrics[leg]
            row[f'{leg}_sigma_min'] = m['sigma_min']
            row[f'{leg}_condition'] = m['condition']
            row[f'{leg}_detJ'] = m['det']
            row[f'{leg}_rankJ'] = m['rank']
            min_sigma = min(min_sigma, m['sigma_min'])
            if math.isfinite(m['condition']):
                worst_cond = max(worst_cond, m['condition'])
            else:
                worst_cond = float('inf')
        row['robot_sigma_min'] = min_sigma
        row['robot_condition_max'] = worst_cond

        for leg in LEG_ORDER:
            for j in JOINTS:
                angle = float(self.joint_angles[leg][j])
                displacement = (
                    angle
                    - float(self.sim_initial_joint_angles[leg][j])
                )
                row[f'{leg}_{j}_angle_deg'] = angle
                row[f'{leg}_{j}_disp_deg'] = displacement
                row[f'{leg}_{j}_vel_deg_s'] = self.joint_vel[leg][j]
                row[f'{leg}_{j}_acc_deg_s2'] = self.joint_acc[leg][j]
        self.sim_log.append(row)

        if min_sigma < 1e-5:
            state = 'SINGULAR'
        elif min_sigma < 0.004 or (math.isfinite(worst_cond) and worst_cond > 100.0):
            state = 'NEAR SINGULAR'
        else:
            state = 'OK'
        cond_text = '∞' if not math.isfinite(worst_cond) else f'{worst_cond:.1f}'
        self.live_singularity_var.set(f'Singularity: {state} | σmin={min_sigma:.5f} | max κ={cond_text}')

    def pause_resume(self):
        if not self.sim_running:
            return
        if self.sim_paused:
            self.sim_paused = False
            self.status_var.set("Simulation resumed.")
            self._sim_frame()
        else:
            self.sim_paused = True
            if self.sim_job:
                try: self.after_cancel(self.sim_job)
                except Exception: pass
                self.sim_job = None
            self.status_var.set("Simulation paused.")

    def stop_sim(self, silent=False):
        if self.sim_job:
            try: self.after_cancel(self.sim_job)
            except Exception: pass
        self.sim_job = None
        self.sim_running = False
        self.sim_paused = False
        if not silent:
            self.status_var.set("Simulation stopped.")

    def _finish_sim(self):
        self.sim_running = False
        self.sim_job = None
        self.live_dir_var.set('Direction: COMPLETE')
        self.status_var.set('ML path simulation complete. analysis graphs refreshed.')
        self._update_3d()
        self._refresh_all_analysis_graphs()
        self.update_joint_kinematics_plot()

    def _refresh_joint_table(self):
        if not hasattr(self, "joint_tree"):
            return
        for item in self.joint_tree.get_children():
            self.joint_tree.delete(item)

        for leg in LEG_ORDER:
            for j in JOINTS:
                displacement = (
                    float(self.joint_angles[leg][j])
                    - float(self.sim_initial_joint_angles[leg][j])
                )
                self.joint_tree.insert("", "end", values=(
                    LEG_NAMES[leg],
                    j,
                    f"{self.joint_angles[leg][j]:.3f}",
                    f"{displacement:.3f}",
                    f"{self.joint_vel[leg][j]:.3f}",
                    f"{self.joint_acc[leg][j]:.3f}",
                ))

    def update_joint_kinematics_plot(self, leg=None, joint=None):
        """Four analysis panels for one selected joint."""
        if not hasattr(self, "joint_kin_axes"):
            return

        leg = leg or self.joint_plot_leg_var.get()
        joint = joint or self.joint_plot_joint_var.get()
        axes = self.joint_kin_axes

        for ax in axes:
            ax.clear()

        if not self.sim_log:
            axes[0].text(
                0.5, 0.5,
                "Run a simulation to generate joint kinematics.",
                ha="center", va="center",
                transform=axes[0].transAxes
            )
            axes[0].set_title("(a) Angular Position")
            axes[1].set_title("(b) Angular Displacement")
            axes[2].set_title("(c) Angular Velocity")
            axes[3].set_title("(d) Angular Acceleration")
        else:
            t = np.asarray(
                [r["time_s"] for r in self.sim_log],
                dtype=float
            )
            angle = np.asarray(
                [r[f"{leg}_{joint}_angle_deg"] for r in self.sim_log],
                dtype=float
            )

            disp_key = f"{leg}_{joint}_disp_deg"
            if disp_key in self.sim_log[0]:
                disp = np.asarray(
                    [r[disp_key] for r in self.sim_log],
                    dtype=float
                )
            else:
                disp = angle - angle[0]

            vel = np.asarray(
                [r[f"{leg}_{joint}_vel_deg_s"] for r in self.sim_log],
                dtype=float
            )
            acc = np.asarray(
                [r[f"{leg}_{joint}_acc_deg_s2"] for r in self.sim_log],
                dtype=float
            )

            axes[0].plot(t, angle, linewidth=1.5)
            axes[0].set_title("(a) Angular Position")
            axes[0].set_ylabel("θ (deg)")

            axes[1].plot(t, disp, linewidth=1.5)
            axes[1].axhline(0.0, linewidth=0.8, alpha=0.5)
            axes[1].set_title("(b) Angular Displacement")
            axes[1].set_ylabel("Δθ (deg)")

            axes[2].plot(t, vel, linewidth=1.5)
            axes[2].axhline(0.0, linewidth=0.8, alpha=0.5)
            axes[2].set_title("(c) Angular Velocity")
            axes[2].set_ylabel("ω (deg/s)")

            axes[3].plot(t, acc, linewidth=1.5)
            axes[3].axhline(0.0, linewidth=0.8, alpha=0.5)
            axes[3].set_title("(d) Angular Acceleration")
            axes[3].set_ylabel("α (deg/s²)")

            for ax in axes:
                ax.set_xlabel("Simulation time (s)")
                ax.grid(True, alpha=0.25)

            peak_v = float(np.max(np.abs(vel))) if len(vel) else 0.0
            peak_a = float(np.max(np.abs(acc))) if len(acc) else 0.0
            total_excursion = (
                float(np.max(angle) - np.min(angle))
                if len(angle) else 0.0
            )
            self.joint_kinematics_fig.suptitle(
                f"{LEG_NAMES[leg]} {joint} Joint Kinematics | "
                f"Excursion={total_excursion:.2f}°, "
                f"Peak |ω|={peak_v:.2f}°/s, "
                f"Peak |α|={peak_a:.2f}°/s²",
                fontsize=10
            )

        self.joint_kinematics_fig.tight_layout()
        self.joint_kin_canvas.draw_idle()

    def _make_joint_kinematics_figure(self, leg, joint):
        """Create a standalone 4-panel figure for bulk analysis export."""
        fig = Figure(figsize=(11, 7), dpi=100)
        axes = [
            fig.add_subplot(221),
            fig.add_subplot(222),
            fig.add_subplot(223),
            fig.add_subplot(224),
        ]

        if not self.sim_log:
            axes[0].text(
                0.5, 0.5, "No simulation data.",
                ha="center", va="center",
                transform=axes[0].transAxes
            )
            return fig

        t = np.asarray([r["time_s"] for r in self.sim_log], dtype=float)
        angle = np.asarray(
            [r[f"{leg}_{joint}_angle_deg"] for r in self.sim_log],
            dtype=float
        )
        disp_key = f"{leg}_{joint}_disp_deg"
        disp = np.asarray(
            [
                r.get(disp_key, r[f"{leg}_{joint}_angle_deg"] - angle[0])
                for r in self.sim_log
            ],
            dtype=float
        )
        vel = np.asarray(
            [r[f"{leg}_{joint}_vel_deg_s"] for r in self.sim_log],
            dtype=float
        )
        acc = np.asarray(
            [r[f"{leg}_{joint}_acc_deg_s2"] for r in self.sim_log],
            dtype=float
        )

        series = (
            (angle, "(a) Angular Position", "θ (deg)"),
            (disp, "(b) Angular Displacement", "Δθ (deg)"),
            (vel, "(c) Angular Velocity", "ω (deg/s)"),
            (acc, "(d) Angular Acceleration", "α (deg/s²)"),
        )
        for ax, (y, title, ylabel) in zip(axes, series):
            ax.plot(t, y, linewidth=1.5)
            ax.set_title(title)
            ax.set_xlabel("Simulation time (s)")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.25)

        peak_v = float(np.max(np.abs(vel))) if len(vel) else 0.0
        peak_a = float(np.max(np.abs(acc))) if len(acc) else 0.0
        excursion = float(np.max(angle)-np.min(angle)) if len(angle) else 0.0
        fig.suptitle(
            f"{LEG_NAMES[leg]} {joint} Joint Kinematics | "
            f"Excursion={excursion:.2f}°, "
            f"Peak |ω|={peak_v:.2f}°/s, "
            f"Peak |α|={peak_a:.2f}°/s²",
            fontsize=10
        )
        fig.tight_layout()
        return fig

    def export_all_joint_kinematics_figures(self):
        if not self.sim_log:
            messagebox.showinfo(
                "Joint Kinematics",
                "Run a simulation first."
            )
            return

        folder = filedialog.askdirectory(
            title="Select Folder for 12-DOF Joint Kinematics Figures"
        )
        if not folder:
            return

        out = Path(folder)
        count = 0
        for leg in LEG_ORDER:
            for joint in JOINTS:
                fig = self._make_joint_kinematics_figure(leg, joint)
                name = f"Joint_{leg}_{joint}_Kinematics"
                fig.savefig(
                    out / f"{name}.png",
                    dpi=600,
                    bbox_inches="tight"
                )
                fig.savefig(
                    out / f"{name}.pdf",
                    bbox_inches="tight"
                )
                count += 1

        self.status_var.set(
            f"Exported {count} joint-kinematics figure sets."
        )
        messagebox.showinfo(
            "12-DOF Joint Figures",
            f"Exported {count} joint figures as 600-dpi PNG + PDF.\n\n{folder}"
        )


    def _head_offset(self):
        """Body-centre to tracked nose point distance."""
        return float(self.fk.g.body_length/2.0 + 0.120)

    def _head_xy(self):
        return head_xy_from_body(
            self.base_x, self.base_y,
            self.base_yaw, self._head_offset()
        )

    def _world_point(self, p):
        R = rot_z(self.base_yaw)[:3, :3]
        return R @ np.asarray(p) + np.array([self.base_x, self.base_y, self.base_z])

    def _update_3d(self):
        ax = self.ax3d
        try: elev, azim = ax.elev, ax.azim
        except Exception: elev, azim = 25, -55
        ax.clear(); ax.view_init(elev=elev, azim=azim)

        size = max(1.5, float(self.plane_size_var.get()))
        half = size/2
        route_points = [(self.base_x, self.base_y)] + list(self.waypoints)
        if self.plan:
            route_points += list(self.plan["path"])
        xs = [p[0] for p in route_points] or [0]
        ys = [p[1] for p in route_points] or [0]
        xmin, xmax = min(-half, min(xs)-0.5), max(half, max(xs)+0.5)
        ymin, ymax = min(-half, min(ys)-0.5), max(half, max(ys)+0.5)

        # Determine a consistent ground from current nominal foot height.
        body_pts = self.fk.all_points(self.joint_angles)
        if not self.sim_log:
            self.ground_z = min(v[2][2] for v in body_pts.values())
        z0 = self.ground_z

        # Plane grid.
        step = 0.5
        for xv in np.arange(math.floor(xmin/step)*step, math.ceil(xmax/step)*step+0.1, step):
            ax.plot([xv, xv], [ymin, ymax], [z0, z0], linewidth=0.4, alpha=0.20)
        for yv in np.arange(math.floor(ymin/step)*step, math.ceil(ymax/step)*step+0.1, step):
            ax.plot([xmin, xmax], [yv, yv], [z0, z0], linewidth=0.4, alpha=0.20)
        ax.plot([xmin, xmax], [0, 0], [z0, z0], linestyle=":", linewidth=1)
        ax.plot([0, 0], [ymin, ymax], [z0, z0], linestyle=":", linewidth=1)
        ax.text(xmax-0.1, 0, z0, " FORWARD +X", fontsize=8)
        ax.text(xmin+0.1, 0, z0, " BACKWARD -X", fontsize=8)

        # Desired path and ML plan.
        if self.waypoints:
            P = np.asarray(self.waypoints)
            ax.plot(
                P[:,0], P[:,1],
                np.full(len(P), z0+0.012),
                marker="o", linewidth=2,
                label="HEAD destinations"
            )
            for i, (wx, wy) in enumerate(self.waypoints):
                ax.text(
                    wx, wy, z0+0.035,
                    f" D{i+1}", fontsize=8
                )
        if self.plan and self.plan.get("head_path"):
            Q = np.asarray(self.plan["head_path"])
            ax.plot(
                Q[:,0], Q[:,1], np.full(len(Q), z0+0.018),
                linestyle="--", linewidth=1.6,
                label="ML planned HEAD path"
            )
        if len(self.body_trail) >= 2:
            B = np.asarray(self.body_trail)
            ax.plot(
                B[:,0], B[:,1], np.full(len(B), z0+0.020),
                linewidth=1.3, alpha=0.65,
                label="Executed body-centre path"
            )
        if len(self.head_trail) >= 2:
            HTR = np.asarray(self.head_trail)
            ax.plot(
                HTR[:,0], HTR[:,1], np.full(len(HTR), z0+0.030),
                linewidth=2.4,
                label="Executed HEAD path"
            )

        # Body box.
        g = self.fk.g; L, W, H = g.body_length, g.body_width, g.body_height
        top_b = np.array([
            [ L/2,  W/2, H/2], [ L/2, -W/2, H/2],
            [-L/2, -W/2, H/2], [-L/2,  W/2, H/2], [ L/2,  W/2, H/2]
        ])
        bot_b = top_b.copy(); bot_b[:,2] = -H/2
        top = np.vstack([self._world_point(p) for p in top_b])
        bot = np.vstack([self._world_point(p) for p in bot_b])
        ax.plot(top[:,0], top[:,1], top[:,2], linewidth=2.2)
        ax.plot(bot[:,0], bot[:,1], bot[:,2], linewidth=2.2)
        for i in range(4):
            ax.plot([top[i,0], bot[i,0]], [top[i,1], bot[i,1]], [top[i,2], bot[i,2]], linewidth=1)

        # Compact dog-like head block.
        head_top_b = np.array([
            [L/2 + 0.025,  W*0.30, H*0.42],
            [L/2 + 0.120,  W*0.30, H*0.42],
            [L/2 + 0.120, -W*0.30, H*0.42],
            [L/2 + 0.025, -W*0.30, H*0.42],
            [L/2 + 0.025,  W*0.30, H*0.42],
        ])
        head_bot_b = head_top_b.copy()
        head_bot_b[:, 2] = -H*0.05
        htop = np.vstack([self._world_point(p) for p in head_top_b])
        hbot = np.vstack([self._world_point(p) for p in head_bot_b])
        ax.plot(htop[:,0], htop[:,1], htop[:,2], linewidth=1.7)
        ax.plot(hbot[:,0], hbot[:,1], hbot[:,2], linewidth=1.7)
        for i in range(4):
            ax.plot(
                [htop[i,0], hbot[i,0]],
                [htop[i,1], hbot[i,1]],
                [htop[i,2], hbot[i,2]],
                linewidth=0.9
            )

        # Short tail at the rear.
        Rtail = rot_z(self.base_yaw)[:3, :3]
        rear = np.array([self.base_x, self.base_y, self.base_z]) + Rtail @ np.array([-L/2, 0.0, 0.0])
        tail = np.array([self.base_x, self.base_y, self.base_z]) + Rtail @ np.array([-L/2-0.12, 0.0, 0.07])
        ax.plot([rear[0], tail[0]], [rear[1], tail[1]], [rear[2], tail[2]], linewidth=2.0)

        # Forward arrow.
        R = rot_z(self.base_yaw)[:3, :3]
        fwd = R @ np.array([1., 0., 0.])
        c = np.array([self.base_x, self.base_y, self.base_z])
        p1, p2 = c + fwd*(L/2), c + fwd*(L/2+0.16)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], linewidth=4)
        ax.text(*p2, " FWD", fontsize=8)

        # Explicit HEAD/Nose tracking marker.
        hx, hy = self._head_xy()
        hz = self.base_z + H*0.08
        ax.scatter([hx], [hy], [hz], s=58, marker="*")
        ax.text(hx, hy, hz+0.025, " HEAD TRACK", fontsize=8)

        # Legs.
        for leg, (hip, knee, foot) in body_pts.items():
            pts = np.vstack([self._world_point(hip), self._world_point(knee), self._world_point(foot)])
            ax.plot(pts[:,0], pts[:,1], pts[:,2], marker="o", linewidth=3, markersize=4)
            ax.text(*pts[2], f" {leg}", fontsize=7)

        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.set_zlim(min(z0-0.06, -0.32), 0.28)
        ax.set_xlabel("World X / Forward (m)")
        ax.set_ylabel("World Y / Left (m)")
        ax.set_zlabel("Z (m)")
        hx_t, hy_t = self._head_xy()
        ax.set_title(
            f"ML Robot Dog | HEAD=({hx_t:.3f}, {hy_t:.3f}) m | "
            f"Body yaw={math.degrees(self.base_yaw):.1f}°"
        )
        ax.grid(True, alpha=0.2)
        if self.waypoints or self.plan or self.body_trail:
            ax.legend(loc="upper right", fontsize=8)
        self.fig3d.tight_layout(); self.canvas3d.draw_idle()

    def _update_path_plot(self):
        axes = getattr(self, 'path_axes', [self.path_ax])
        for ax in axes:
            ax.clear()
        ax0 = axes[0]
        if self.waypoints:
            P = np.asarray(self.waypoints)
            ax0.plot(P[:,0], P[:,1], marker='o', linewidth=2, label='Desired HEAD waypoints')
        if self.plan and self.plan.get('head_path'):
            Q = np.asarray(self.plan['head_path'])
            ax0.plot(Q[:,0], Q[:,1], linewidth=1.7, label='ML planned HEAD path')
        if len(self.head_trail) >= 2:
            HTR = np.asarray(self.head_trail)
            ax0.plot(HTR[:,0], HTR[:,1], linewidth=2.0, label='Executed HEAD path')
        if len(self.body_trail) >= 2:
            B = np.asarray(self.body_trail)
            ax0.plot(B[:,0], B[:,1], linewidth=1.0, alpha=0.55, label='Body-centre path')
        ax0.scatter([float(self.start_x_var.get())],[float(self.start_y_var.get())], marker='s', s=50, label='Start body centre')
        ax0.set_xlabel('World X (m)'); ax0.set_ylabel('World Y (m)'); ax0.set_title('(a) Desired / Planned / Executed Trajectory'); ax0.axis('equal'); ax0.grid(True, alpha=0.25); ax0.legend(fontsize=7)

        if self.sim_log:
            t = np.asarray([r['time_s'] for r in self.sim_log], dtype=float)
            hx = np.asarray([r['head_x_m'] for r in self.sim_log], dtype=float)
            hy = np.asarray([r['head_y_m'] for r in self.sim_log], dtype=float)
            yaw = np.asarray([r['body_yaw_deg'] for r in self.sim_log], dtype=float)
            err = np.asarray([r['head_waypoint_error_m'] for r in self.sim_log], dtype=float)
            axes[1].plot(t, hx, label='Head X'); axes[1].plot(t, hy, label='Head Y')
            axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Position (m)'); axes[1].set_title('(b) HEAD Cartesian Position'); axes[1].grid(True, alpha=0.25); axes[1].legend(fontsize=8)
            axes[2].plot(t, yaw); axes[2].set_xlabel('Time (s)'); axes[2].set_ylabel('Yaw (deg)'); axes[2].set_title('(c) Body Heading'); axes[2].grid(True, alpha=0.25)
            axes[3].plot(t, err); axes[3].set_xlabel('Time (s)'); axes[3].set_ylabel('HEAD Error (m)'); axes[3].set_title('(d) Waypoint Tracking Error'); axes[3].grid(True, alpha=0.25)
        else:
            for i, title in ((1,'(b) HEAD Cartesian Position'),(2,'(c) Body Heading'),(3,'(d) Waypoint Tracking Error')):
                axes[i].text(0.5,0.5,'Run a simulation to populate this graph',ha='center',va='center',transform=axes[i].transAxes)
                axes[i].set_title(title)
        self.path_fig.tight_layout(); self.path_canvas.draw_idle()

    # ------------------------------------------------------------------
    # analysis / analysis helpers
    # ------------------------------------------------------------------

    def _support_legs(self, phase):
        duty = min(max(float(self.duty_var.get()), 0.25), 0.90)
        offsets = {'FL':0.0, 'RR':0.0, 'FR':0.5, 'RL':0.5}
        return [leg for leg in LEG_ORDER if ((float(phase)+offsets[leg]) % 1.0) < duty]

    def _refresh_geometry_totals(self):
        if not hasattr(self, 'geometry_total_labels'):
            return
        for leg in LEG_ORDER:
            total = float(self.leg_upper_vars[leg].get()) + float(self.leg_lower_vars[leg].get())
            self.geometry_total_labels[leg].configure(text=f'{total:.4f} m')

    def apply_geometry_from_gui(self):
        try:
            g = Geometry(
                body_length=float(self.body_length_var.get()),
                body_width=float(self.body_width_var.get()),
                body_height=float(self.body_height_var.get()),
                FL_upper=float(self.leg_upper_vars['FL'].get()), FL_lower=float(self.leg_lower_vars['FL'].get()),
                FR_upper=float(self.leg_upper_vars['FR'].get()), FR_lower=float(self.leg_lower_vars['FR'].get()),
                RL_upper=float(self.leg_upper_vars['RL'].get()), RL_lower=float(self.leg_lower_vars['RL'].get()),
                RR_upper=float(self.leg_upper_vars['RR'].get()), RR_lower=float(self.leg_lower_vars['RR'].get()),
            )
            values = [g.body_length, g.body_width, g.body_height] + [g.upper(l) for l in LEG_ORDER] + [g.lower(l) for l in LEG_ORDER]
            if any(v <= 0.0 for v in values):
                raise ValueError('Every body/link dimension must be positive.')
            self.fk = QuadrupedFK(g)
            self._refresh_geometry_totals()
            self._update_3d(); self.generate_workspace_graph(); self.update_singularity_plot()
            self.status_var.set('Independent leg geometry applied.')
        except Exception as exc:
            messagebox.showerror('Geometry', str(exc))

    def reset_compact_geometry(self):
        defaults = Geometry()
        self.body_length_var.set(defaults.body_length); self.body_width_var.set(defaults.body_width); self.body_height_var.set(defaults.body_height)
        for leg in LEG_ORDER:
            self.leg_upper_vars[leg].set(defaults.upper(leg)); self.leg_lower_vars[leg].set(defaults.lower(leg))
        self.apply_geometry_from_gui()

    def generate_workspace_graph(self):
        if not hasattr(self, 'workspace_fig'):
            return
        for ax in (self.ws_ax3d, self.ws_axxy, self.ws_axxz, self.ws_axyz):
            ax.clear()
        self.workspace_cache = []
        selected = self.workspace_leg_var.get()
        legs = list(LEG_ORDER) if selected == 'ALL' else [selected]
        mode = self.workspace_mode_var.get()
        n = max(7, min(21, int(self.workspace_samples_var.get())))

        for leg in legs:
            pts = []
            if mode == 'Full leg workspace':
                q1s = np.linspace(*JOINT_LIMITS['q1'], n)
                q2s = np.linspace(*JOINT_LIMITS['q2'], n)
                q3s = np.linspace(*JOINT_LIMITS['q3'], n)
                for q1 in q1s:
                    for q2 in q2s:
                        for q3 in q3s:
                            q = {'q1':float(q1),'q2':float(q2),'q3':float(q3)}
                            p = self.fk.leg_points(leg, q)[2]
                            pts.append(p); self.workspace_cache.append((leg,q1,q2,q3,p[0],p[1],p[2]))
            else:
                joint = mode.split()[0]
                values = np.linspace(*JOINT_LIMITS[joint], max(60, n*6))
                for value in values:
                    q = dict(STAND[leg]); q[joint] = float(value)
                    p = self.fk.leg_points(leg, q)[2]
                    pts.append(p); self.workspace_cache.append((leg,q['q1'],q['q2'],q['q3'],p[0],p[1],p[2]))
            P = np.asarray(pts, dtype=float)
            if mode == 'Full leg workspace':
                self.ws_ax3d.scatter(P[:,0],P[:,1],P[:,2],s=2,alpha=0.18,label=leg)
                self.ws_axxy.scatter(P[:,0],P[:,1],s=2,alpha=0.18,label=leg)
                self.ws_axxz.scatter(P[:,0],P[:,2],s=2,alpha=0.18,label=leg)
                self.ws_axyz.scatter(P[:,1],P[:,2],s=2,alpha=0.18,label=leg)
            else:
                self.ws_ax3d.plot(P[:,0],P[:,1],P[:,2],linewidth=1.6,label=f'{leg} {mode}')
                self.ws_axxy.plot(P[:,0],P[:,1],linewidth=1.4,label=leg)
                self.ws_axxz.plot(P[:,0],P[:,2],linewidth=1.4,label=leg)
                self.ws_axyz.plot(P[:,1],P[:,2],linewidth=1.4,label=leg)

        self.ws_ax3d.set_title('(a) 3D Reachable Workspace'); self.ws_ax3d.set_xlabel('X (m)'); self.ws_ax3d.set_ylabel('Y (m)'); self.ws_ax3d.set_zlabel('Z (m)'); self.ws_ax3d.legend(fontsize=8)
        self.ws_axxy.set_title('(b) X-Y Projection'); self.ws_axxy.set_xlabel('X (m)'); self.ws_axxy.set_ylabel('Y (m)'); self.ws_axxy.axis('equal'); self.ws_axxy.grid(True,alpha=0.25)
        self.ws_axxz.set_title('(c) X-Z Projection'); self.ws_axxz.set_xlabel('X (m)'); self.ws_axxz.set_ylabel('Z (m)'); self.ws_axxz.grid(True,alpha=0.25)
        self.ws_axyz.set_title('(d) Y-Z Projection'); self.ws_axyz.set_xlabel('Y (m)'); self.ws_axyz.set_ylabel('Z (m)'); self.ws_axyz.grid(True,alpha=0.25)
        self.workspace_fig.tight_layout(); self.workspace_canvas.draw_idle()

    def save_workspace_csv(self):
        if not self.workspace_cache:
            self.generate_workspace_graph()
        path = filedialog.asksaveasfilename(title='Save Workspace Data', defaultextension='.csv', filetypes=[('CSV','*.csv')])
        if not path: return
        with open(path,'w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['leg','q1_deg','q2_deg','q3_deg','foot_x_m','foot_y_m','foot_z_m']); w.writerows(self.workspace_cache)
        self.status_var.set(f'Workspace data saved: {Path(path).name}')

    def update_planning_plot(self):
        if not hasattr(self, 'planning_fig'):
            return
        axes = self.plan_axes
        for ax in axes: ax.clear()
        u = np.linspace(0.0,1.0,500)
        profiles = {
            'Linear': (u, np.ones_like(u), np.zeros_like(u), np.zeros_like(u)),
            'Cubic': (3*u**2-2*u**3, 6*u-6*u**2, 6-12*u, np.full_like(u,-12.0)),
            'Quintic': (10*u**3-15*u**4+6*u**5, 30*u**2-60*u**3+30*u**4, 60*u-180*u**2+120*u**3, 60-360*u+360*u**2),
        }
        labels = [('Position blend s(u)',0),('Normalized velocity ds/du',1),('Normalized acceleration d²s/du²',2),('Normalized jerk d³s/du³',3)]
        for ax, (title, idx) in zip(axes, labels):
            for name, vals in profiles.items(): ax.plot(u, vals[idx], label=name)
            ax.set_xlabel('Normalized time u=t/T'); ax.set_title(title); ax.grid(True,alpha=0.25); ax.legend(fontsize=8)
        self.planning_fig.suptitle(f'Trajectory Planning Comparison | Cycle T={float(self.cycle_time_var.get()):.3f} s', fontsize=11)
        self.planning_fig.tight_layout(); self.planning_canvas.draw_idle()

    def update_singularity_plot(self):
        if not hasattr(self, 'singularity_fig'):
            return
        axes = self.sing_axes
        for ax in axes: ax.clear()
        if not self.sim_log:
            current = self.fk.all_singularity_metrics(self.joint_angles)
            legs = list(LEG_ORDER)
            sig = [current[l]['sigma_min'] for l in legs]
            cond = [current[l]['condition'] if math.isfinite(current[l]['condition']) else 1e6 for l in legs]
            det = [abs(current[l]['det']) for l in legs]
            rank = [current[l]['rank'] for l in legs]
            axes[0].bar(legs,sig); axes[0].set_title('(a) Current σmin by Leg'); axes[0].set_ylabel('σmin')
            axes[1].bar(legs,cond); axes[1].set_yscale('log'); axes[1].set_title('(b) Current Condition Number κ')
            axes[2].bar(legs,det); axes[2].set_yscale('log'); axes[2].set_title('(c) |det(J)|')
            axes[3].bar(legs,rank); axes[3].set_ylim(0,3.3); axes[3].set_title('(d) Jacobian Rank')
        else:
            t = np.asarray([r['time_s'] for r in self.sim_log],dtype=float)
            for leg in LEG_ORDER:
                axes[0].plot(t,[r[f'{leg}_sigma_min'] for r in self.sim_log],label=leg)
                cond = [min(1e6, r[f'{leg}_condition']) if math.isfinite(r[f'{leg}_condition']) else 1e6 for r in self.sim_log]
                axes[1].plot(t,cond,label=leg)
                axes[2].plot(t,[abs(r[f'{leg}_detJ']) for r in self.sim_log],label=leg)
            axes[3].plot(t,[r['robot_sigma_min'] for r in self.sim_log],label='Robot min σ')
            axes[3].plot(t,[min(1e4,r['robot_condition_max']) if math.isfinite(r['robot_condition_max']) else 1e4 for r in self.sim_log],label='Robot max κ')
            axes[0].set_title('(a) Live Minimum Singular Value'); axes[0].set_ylabel('σmin'); axes[0].legend(fontsize=7)
            axes[1].set_title('(b) Live Jacobian Condition Number'); axes[1].set_yscale('log'); axes[1].set_ylabel('κ'); axes[1].legend(fontsize=7)
            axes[2].set_title('(c) Live |det(J)|'); axes[2].set_yscale('log'); axes[2].legend(fontsize=7)
            axes[3].set_title('(d) Robot-Wide Singularity Indicators'); axes[3].set_yscale('log'); axes[3].legend(fontsize=8)
            for ax in axes: ax.set_xlabel('Time (s)'); ax.grid(True,alpha=0.25)
        self.singularity_fig.tight_layout(); self.singularity_canvas.draw_idle()

    def update_gait_plot(self):
        if not hasattr(self, 'gait_fig'):
            return
        axes = self.gait_axes
        for ax in axes: ax.clear()
        direction = self.gait_plot_direction_var.get()
        duty = min(max(float(self.duty_var.get()),0.25),0.90)
        phase = np.linspace(0,1,600)
        offsets={'FL':0.0,'RR':0.0,'FR':0.5,'RL':0.5}
        ybase={'RR':0,'RL':1,'FR':2,'FL':3}
        for leg in LEG_ORDER:
            lp=(phase+offsets[leg])%1.0
            stance=(lp<duty).astype(float)
            axes[0].step(phase*100,ybase[leg]+0.72*stance,where='post',label=leg)
        axes[0].set_yticks([0.35,1.35,2.35,3.35]); axes[0].set_yticklabels(['RR','RL','FR','FL']); axes[0].set_xlabel('Gait cycle (%)'); axes[0].set_title('(a) Trot Stance/Swing Phase Diagram'); axes[0].grid(True,alpha=0.25)

        q2_amp = 18.0; q3_lift = 28.0
        support=[]
        for leg in LEG_ORDER:
            q2=[]; q3=[]
            for p in phase:
                q=self.gait_angles(float(p),direction,q2_amp,q3_lift)
                q2.append(q[leg]['q2']); q3.append(q[leg]['q3'])
            axes[1].plot(phase*100,q2,label=leg)
            axes[2].plot(phase*100,q3,label=leg)
        for p in phase:
            support.append(len(self._support_legs(float(p))))
        axes[1].set_title('(b) Hip Pitch q2'); axes[1].set_ylabel('Angle (deg)'); axes[1].legend(fontsize=7)
        axes[2].set_title('(c) Knee q3 / Swing Clearance'); axes[2].set_ylabel('Angle (deg)'); axes[2].legend(fontsize=7)
        axes[3].plot(phase*100,support); axes[3].set_ylim(0,4.3); axes[3].set_title('(d) Number of Support Legs'); axes[3].set_ylabel('Support count')
        for ax in axes[1:]: ax.set_xlabel('Gait cycle (%)'); ax.grid(True,alpha=0.25)
        self.gait_fig.suptitle(f'Gait Analysis — {direction} | Duty factor β={duty:.2f}',fontsize=11)
        self.gait_fig.tight_layout(); self.gait_canvas.draw_idle()

    def update_error_plot(self):
        if not hasattr(self, 'error_fig'):
            return
        axes=self.error_axes
        for ax in axes: ax.clear()
        if not self.sim_log:
            axes[0].text(0.5,0.5,'Run a simulation to generate tracking-error data',ha='center',va='center',transform=axes[0].transAxes)
            axes[0].set_title('(a) HEAD Waypoint Error')
            for i,title in ((1,'(b) Cumulative MAE / RMSE'),(2,'(c) Error Distribution'),(3,'(d) Error by ML Command')):
                axes[i].set_title(title)
            self.error_fig.tight_layout(); self.error_canvas.draw_idle(); return
        t=np.asarray([r['time_s'] for r in self.sim_log],dtype=float)
        e=np.asarray([r['head_waypoint_error_m'] for r in self.sim_log],dtype=float)
        axes[0].plot(t,e); axes[0].set_title('(a) HEAD Waypoint Error vs Time'); axes[0].set_xlabel('Time (s)'); axes[0].set_ylabel('Error (m)'); axes[0].grid(True,alpha=0.25)
        n=np.arange(1,len(e)+1)
        cum_mae=np.cumsum(np.abs(e))/n
        cum_rmse=np.sqrt(np.cumsum(e**2)/n)
        axes[1].plot(t,cum_mae,label='Cumulative MAE'); axes[1].plot(t,cum_rmse,label='Cumulative RMSE'); axes[1].set_title('(b) Cumulative Error Metrics'); axes[1].set_xlabel('Time (s)'); axes[1].set_ylabel('Error (m)'); axes[1].legend(fontsize=8); axes[1].grid(True,alpha=0.25)
        if float(np.ptp(e)) < 1e-12:
            axes[2].hist(e,bins=1)
        else:
            axes[2].hist(e,bins=min(30,max(8,int(math.sqrt(len(e))))))
        axes[2].set_title('(c) Error Distribution'); axes[2].set_xlabel('HEAD error (m)'); axes[2].set_ylabel('Frequency'); axes[2].grid(True,axis='y',alpha=0.25)
        cmd_ids=sorted(set(int(r['command_index']) for r in self.sim_log))
        means=[]; maxima=[]
        for cid in cmd_ids:
            vals=[r['head_waypoint_error_m'] for r in self.sim_log if int(r['command_index'])==cid]
            means.append(float(np.mean(vals))); maxima.append(float(np.max(vals)))
        x=np.arange(len(cmd_ids)); axes[3].bar(x-0.18,means,width=0.36,label='Mean'); axes[3].bar(x+0.18,maxima,width=0.36,label='Max'); axes[3].set_xticks(x); axes[3].set_xticklabels(cmd_ids,fontsize=7); axes[3].set_title('(d) Error by ML Command'); axes[3].set_xlabel('Command index'); axes[3].set_ylabel('Error (m)'); axes[3].legend(fontsize=8); axes[3].grid(True,axis='y',alpha=0.25)
        self.error_fig.suptitle(f'Error Analysis | MAE={float(np.mean(np.abs(e))):.4f} m, RMSE={float(np.sqrt(np.mean(e**2))):.4f} m, Max={float(np.max(e)):.4f} m',fontsize=10)
        self.error_fig.tight_layout(); self.error_canvas.draw_idle()

    def save_figure_dialog(self, fig, default_name):
        path=filedialog.asksaveasfilename(
            title='Save analysis Figure', initialfile=f'{default_name}.png', defaultextension='.png',
            filetypes=[('PNG image','*.png'),('PDF vector','*.pdf'),('SVG vector','*.svg')]
        )
        if not path: return
        ext=Path(path).suffix.lower()
        dpi=600 if ext=='.png' else None
        fig.savefig(path,dpi=dpi,bbox_inches='tight')
        self.status_var.set(f'Figure saved: {Path(path).name}')

    def _refresh_all_analysis_graphs(self):
        self.generate_workspace_graph()
        self._update_path_plot()
        self.update_planning_plot()
        self.update_singularity_plot()
        self.update_gait_plot()
        self._update_score_plot()
        self.update_error_plot()
        self.update_joint_kinematics_plot()

    def export_all_analysis_figures(self):
        folder=filedialog.askdirectory(title='Select Folder for Analysis Figures')
        if not folder: return
        self._refresh_all_analysis_graphs()
        figs=[
            ('Fig01_Workspace',getattr(self,'workspace_fig',None)),
            ('Fig02_Trajectory',getattr(self,'path_fig',None)),
            ('Fig03_Trajectory_Planning',getattr(self,'planning_fig',None)),
            ('Fig04_Live_Singularity',getattr(self,'singularity_fig',None)),
            ('Fig05_Gait_Analysis',getattr(self,'gait_fig',None)),
            ('Fig06_ML_Algorithms',getattr(self,'score_fig',None)),
            ('Fig07_Error_Analysis',getattr(self,'error_fig',None)),
            ('Fig08_Selected_Joint_Kinematics',getattr(self,'joint_kinematics_fig',None)),
        ]
        out=Path(folder); count=0
        for name,fig in figs:
            if fig is None: continue
            fig.savefig(out/f'{name}.png',dpi=600,bbox_inches='tight')
            fig.savefig(out/f'{name}.pdf',bbox_inches='tight')
            count+=1
        self.status_var.set(f'Exported {count} Analysis figures as 600-dpi PNG + PDF.')
        messagebox.showinfo('Analysis Figures',f'Exported {count} figure sets (PNG + PDF) to:\n{folder}')

    def save_planned_path(self):
        if not self.plan:
            messagebox.showinfo("Save Path", "Plan a path first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "head_x_m", "head_y_m"])
            for i, (x, y) in enumerate(self.plan.get("head_path", self.plan["path"])):
                w.writerow([i, x, y])
        self.status_var.set(f"Saved planned path: {Path(path).name}")

    def save_sim_log(self):
        if not self.sim_log:
            messagebox.showinfo("Save Simulation", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path: return
        fields = list(self.sim_log[0])
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(self.sim_log)
        self.status_var.set(f"Saved simulation data: {Path(path).name}")

    def save_animation_gif(self):
        if Image is None:
            messagebox.showerror("GIF", "Install Pillow: pip install pillow")
            return
        if not self.sim_log:
            messagebox.showinfo("GIF", "Run a simulation first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".gif", filetypes=[("GIF", "*.gif")])
        if not path: return
        try:
            rows = self.sim_log
            stride = max(1, math.ceil(len(rows)/240))
            rows = rows[::stride]
            frames = []
            fig = Figure(figsize=(7.0, 5.2), dpi=85)
            ax = fig.add_subplot(111, projection="3d")
            size = max(1.5, float(self.plane_size_var.get())); half=size/2
            path_xy = np.asarray(self.plan["path"]) if self.plan else np.array([[0,0]])
            xmin=min(-half, path_xy[:,0].min()-0.5); xmax=max(half, path_xy[:,0].max()+0.5)
            ymin=min(-half, path_xy[:,1].min()-0.5); ymax=max(half, path_xy[:,1].max()+0.5)
            g=self.fk.g; L,W,H=g.body_length,g.body_width,g.body_height

            for k,row in enumerate(rows):
                ax.clear()
                bx=float(row["body_x_m"]); by=float(row["body_y_m"]); yaw=math.radians(float(row["body_yaw_deg"]))
                q={leg:{j:float(row[f"{leg}_{j}_angle_deg"]) for j in JOINTS} for leg in LEG_ORDER}
                pts_all=self.fk.all_points(q); R=rot_z(yaw)[:3,:3]; t=np.array([bx,by,0.0])
                wp=lambda p: R@np.asarray(p)+t
                z0=self.ground_z
                # grid
                for xv in np.arange(math.floor(xmin/.5)*.5, math.ceil(xmax/.5)*.5+.1,.5):
                    ax.plot([xv,xv],[ymin,ymax],[z0,z0],linewidth=.35,alpha=.18)
                for yv in np.arange(math.floor(ymin/.5)*.5, math.ceil(ymax/.5)*.5+.1,.5):
                    ax.plot([xmin,xmax],[yv,yv],[z0,z0],linewidth=.35,alpha=.18)
                if self.waypoints:
                    P=np.asarray(self.waypoints); ax.plot(P[:,0],P[:,1],np.full(len(P),z0+.01),marker="o",linewidth=1.8)
                # body
                top_b=np.array([[L/2,W/2,H/2],[L/2,-W/2,H/2],[-L/2,-W/2,H/2],[-L/2,W/2,H/2],[L/2,W/2,H/2]])
                bot_b=top_b.copy();bot_b[:,2]=-H/2
                top=np.vstack([wp(p) for p in top_b]);bot=np.vstack([wp(p) for p in bot_b])
                ax.plot(top[:,0],top[:,1],top[:,2],linewidth=2);ax.plot(bot[:,0],bot[:,1],bot[:,2],linewidth=2)
                for i in range(4): ax.plot([top[i,0],bot[i,0]],[top[i,1],bot[i,1]],[top[i,2],bot[i,2]],linewidth=1)

                # Head.
                ht_b=np.array([
                    [L/2+.025,W*.30,H*.42],[L/2+.120,W*.30,H*.42],
                    [L/2+.120,-W*.30,H*.42],[L/2+.025,-W*.30,H*.42],
                    [L/2+.025,W*.30,H*.42]
                ])
                hb_b=ht_b.copy();hb_b[:,2]=-H*.05
                ht=np.vstack([wp(p) for p in ht_b]);hb=np.vstack([wp(p) for p in hb_b])
                ax.plot(ht[:,0],ht[:,1],ht[:,2],linewidth=1.5);ax.plot(hb[:,0],hb[:,1],hb[:,2],linewidth=1.5)
                for i in range(4): ax.plot([ht[i,0],hb[i,0]],[ht[i,1],hb[i,1]],[ht[i,2],hb[i,2]],linewidth=.8)

                # Tail.
                rear=wp(np.array([-L/2,0,0]));tail=wp(np.array([-L/2-.12,0,.07]))
                ax.plot([rear[0],tail[0]],[rear[1],tail[1]],[rear[2],tail[2]],linewidth=1.8)

                for leg,(h,kn,ft) in pts_all.items():
                    P=np.vstack([wp(h),wp(kn),wp(ft)]);ax.plot(P[:,0],P[:,1],P[:,2],marker="o",linewidth=2.5,markersize=3)
                body_trail=np.array([[float(r["body_x_m"]),float(r["body_y_m"])] for r in rows[:k+1]])
                ax.plot(
                    body_trail[:,0], body_trail[:,1],
                    np.full(len(body_trail),z0+.018),
                    linewidth=1.0, alpha=.55
                )
                if "head_x_m" in row:
                    head_trail=np.array([[float(r["head_x_m"]),float(r["head_y_m"])] for r in rows[:k+1]])
                    ax.plot(
                        head_trail[:,0],head_trail[:,1],
                        np.full(len(head_trail),z0+.028),
                        linewidth=2.0
                    )
                    ax.scatter(
                        [float(row["head_x_m"])],
                        [float(row["head_y_m"])],
                        [z0+.045], s=45, marker="*"
                    )
                ax.set_xlim(xmin,xmax);ax.set_ylim(ymin,ymax);ax.set_zlim(min(z0-.06,-.32),.28)
                ax.set_xlabel("+X Forward");ax.set_ylabel("+Y Left");ax.set_zlabel("Z")
                ax.set_title(f"ML Path Following | t={float(row['time_s']):.2f}s | {row['direction']}")
                ax.view_init(elev=25,azim=-55)
                buf=BytesIO();fig.savefig(buf,format="png",bbox_inches="tight");buf.seek(0)
                im=Image.open(buf).convert("P",palette=Image.ADAPTIVE);frames.append(im.copy());buf.close()
            frames[0].save(path,save_all=True,append_images=frames[1:],duration=50,loop=0,optimize=False)
            self.status_var.set(f"Saved animation: {Path(path).name}")
        except Exception as exc:
            messagebox.showerror("Save GIF", str(exc))


# Additional imports for V1.9 path planning
import heapq
import time
from matplotlib.patches import Circle

# ---------------------------------------------------------------------
# Path-planning helpers
# ---------------------------------------------------------------------
def euclid(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def path_length_2d(path):
    if not path or len(path) < 2:
        return 0.0
    return float(sum(euclid(path[i], path[i+1]) for i in range(len(path)-1)))


def path_smoothness_deg(path):
    """Total absolute heading-change along the polyline path."""
    if not path or len(path) < 3:
        return 0.0
    headings = []
    for i in range(len(path)-1):
        dx = float(path[i+1][0]) - float(path[i][0])
        dy = float(path[i+1][1]) - float(path[i][1])
        headings.append(math.atan2(dy, dx))
    total = 0.0
    for i in range(len(headings)-1):
        da = (headings[i+1] - headings[i] + math.pi) % (2*math.pi) - math.pi
        total += abs(math.degrees(da))
    return float(total)


def in_bounds(pt, bounds):
    x, y = float(pt[0]), float(pt[1])
    xmin, xmax, ymin, ymax = bounds
    return (xmin <= x <= xmax) and (ymin <= y <= ymax)


def point_collision_free(pt, obstacles, clearance=0.0):
    x, y = float(pt[0]), float(pt[1])
    for ox, oy, r in obstacles:
        if math.hypot(x-float(ox), y-float(oy)) <= float(r) + float(clearance):
            return False
    return True


def segment_collision_free(p0, p1, obstacles, clearance=0.0, step=0.03):
    if not point_collision_free(p0, obstacles, clearance):
        return False
    if not point_collision_free(p1, obstacles, clearance):
        return False
    L = euclid(p0, p1)
    n = max(2, int(math.ceil(L / max(step, 1e-6))) + 1)
    for i in range(n + 1):
        t = i / n
        x = (1.0 - t) * float(p0[0]) + t * float(p1[0])
        y = (1.0 - t) * float(p0[1]) + t * float(p1[1])
        if not point_collision_free((x, y), obstacles, clearance):
            return False
    return True


def shorten_path(path, obstacles, clearance=0.0):
    """Greedy line-of-sight path shortening."""
    if not path or len(path) <= 2:
        return list(path or [])
    short = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if segment_collision_free(path[i], path[j], obstacles, clearance):
                break
            j -= 1
        short.append(path[j])
        i = j
    return short


def _nearest_free_index(occ, i0, j0):
    nx, ny = occ.shape
    if 0 <= i0 < nx and 0 <= j0 < ny and not occ[i0, j0]:
        return (i0, j0)
    best = None
    best_d = 1e9
    for i in range(nx):
        for j in range(ny):
            if occ[i, j]:
                continue
            d = (i - i0)**2 + (j - j0)**2
            if d < best_d:
                best_d = d
                best = (i, j)
    return best


def grid_search_planner(start, goal, obstacles, bounds, resolution=0.08,
                        use_astar=True, clearance=0.04):
    t0 = time.perf_counter()
    xmin, xmax, ymin, ymax = bounds
    resolution = max(0.03, float(resolution))

    xs = np.arange(xmin, xmax + 0.5 * resolution, resolution)
    ys = np.arange(ymin, ymax + 0.5 * resolution, resolution)
    nx, ny = len(xs), len(ys)

    occ = np.zeros((nx, ny), dtype=bool)
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            if not point_collision_free((x, y), obstacles, clearance):
                occ[i, j] = True

    si = int(np.argmin(np.abs(xs - float(start[0]))))
    sj = int(np.argmin(np.abs(ys - float(start[1]))))
    gi = int(np.argmin(np.abs(xs - float(goal[0]))))
    gj = int(np.argmin(np.abs(ys - float(goal[1]))))

    sidx = _nearest_free_index(occ, si, sj)
    gidx = _nearest_free_index(occ, gi, gj)
    if sidx is None or gidx is None:
        return {
            "name": "A*" if use_astar else "Dijkstra",
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": 0,
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": [],
            "message": "Start or goal is fully blocked."
        }

    neigh = [
        (-1, 0, resolution), (1, 0, resolution),
        (0, -1, resolution), (0, 1, resolution),
        (-1, -1, resolution * math.sqrt(2)),
        (-1, 1, resolution * math.sqrt(2)),
        (1, -1, resolution * math.sqrt(2)),
        (1, 1, resolution * math.sqrt(2)),
    ]

    def h(idx):
        return euclid((xs[idx[0]], ys[idx[1]]), (xs[gidx[0]], ys[gidx[1]])) if use_astar else 0.0

    pq = [(h(sidx), 0.0, sidx)]
    parent = {sidx: None}
    g_cost = {sidx: 0.0}
    visited = set()
    expanded = 0

    while pq:
        _, gcur, idx = heapq.heappop(pq)
        if idx in visited:
            continue
        visited.add(idx)
        expanded += 1

        if idx == gidx:
            break

        i, j = idx
        for di, dj, step_cost in neigh:
            ni, nj = i + di, j + dj
            if not (0 <= ni < nx and 0 <= nj < ny):
                continue
            if occ[ni, nj]:
                continue
            cand = gcur + step_cost
            nidx = (ni, nj)
            if cand < g_cost.get(nidx, float("inf")):
                g_cost[nidx] = cand
                parent[nidx] = idx
                heapq.heappush(pq, (cand + h(nidx), cand, nidx))

    if gidx not in parent:
        return {
            "name": "A*" if use_astar else "Dijkstra",
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": expanded,
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": [],
            "message": "No feasible grid path found."
        }

    path = []
    cur = gidx
    while cur is not None:
        path.append((float(xs[cur[0]]), float(ys[cur[1]])))
        cur = parent[cur]
    path.reverse()

    path[0] = (float(start[0]), float(start[1]))
    path[-1] = (float(goal[0]), float(goal[1]))
    path = shorten_path(path, obstacles, clearance)

    elapsed = time.perf_counter() - t0
    return {
        "name": "A*" if use_astar else "Dijkstra",
        "success": True,
        "path": path,
        "time_s": elapsed,
        "path_length_m": path_length_2d(path),
        "nodes": expanded,
        "points": len(path),
        "smoothness_deg": path_smoothness_deg(path),
        "history": [],
        "message": "OK",
    }


def _steer(p_from, p_to, step_size):
    d = euclid(p_from, p_to)
    if d <= step_size:
        return (float(p_to[0]), float(p_to[1]))
    t = step_size / max(d, 1e-9)
    return (
        (1.0 - t) * float(p_from[0]) + t * float(p_to[0]),
        (1.0 - t) * float(p_from[1]) + t * float(p_to[1]),
    )


def rrt_planner(start, goal, obstacles, bounds, step_size=0.12, max_iter=2200,
                goal_sample_rate=0.12, clearance=0.04, seed=42,
                rewire=False, rewire_radius=0.30):
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    name = "RRT*" if rewire else "RRT"

    if not point_collision_free(start, obstacles, clearance):
        return {
            "name": name, "success": False, "path": [],
            "time_s": 0.0, "path_length_m": float("nan"), "nodes": 0,
            "points": 0, "smoothness_deg": float("nan"), "history": [],
            "message": "Start is inside an obstacle."
        }
    if not point_collision_free(goal, obstacles, clearance):
        return {
            "name": name, "success": False, "path": [],
            "time_s": 0.0, "path_length_m": float("nan"), "nodes": 0,
            "points": 0, "smoothness_deg": float("nan"), "history": [],
            "message": "Goal is inside an obstacle."
        }

    xmin, xmax, ymin, ymax = bounds
    nodes = [(float(start[0]), float(start[1]))]
    parents = [-1]
    costs = [0.0]
    goal_indices = []
    history = []
    best_goal_cost = float("nan")

    for it in range(int(max_iter)):
        if rng.random() < float(goal_sample_rate):
            sample = (float(goal[0]), float(goal[1]))
        else:
            sample = (
                float(rng.uniform(xmin, xmax)),
                float(rng.uniform(ymin, ymax))
            )

        dists = [euclid(n, sample) for n in nodes]
        nearest = int(np.argmin(dists))
        new_pt = _steer(nodes[nearest], sample, max(0.03, float(step_size)))
        if not in_bounds(new_pt, bounds):
            history.append(best_goal_cost)
            continue
        if not segment_collision_free(nodes[nearest], new_pt, obstacles, clearance):
            history.append(best_goal_cost)
            continue

        parent = nearest
        new_cost = costs[nearest] + euclid(nodes[nearest], new_pt)

        if rewire:
            near = [i for i, n in enumerate(nodes) if euclid(n, new_pt) <= float(rewire_radius)]
            for i in near:
                if segment_collision_free(nodes[i], new_pt, obstacles, clearance):
                    c = costs[i] + euclid(nodes[i], new_pt)
                    if c < new_cost:
                        parent = i
                        new_cost = c
        nodes.append(new_pt)
        parents.append(parent)
        costs.append(new_cost)
        new_idx = len(nodes) - 1

        if rewire:
            for i, n in enumerate(nodes[:-1]):
                if euclid(n, new_pt) <= float(rewire_radius):
                    c = new_cost + euclid(new_pt, n)
                    if c + 1e-12 < costs[i] and segment_collision_free(new_pt, n, obstacles, clearance):
                        parents[i] = new_idx
                        costs[i] = c

        if euclid(new_pt, goal) <= max(0.03, float(step_size)) and segment_collision_free(new_pt, goal, obstacles, clearance):
            goal_cost = new_cost + euclid(new_pt, goal)
            nodes.append((float(goal[0]), float(goal[1])))
            parents.append(new_idx)
            costs.append(goal_cost)
            goal_indices.append(len(nodes) - 1)

            if (not rewire):
                best_goal_cost = goal_cost
                history.append(best_goal_cost)
                break
            else:
                if (not math.isfinite(best_goal_cost)) or goal_cost < best_goal_cost:
                    best_goal_cost = goal_cost

        history.append(best_goal_cost)

    if not goal_indices:
        return {
            "name": name,
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": len(nodes),
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": history,
            "message": "No path found within iteration limit.",
        }

    best_goal = min(goal_indices, key=lambda idx: costs[idx])
    path = []
    cur = best_goal
    while cur != -1:
        path.append(nodes[cur])
        cur = parents[cur]
    path.reverse()

    path[0] = (float(start[0]), float(start[1]))
    path[-1] = (float(goal[0]), float(goal[1]))
    path = shorten_path(path, obstacles, clearance)

    return {
        "name": name,
        "success": True,
        "path": path,
        "time_s": time.perf_counter() - t0,
        "path_length_m": path_length_2d(path),
        "nodes": len(nodes),
        "points": len(path),
        "smoothness_deg": path_smoothness_deg(path),
        "history": history,
        "message": "OK",
    }


# ---------------------------------------------------------------------
# Extended GUI
# ---------------------------------------------------------------------
class MLRobotDogAppV19(MLRobotDogApp):
    def __init__(self):
        super().__init__()
        self.title("ROBOQUAD FK V1.9 — Path Planning Algorithms + ML Robot Dog")

        # Rename the older trajectory-profile tab for clarity.
        try:
            self.main_nb.tab(5, text="Trajectory Profiles")
        except Exception:
            pass

        # State for classical path planning.
        self.pp_goal_x_var = tk.DoubleVar(value=self.waypoints[-1][0] if self.waypoints else 1.0)
        self.pp_goal_y_var = tk.DoubleVar(value=self.waypoints[-1][1] if self.waypoints else 0.8)
        self.pp_algo_var = tk.StringVar(value="A*")
        self.pp_resolution_var = tk.DoubleVar(value=0.08)
        self.pp_max_iter_var = tk.IntVar(value=2200)
        self.pp_step_size_var = tk.DoubleVar(value=0.12)
        self.pp_goal_bias_var = tk.DoubleVar(value=0.12)
        self.pp_rewire_radius_var = tk.DoubleVar(value=0.32)
        self.pp_clearance_var = tk.DoubleVar(value=0.05)
        self.pp_obs_x_var = tk.DoubleVar(value=0.55)
        self.pp_obs_y_var = tk.DoubleVar(value=0.25)
        self.pp_obs_r_var = tk.DoubleVar(value=0.18)
        self.pp_status_var = tk.StringVar(value="Planner status: no path planned yet.")
        self.pp_obstacles = [
            (0.45, 0.22, 0.16),
            (0.90, 0.48, 0.18),
            (0.65, -0.18, 0.14),
        ]
        self.pp_results = {}
        self.pp_selected_path_name = None

        # Add a horizontal top-level tab.
        self.pathplan_tab = ttk.Frame(self.main_nb)
        try:
            self.main_nb.insert(6, self.pathplan_tab, text="Path Planning")
        except Exception:
            self.main_nb.add(self.pathplan_tab, text="Path Planning")
        self._build_path_planning_tab(self.pathplan_tab)

        self.status_var.set("V1.9 ready. Classical path planning + ML locomotion comparison enabled.")

    # ------------------------------------------------------------------
    # Path-planning UI
    # ------------------------------------------------------------------
    def _build_path_planning_tab(self, parent):
        paned = ttk.Panedwindow(parent, orient="horizontal")
        paned.pack(fill="both", expand=True)

        left_wrap = ttk.Frame(paned, width=460)
        right = ttk.Frame(paned)
        paned.add(left_wrap, weight=0)
        paned.add(right, weight=1)

        scroll = ScrollableFrame(left_wrap)
        scroll.pack(fill="both", expand=True)
        left = scroll.inner

        intro = ttk.LabelFrame(left, text="Classical Path Planning Layer", padding=8)
        intro.pack(fill="x", padx=8, pady=(8, 4))
        ttk.Label(
            intro,
            text=(
                "Plan a collision-free 2D path using Dijkstra, A*, RRT, or RRT*. "
                "Then transfer the selected planner path to the existing ML waypoint follower. "
                "This keeps the inherited machine-learning controller unchanged while enabling "
                "algorithm-level comparison for Analysis analysis."
            ),
            wraplength=405,
            justify="left"
        ).pack(anchor="w")

        sg = ttk.LabelFrame(left, text="1. Start and Goal", padding=8)
        sg.pack(fill="x", padx=8, pady=4)
        ttk.Label(sg, text="Planner start uses the same Start X / Y from ML Control.").grid(row=0, column=0, columnspan=4, sticky="w", pady=(0, 5))
        ttk.Label(sg, text="Goal X (m)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(sg, textvariable=self.pp_goal_x_var, width=10).grid(row=1, column=1, padx=4, pady=2)
        ttk.Label(sg, text="Goal Y (m)").grid(row=1, column=2, sticky="w", pady=2)
        ttk.Entry(sg, textvariable=self.pp_goal_y_var, width=10).grid(row=1, column=3, padx=4, pady=2)
        ttk.Button(sg, text="Use Last ML Waypoint as Goal", command=self.pp_use_last_waypoint_as_goal).grid(row=2, column=0, columnspan=4, sticky="ew", pady=(6, 0))

        obs = ttk.LabelFrame(left, text="2. Obstacles (circles on ground plane)", padding=8)
        obs.pack(fill="x", padx=8, pady=4)
        self.pp_obs_list = tk.Listbox(obs, height=6, exportselection=False)
        self.pp_obs_list.pack(fill="x")

        row = ttk.Frame(obs)
        row.pack(fill="x", pady=(5, 0))
        ttk.Label(row, text="X").pack(side="left")
        ttk.Entry(row, textvariable=self.pp_obs_x_var, width=7).pack(side="left", padx=2)
        ttk.Label(row, text="Y").pack(side="left")
        ttk.Entry(row, textvariable=self.pp_obs_y_var, width=7).pack(side="left", padx=2)
        ttk.Label(row, text="R").pack(side="left")
        ttk.Entry(row, textvariable=self.pp_obs_r_var, width=7).pack(side="left", padx=2)

        row2 = ttk.Frame(obs)
        row2.pack(fill="x", pady=(5, 0))
        ttk.Button(row2, text="Add", command=self.pp_add_obstacle).pack(side="left", padx=2)
        ttk.Button(row2, text="Remove", command=self.pp_remove_obstacle).pack(side="left", padx=2)
        ttk.Button(row2, text="Clear", command=self.pp_clear_obstacles).pack(side="left", padx=2)
        ttk.Button(row2, text="Load Example", command=self.pp_load_obstacle_example).pack(side="left", padx=2)

        pars = ttk.LabelFrame(left, text="3. Planner Settings", padding=8)
        pars.pack(fill="x", padx=8, pady=4)

        ttk.Label(pars, text="Algorithm").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Combobox(
            pars,
            textvariable=self.pp_algo_var,
            state="readonly",
            values=("Dijkstra", "A*", "RRT", "RRT*"),
            width=16
        ).grid(row=0, column=1, padx=4, pady=2, sticky="ew")

        settings = (
            ("Grid resolution (m)", self.pp_resolution_var),
            ("RRT step size (m)", self.pp_step_size_var),
            ("RRT goal bias", self.pp_goal_bias_var),
            ("RRT max iterations", self.pp_max_iter_var),
            ("RRT* rewire radius (m)", self.pp_rewire_radius_var),
            ("Obstacle clearance (m)", self.pp_clearance_var),
        )
        for i, (label, var) in enumerate(settings, start=1):
            ttk.Label(pars, text=label).grid(row=i, column=0, sticky="w", pady=2)
            ttk.Entry(pars, textvariable=var, width=12).grid(row=i, column=1, padx=4, pady=2, sticky="ew")

        acts = ttk.LabelFrame(left, text="4. Run / Transfer / Export", padding=8)
        acts.pack(fill="x", padx=8, pady=4)
        ttk.Button(acts, text="PLAN SELECTED ALGORITHM", command=self.pp_plan_selected).pack(fill="x", pady=2)
        ttk.Button(acts, text="COMPARE ALL ALGORITHMS", command=self.pp_compare_all).pack(fill="x", pady=2)
        ttk.Button(acts, text="Use Selected Path as ML Waypoints", command=self.pp_use_selected_path_as_ml_waypoints).pack(fill="x", pady=2)
        ttk.Button(acts, text="Use Shortest Successful Path as ML Waypoints", command=self.pp_use_best_path_as_ml_waypoints).pack(fill="x", pady=2)
        ttk.Button(acts, text="Save Planner Summary CSV", command=self.pp_save_summary_csv).pack(fill="x", pady=(6, 2))
        ttk.Button(acts, text="Save Selected Planner Path CSV", command=self.pp_save_selected_path_csv).pack(fill="x", pady=2)
        ttk.Button(
            acts,
            text="Save Comparison Figure",
            command=lambda: self.save_figure_dialog(self.pp_fig, "Fig09_Path_Planning_Algorithms")
        ).pack(fill="x", pady=(8, 2))

        stat = ttk.LabelFrame(left, text="5. Status", padding=8)
        stat.pack(fill="x", padx=8, pady=(4, 10))
        ttk.Label(stat, textvariable=self.pp_status_var, wraplength=400, justify="left").pack(anchor="w")

        # Right side: result table + analysis figure
        top = ttk.Frame(right, padding=(6, 6, 6, 0))
        top.pack(fill="x")
        ttk.Label(top, text="Path-Planning Comparison", style="Header.TLabel").pack(side="left")

        cols = ("alg", "success", "time", "length", "nodes", "points", "smooth")
        self.pp_tree = ttk.Treeview(right, columns=cols, show="headings", height=5)
        heads = {
            "alg": "Algorithm",
            "success": "Success",
            "time": "Time (s)",
            "length": "Path length (m)",
            "nodes": "Nodes / Expansions",
            "points": "Path points",
            "smooth": "Smoothness (deg)",
        }
        widths = {"alg":120, "success":75, "time":90, "length":115, "nodes":130, "points":90, "smooth":120}
        for c in cols:
            self.pp_tree.heading(c, text=heads[c])
            self.pp_tree.column(c, width=widths[c], anchor="center")
        self.pp_tree.pack(fill="x", padx=6, pady=(4, 6))
        self.pp_tree.bind("<<TreeviewSelect>>", lambda e: self.pp_on_tree_select())

        self.pp_fig = Figure(figsize=(12.5, 7.4), dpi=100)
        self.pp_axes = [self.pp_fig.add_subplot(231),
                        self.pp_fig.add_subplot(232),
                        self.pp_fig.add_subplot(233),
                        self.pp_fig.add_subplot(234),
                        self.pp_fig.add_subplot(235),
                        self.pp_fig.add_subplot(236)]
        self.pp_canvas = FigureCanvasTkAgg(self.pp_fig, master=right)
        self.pp_canvas.get_tk_widget().pack(fill="both", expand=True, padx=4, pady=(0, 6))

        self.pp_refresh_obstacle_list()
        self.pp_update_figure()

    # ------------------------------------------------------------------
    # Path-planning data and actions
    # ------------------------------------------------------------------
    def pp_use_last_waypoint_as_goal(self):
        if self.waypoints:
            gx, gy = self.waypoints[-1]
            self.pp_goal_x_var.set(gx)
            self.pp_goal_y_var.set(gy)
            self.pp_status_var.set("Planner goal copied from the last ML waypoint.")
        else:
            self.pp_status_var.set("No ML waypoint available to copy.")

    def pp_refresh_obstacle_list(self):
        if not hasattr(self, "pp_obs_list"):
            return
        self.pp_obs_list.delete(0, "end")
        for i, (x, y, r) in enumerate(self.pp_obstacles):
            self.pp_obs_list.insert("end", f"O{i+1}: X={x:+.3f}, Y={y:+.3f}, R={r:.3f}")

    def pp_add_obstacle(self):
        try:
            x = float(self.pp_obs_x_var.get())
            y = float(self.pp_obs_y_var.get())
            r = float(self.pp_obs_r_var.get())
            if r <= 0.0:
                raise ValueError
            self.pp_obstacles.append((x, y, r))
            self.pp_refresh_obstacle_list()
            self.pp_update_figure()
        except Exception:
            messagebox.showerror("Obstacle", "Enter valid numeric X, Y and positive radius.")

    def pp_remove_obstacle(self):
        sel = self.pp_obs_list.curselection()
        if not sel:
            return
        del self.pp_obstacles[int(sel[0])]
        self.pp_refresh_obstacle_list()
        self.pp_update_figure()

    def pp_clear_obstacles(self):
        self.pp_obstacles = []
        self.pp_refresh_obstacle_list()
        self.pp_update_figure()

    def pp_load_obstacle_example(self):
        self.pp_obstacles = [
            (0.32, 0.18, 0.15),
            (0.72, 0.30, 0.16),
            (0.68, -0.18, 0.13),
            (1.02, 0.06, 0.14),
        ]
        self.pp_refresh_obstacle_list()
        self.pp_update_figure()
        self.pp_status_var.set("Example obstacle field loaded.")

    def _planner_bounds(self):
        half = max(1.0, float(self.plane_size_var.get()) / 2.0)
        return (-half, half, -half, half)

    def _planner_start_goal(self):
        start = (float(self.start_x_var.get()), float(self.start_y_var.get()))
        goal = (float(self.pp_goal_x_var.get()), float(self.pp_goal_y_var.get()))
        return start, goal

    def _run_one_planner(self, name):
        start, goal = self._planner_start_goal()
        bounds = self._planner_bounds()
        clearance = float(self.pp_clearance_var.get())
        if name == "Dijkstra":
            return grid_search_planner(
                start, goal, self.pp_obstacles, bounds,
                resolution=float(self.pp_resolution_var.get()),
                use_astar=False,
                clearance=clearance
            )
        if name == "A*":
            return grid_search_planner(
                start, goal, self.pp_obstacles, bounds,
                resolution=float(self.pp_resolution_var.get()),
                use_astar=True,
                clearance=clearance
            )
        if name == "RRT":
            return rrt_planner(
                start, goal, self.pp_obstacles, bounds,
                step_size=float(self.pp_step_size_var.get()),
                max_iter=int(self.pp_max_iter_var.get()),
                goal_sample_rate=float(self.pp_goal_bias_var.get()),
                clearance=clearance,
                seed=int(self.seed_var.get()),
                rewire=False,
                rewire_radius=float(self.pp_rewire_radius_var.get())
            )
        if name == "RRT*":
            return rrt_planner(
                start, goal, self.pp_obstacles, bounds,
                step_size=float(self.pp_step_size_var.get()),
                max_iter=int(self.pp_max_iter_var.get()),
                goal_sample_rate=float(self.pp_goal_bias_var.get()),
                clearance=clearance,
                seed=int(self.seed_var.get()),
                rewire=True,
                rewire_radius=float(self.pp_rewire_radius_var.get())
            )
        raise ValueError(f"Unknown planner: {name}")

    def pp_plan_selected(self):
        alg = self.pp_algo_var.get()
        self.status_var.set(f"Running path planner: {alg} ...")
        self.update_idletasks()
        result = self._run_one_planner(alg)
        self.pp_results = {alg: result}
        self.pp_selected_path_name = alg
        self.pp_refresh_results_tree()
        self.pp_update_figure()
        msg = result.get("message", "")
        self.pp_status_var.set(
            f"{alg}: {'SUCCESS' if result['success'] else 'FAILED'} | "
            f"time={result['time_s']:.4f} s | "
            f"path length={result['path_length_m']:.4f} m | "
            f"nodes={result['nodes']} | {msg}"
        )
        self.status_var.set(f"{alg} planning complete.")

    def pp_compare_all(self):
        algs = ["Dijkstra", "A*", "RRT", "RRT*"]
        self.status_var.set("Comparing Dijkstra, A*, RRT and RRT* ...")
        self.update_idletasks()
        self.pp_results = {name: self._run_one_planner(name) for name in algs}

        successful = [r for r in self.pp_results.values() if r["success"]]
        if successful:
            best = min(successful, key=lambda r: (r["path_length_m"], r["time_s"]))
            self.pp_selected_path_name = best["name"]
            self.pp_algo_var.set(best["name"])
            self.pp_status_var.set(
                f"Comparison complete. Best successful path = {best['name']} | "
                f"path length={best['path_length_m']:.4f} m | "
                f"time={best['time_s']:.4f} s | "
                f"smoothness={best['smoothness_deg']:.2f}°"
            )
        else:
            self.pp_selected_path_name = None
            self.pp_status_var.set("Comparison complete. No algorithm found a feasible path.")
        self.pp_refresh_results_tree()
        self.pp_update_figure()
        self.status_var.set("Path-planning comparison complete.")

    def pp_refresh_results_tree(self):
        if not hasattr(self, "pp_tree"):
            return
        for item in self.pp_tree.get_children():
            self.pp_tree.delete(item)

        order = ["Dijkstra", "A*", "RRT", "RRT*"]
        keys = [k for k in order if k in self.pp_results] + [k for k in self.pp_results if k not in order]
        for k in keys:
            r = self.pp_results[k]
            self.pp_tree.insert(
                "",
                "end",
                iid=k,
                values=(
                    k,
                    "Yes" if r["success"] else "No",
                    f"{r['time_s']:.4f}",
                    f"{r['path_length_m']:.4f}" if math.isfinite(r["path_length_m"]) else "—",
                    f"{r['nodes']}",
                    f"{r['points']}",
                    f"{r['smoothness_deg']:.2f}" if math.isfinite(r["smoothness_deg"]) else "—",
                )
            )

        if self.pp_selected_path_name and self.pp_selected_path_name in self.pp_tree.get_children():
            self.pp_tree.selection_set(self.pp_selected_path_name)

    def pp_on_tree_select(self):
        sel = self.pp_tree.selection()
        if sel:
            self.pp_selected_path_name = sel[0]
            self.pp_algo_var.set(sel[0])
            self.pp_update_figure()

    def pp_update_figure(self):
        if not hasattr(self, "pp_fig"):
            return
        axes = self.pp_axes
        for ax in axes:
            ax.clear()

        start, goal = self._planner_start_goal()
        bounds = self._planner_bounds()

        # (a) 2D path overlay with obstacles
        ax = axes[0]
        for (ox, oy, r) in self.pp_obstacles:
            circ = Circle((ox, oy), r, fill=False, linewidth=1.6)
            ax.add_patch(circ)
        ax.scatter([start[0]], [start[1]], marker="s", s=60, label="Start")
        ax.scatter([goal[0]], [goal[1]], marker="*", s=100, label="Goal")
        for name, res in self.pp_results.items():
            if res["success"] and res["path"]:
                P = np.asarray(res["path"], dtype=float)
                lw = 2.6 if name == self.pp_selected_path_name else 1.6
                ax.plot(P[:, 0], P[:, 1], linewidth=lw, label=name)
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect("equal")
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.set_title("(a) Path Planning Overlay")
        ax.grid(True, alpha=0.25)
        if self.pp_results or self.pp_obstacles:
            ax.legend(fontsize=7, loc="best")

        alg_order = [a for a in ("Dijkstra", "A*", "RRT", "RRT*") if a in self.pp_results]
        if alg_order:
            times = [self.pp_results[a]["time_s"] for a in alg_order]
            lengths = [self.pp_results[a]["path_length_m"] if math.isfinite(self.pp_results[a]["path_length_m"]) else 0.0 for a in alg_order]
            nodes = [self.pp_results[a]["nodes"] for a in alg_order]
            smooth = [self.pp_results[a]["smoothness_deg"] if math.isfinite(self.pp_results[a]["smoothness_deg"]) else 0.0 for a in alg_order]
            points = [self.pp_results[a]["points"] for a in alg_order]

            axes[1].bar(alg_order, times)
            axes[1].set_title("(b) Planning Time")
            axes[1].set_ylabel("Time (s)")
            axes[1].tick_params(axis='x', rotation=15)
            axes[1].grid(True, axis="y", alpha=0.25)

            axes[2].bar(alg_order, lengths)
            axes[2].set_title("(c) Path Length")
            axes[2].set_ylabel("Length (m)")
            axes[2].tick_params(axis='x', rotation=15)
            axes[2].grid(True, axis="y", alpha=0.25)

            axes[3].bar(alg_order, nodes)
            axes[3].set_title("(d) Nodes / Expansions")
            axes[3].set_ylabel("Count")
            axes[3].tick_params(axis='x', rotation=15)
            axes[3].grid(True, axis="y", alpha=0.25)

            axes[4].bar(alg_order, smooth)
            axes[4].set_title("(e) Path Smoothness")
            axes[4].set_ylabel("Σ|Δ heading| (deg)")
            axes[4].tick_params(axis='x', rotation=15)
            axes[4].grid(True, axis="y", alpha=0.25)

            any_hist = any(len(self.pp_results[a].get("history", [])) > 0 for a in alg_order)
            if any_hist:
                for a in alg_order:
                    hist = self.pp_results[a].get("history", [])
                    if not hist:
                        continue
                    y = np.asarray(hist, dtype=float)
                    if np.isfinite(y).any():
                        axes[5].plot(np.arange(1, len(y)+1), y, linewidth=1.4, label=a)
                axes[5].set_title("(f) Sampling Planner Convergence")
                axes[5].set_xlabel("Iteration")
                axes[5].set_ylabel("Best path cost (m)")
                axes[5].grid(True, alpha=0.25)
                axes[5].legend(fontsize=7)
            else:
                axes[5].bar(alg_order, points)
                axes[5].set_title("(f) Path Points")
                axes[5].set_ylabel("Count")
                axes[5].tick_params(axis='x', rotation=15)
                axes[5].grid(True, axis="y", alpha=0.25)

            successful = [self.pp_results[a] for a in alg_order if self.pp_results[a]["success"]]
            if successful:
                best = min(successful, key=lambda r: (r["path_length_m"], r["time_s"]))
                ttl = (
                    f"Path Planning Algorithm Comparison | "
                    f"Best={best['name']} | "
                    f"L={best['path_length_m']:.4f} m | "
                    f"T={best['time_s']:.4f} s"
                )
            else:
                ttl = "Path Planning Algorithm Comparison | No feasible path found"
            self.pp_fig.suptitle(ttl, fontsize=11)
        else:
            titles = [
                "(b) Planning Time", "(c) Path Length", "(d) Nodes / Expansions",
                "(e) Path Smoothness", "(f) Sampling Planner Convergence / Path Points"
            ]
            for ax, title in zip(axes[1:], titles):
                ax.text(0.5, 0.5, "Run a planner to populate this graph.", ha="center", va="center", transform=ax.transAxes)
                ax.set_title(title)
            self.pp_fig.suptitle("Path Planning Algorithm Comparison", fontsize=11)

        self.pp_fig.tight_layout()
        self.pp_canvas.draw_idle()

    def pp_use_selected_path_as_ml_waypoints(self):
        name = self.pp_selected_path_name or self.pp_algo_var.get()
        if not name or name not in self.pp_results:
            messagebox.showinfo("Path Planning", "Run a planner first.")
            return
        r = self.pp_results[name]
        if not r["success"] or not r["path"]:
            messagebox.showinfo("Path Planning", f"{name} does not currently have a successful path.")
            return
        self.waypoints = list(r["path"])
        self._refresh_waypoints()
        self.plan = None
        self.body_trail = []
        self.head_trail = []
        self._update_path_plot()
        self.status_var.set(f"{name} path transferred to ML waypoints.")
        self.pp_status_var.set(f"Transferred {name} path to ML Control. You can now use Plan Path / Simulate.")
        try:
            self.main_nb.select(0)
        except Exception:
            pass

    def pp_use_best_path_as_ml_waypoints(self):
        successful = [r for r in self.pp_results.values() if r["success"]]
        if not successful:
            messagebox.showinfo("Path Planning", "No successful planner path is available.")
            return
        best = min(successful, key=lambda r: (r["path_length_m"], r["time_s"]))
        self.pp_selected_path_name = best["name"]
        self.pp_algo_var.set(best["name"])
        self.pp_use_selected_path_as_ml_waypoints()

    def pp_save_summary_csv(self):
        if not self.pp_results:
            messagebox.showinfo("Path Planning", "Run at least one path planner first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save Path Planner Summary",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "algorithm", "success", "time_s", "path_length_m",
                "nodes_expanded", "path_points", "smoothness_deg", "message"
            ])
            for name in ("Dijkstra", "A*", "RRT", "RRT*"):
                if name not in self.pp_results:
                    continue
                r = self.pp_results[name]
                w.writerow([
                    name, int(r["success"]), r["time_s"], r["path_length_m"],
                    r["nodes"], r["points"], r["smoothness_deg"], r.get("message", "")
                ])
        self.status_var.set(f"Planner summary saved: {Path(path).name}")

    def pp_save_selected_path_csv(self):
        name = self.pp_selected_path_name or self.pp_algo_var.get()
        if not name or name not in self.pp_results:
            messagebox.showinfo("Path Planning", "Run a planner first.")
            return
        r = self.pp_results[name]
        if not r["success"] or not r["path"]:
            messagebox.showinfo("Path Planning", f"{name} does not currently have a successful path.")
            return
        path = filedialog.asksaveasfilename(
            title=f"Save {name} Path",
            initialfile=f"{name.replace('*','star')}_path.csv",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["index", "x_m", "y_m"])
            for i, (x, y) in enumerate(r["path"]):
                w.writerow([i, x, y])
        self.status_var.set(f"Selected planner path saved: {Path(path).name}")

    # ------------------------------------------------------------------
    # Extend inherited analysis export
    # ------------------------------------------------------------------
    def _refresh_all_analysis_graphs(self):
        super()._refresh_all_analysis_graphs()
        self.pp_update_figure()

    def export_all_analysis_figures(self):
        folder = filedialog.askdirectory(title="Select Folder for Analysis Figures")
        if not folder:
            return
        self._refresh_all_analysis_graphs()
        figs = [
            ("Fig01_Workspace", getattr(self, "workspace_fig", None)),
            ("Fig02_Trajectory", getattr(self, "path_fig", None)),
            ("Fig03_Trajectory_Profiles", getattr(self, "planning_fig", None)),
            ("Fig04_Live_Singularity", getattr(self, "singularity_fig", None)),
            ("Fig05_Gait_Analysis", getattr(self, "gait_fig", None)),
            ("Fig06_ML_Algorithms", getattr(self, "score_fig", None)),
            ("Fig07_Error_Analysis", getattr(self, "error_fig", None)),
            ("Fig08_Selected_Joint_Kinematics", getattr(self, "joint_kinematics_fig", None)),
            ("Fig09_Path_Planning_Algorithms", getattr(self, "pp_fig", None)),
        ]
        out = Path(folder)
        count = 0
        for name, fig in figs:
            if fig is None:
                continue
            fig.savefig(out / f"{name}.png", dpi=600, bbox_inches="tight")
            fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
            count += 1
        self.status_var.set(f"Exported {count} Analysis figures as 600-dpi PNG + PDF.")
        messagebox.showinfo(
            "Analysis Figures",
            f"Exported {count} figure sets (PNG + PDF) to:\n{folder}"
        )


# ===========================================================================
# V2.0 — CLARIFIED ML CONTROLLER / PATH-PLANNER ARCHITECTURE
# ===========================================================================

def point_to_polyline_distance(point, polyline):
    """Minimum Euclidean distance from a 2D point to a polyline."""
    if not polyline:
        return float("nan")
    if len(polyline) == 1:
        return euclid(point, polyline[0])

    p = np.asarray(point, dtype=float)
    best = float("inf")
    for i in range(len(polyline)-1):
        a = np.asarray(polyline[i], dtype=float)
        b = np.asarray(polyline[i+1], dtype=float)
        ab = b - a
        den = float(np.dot(ab, ab))
        if den < 1e-15:
            d = float(np.linalg.norm(p-a))
        else:
            u = float(np.dot(p-a, ab) / den)
            u = min(max(u, 0.0), 1.0)
            proj = a + u*ab
            d = float(np.linalg.norm(p-proj))
        best = min(best, d)
    return best


def polyline_tracking_rmse(executed_path, reference_path):
    if not executed_path or not reference_path:
        return float("nan")
    d = np.asarray([
        point_to_polyline_distance(p, reference_path)
        for p in executed_path
    ], dtype=float)
    if not len(d):
        return float("nan")
    return float(np.sqrt(np.mean(d*d)))


def polyline_tracking_mae(executed_path, reference_path):
    if not executed_path or not reference_path:
        return float("nan")
    d = np.asarray([
        point_to_polyline_distance(p, reference_path)
        for p in executed_path
    ], dtype=float)
    if not len(d):
        return float("nan")
    return float(np.mean(np.abs(d)))


def path_collision_free(path, obstacles, clearance=0.0):
    if not path or len(path) < 2:
        return True
    return all(
        segment_collision_free(
            path[i], path[i+1],
            obstacles, clearance
        )
        for i in range(len(path)-1)
    )


class MLRobotDogAppV20(MLRobotDogAppV19):
    """
    V2.0 clarifies the hierarchy:

        PATH PLANNER
        Dijkstra / A* / RRT / RRT*
                ↓
        Desired collision-free HEAD route
                ↓
        ML CONTROLLER
        Random Forest / Extra Trees / KNN / MLP
                ↓
        Turn / Forward gait commands
                ↓
        12-DOF simulated execution

    The ML algorithms are controller models, not classical obstacle-aware
    path planners. The GUI now names both algorithm layers explicitly.
    """

    def __init__(self):
        # These vars are created lazily in _build_control_tab because Tk()
        # is initialized inside the inherited constructor.
        self.comparison_matrix = {}
        self.comparison_selected = {}
        super().__init__()

        self.title(
            "ROBOQUAD FK V2.0 — ML Controller + Classical Path Planner Comparison"
        )

        # Horizontal comparison tab immediately after Path Planning.
        self.compare_tab = ttk.Frame(self.main_nb)
        try:
            self.main_nb.insert(7, self.compare_tab, text="ML vs Planners")
        except Exception:
            self.main_nb.add(self.compare_tab, text="ML vs Planners")
        self._build_ml_planner_comparison_tab(self.compare_tab)

        self._refresh_algorithm_identity()
        self.status_var.set(
            "V2.0 ready: choose an ML controller and a path planner independently."
        )

    # ------------------------------------------------------------------
    # GUI clarification / ML controller selection
    # ------------------------------------------------------------------
    def _ensure_v20_vars(self):
        if not hasattr(self, "ml_controller_choice_var"):
            self.ml_controller_choice_var = tk.StringVar(value="Auto Best")
            self.active_ml_algorithm_var = tk.StringVar(
                value="Active ML controller: —"
            )
            self.route_source_var = tk.StringVar(
                value="Route source: Manual HEAD waypoints"
            )
            self.execution_stack_var = tk.StringVar(
                value="Execution stack: Route → ML Controller → 12-DOF gait"
            )
            self.desired_actual_var = tk.StringVar(
                value="Desired vs actual: run a simulation to compute tracking metrics."
            )
            self.comp_ml_var = tk.StringVar(value="Extra Trees")
            self.comp_planner_var = tk.StringVar(value="A*")
            self.comp_status_var = tk.StringVar(
                value="Comparison: train ML models and run planner comparison."
            )

    def _build_control_tab(self, parent):
        self._ensure_v20_vars()
        MLRobotDogApp._build_control_tab(self, parent)

        # Insert explicit ML-controller selector near the top of ML Control.
        try:
            scroll = parent.winfo_children()[0]
            root = scroll.inner
            children = root.winfo_children()
            cols = children[1] if len(children) > 1 else None

            selector = ttk.LabelFrame(
                root,
                text="ML Controller Algorithm — separate from Path Planning",
                padding=8
            )
            if cols is not None:
                selector.pack(
                    fill="x", padx=8, pady=4,
                    before=cols
                )
            else:
                selector.pack(fill="x", padx=8, pady=4)

            ttk.Label(
                selector,
                text="Controller"
            ).grid(row=0, column=0, sticky="w", padx=(0, 4))

            self.ml_controller_combo = ttk.Combobox(
                selector,
                textvariable=self.ml_controller_choice_var,
                state="readonly",
                values=(
                    "Auto Best",
                    "Random Forest",
                    "Extra Trees",
                    "KNN",
                    "MLP",
                ),
                width=20
            )
            self.ml_controller_combo.grid(
                row=0, column=1, sticky="w", padx=4
            )
            self.ml_controller_combo.bind(
                "<<ComboboxSelected>>",
                lambda _e: self._refresh_algorithm_identity()
            )

            ttk.Label(
                selector,
                textvariable=self.active_ml_algorithm_var,
                style="Sub.TLabel"
            ).grid(row=0, column=2, sticky="w", padx=12)

            ttk.Label(
                selector,
                text=(
                    "This ML algorithm predicts travel/yaw/q2/q3 gait-control commands. "
                    "Dijkstra/A*/RRT/RRT* generate the route in the Path Planning tab."
                ),
                wraplength=700,
                justify="left"
            ).grid(
                row=1, column=0, columnspan=3,
                sticky="w", pady=(5, 0)
            )
            selector.columnconfigure(2, weight=1)

            # Clarify old button labels / group name.
            self._rename_widget_text(
                root,
                {
                    "3. Path Planning / Simulation":
                        "3. ML Controller / Gait Simulation",
                    "PLAN ML PATH":
                        "GENERATE ML CONTROL PLAN",
                    "▶ PLAN + SIMULATE":
                        "▶ GENERATE + SIMULATE",
                }
            )
        except Exception:
            pass

    def _build_simulation_tab(self, parent):
        self._ensure_v20_vars()
        MLRobotDogApp._build_simulation_tab(self, parent)

        # Add a visible algorithm-identity banner above live simulation.
        children = parent.winfo_children()
        before_widget = children[0] if children else None

        ident = ttk.LabelFrame(
            parent,
            text="Simulation Algorithm Stack",
            padding=7
        )
        if before_widget is not None:
            ident.pack(
                fill="x", padx=6, pady=(6, 2),
                before=before_widget
            )
        else:
            ident.pack(fill="x", padx=6, pady=(6, 2))

        ttk.Label(
            ident, textvariable=self.route_source_var,
            style="Sub.TLabel"
        ).grid(row=0, column=0, sticky="w", padx=5)

        ttk.Label(
            ident, textvariable=self.active_ml_algorithm_var,
            style="Sub.TLabel"
        ).grid(row=0, column=1, sticky="w", padx=12)

        ttk.Label(
            ident, textvariable=self.execution_stack_var
        ).grid(
            row=1, column=0, columnspan=2,
            sticky="w", padx=5, pady=(3, 0)
        )

        ttk.Label(
            ident, textvariable=self.desired_actual_var,
            wraplength=1250, justify="left"
        ).grid(
            row=2, column=0, columnspan=2,
            sticky="w", padx=5, pady=(3, 0)
        )
        ident.columnconfigure(0, weight=1)
        ident.columnconfigure(1, weight=1)

        self._rename_widget_text(
            parent,
            {
                "Live ML Path-Following Simulation":
                    "Live Route Execution — Desired Route + ML Controller",
                "▶ Plan + Simulate":
                    "▶ Generate ML Commands + Simulate",
            }
        )

    def _rename_widget_text(self, widget, mapping):
        """Recursively rename buttons / label frames for clearer terminology."""
        try:
            current = widget.cget("text")
            if current in mapping:
                widget.configure(text=mapping[current])
        except Exception:
            pass

        for child in widget.winfo_children():
            self._rename_widget_text(child, mapping)

    # ------------------------------------------------------------------
    # ML controller identity
    # ------------------------------------------------------------------
    def _active_ml_model(self):
        choice = (
            self.ml_controller_choice_var.get()
            if hasattr(self, "ml_controller_choice_var")
            else "Auto Best"
        )

        if choice != "Auto Best" and choice in self.models:
            return self.models[choice], choice

        if self.best_model is not None:
            return self.best_model, (
                self.best_model_name or "Loaded / Best Model"
            )

        return None, "—"

    def _refresh_algorithm_identity(self):
        self._ensure_v20_vars()
        _model, name = self._active_ml_model()
        choice = self.ml_controller_choice_var.get()

        if choice == "Auto Best":
            self.active_ml_algorithm_var.set(
                f"Active ML controller: Auto Best → {name}"
            )
        else:
            self.active_ml_algorithm_var.set(
                f"Active ML controller: {name}"
            )

        self.execution_stack_var.set(
            f"Execution stack: {self.route_source_var.get().replace('Route source: ', '')}"
            f" → ML Controller [{name}] → Turn/Forward gait → 12-DOF FK simulation"
        )

    def train_models(self):
        super().train_models()
        self._refresh_algorithm_identity()

    def load_model(self):
        super().load_model()
        self.ml_controller_choice_var.set("Auto Best")
        self._refresh_algorithm_identity()

    # ------------------------------------------------------------------
    # Manual route source tracking
    # ------------------------------------------------------------------
    def _mark_manual_route(self):
        self.route_source_var.set(
            "Route source: Manual HEAD waypoints"
        )
        self._refresh_algorithm_identity()

    def add_waypoint(self):
        super().add_waypoint()
        self._mark_manual_route()

    def remove_waypoint(self):
        super().remove_waypoint()
        self._mark_manual_route()

    def clear_waypoints(self):
        super().clear_waypoints()
        self._mark_manual_route()

    def load_forward_example(self):
        super().load_forward_example()
        self._mark_manual_route()

    def load_fb_example(self):
        super().load_fb_example()
        self._mark_manual_route()

    # ------------------------------------------------------------------
    # Classical planners use the same HEAD/Nose reference used by ML.
    # ------------------------------------------------------------------
    def _planner_start_goal(self):
        bx = float(self.start_x_var.get())
        by = float(self.start_y_var.get())
        yaw = math.radians(
            float(self.start_yaw_var.get())
        )
        hx, hy = head_xy_from_body(
            bx, by, yaw, self._head_offset()
        )
        goal = (
            float(self.pp_goal_x_var.get()),
            float(self.pp_goal_y_var.get())
        )
        return (hx, hy), goal

    def pp_use_selected_path_as_ml_waypoints(self):
        name = (
            self.pp_selected_path_name
            or self.pp_algo_var.get()
        )
        super().pp_use_selected_path_as_ml_waypoints()

        if (
            name
            and name in self.pp_results
            and self.pp_results[name]["success"]
        ):
            self.route_source_var.set(
                f"Route source: {name} path planner"
            )
            self._refresh_algorithm_identity()

    def pp_use_best_path_as_ml_waypoints(self):
        successful = [
            r for r in self.pp_results.values()
            if r["success"]
        ]
        if not successful:
            messagebox.showinfo(
                "Path Planning",
                "No successful planner path is available."
            )
            return

        best = min(
            successful,
            key=lambda r: (
                r["path_length_m"],
                r["time_s"]
            )
        )
        self.pp_selected_path_name = best["name"]
        self.pp_algo_var.set(best["name"])
        self.pp_use_selected_path_as_ml_waypoints()

    # ------------------------------------------------------------------
    # ML planning is controller-command generation, not obstacle planning.
    # ------------------------------------------------------------------
    def plan_path(self, simulate=False):
        model, model_name = self._active_ml_model()

        if model is None:
            messagebox.showinfo(
                "ML Controller",
                "Train the ML algorithms or load a model first."
            )
            return

        if not self.waypoints:
            messagebox.showerror(
                "ML Controller",
                "Define or transfer at least one HEAD waypoint."
            )
            return

        try:
            start = (
                float(self.start_x_var.get()),
                float(self.start_y_var.get()),
                math.radians(
                    float(self.start_yaw_var.get())
                ),
            )

            allow_reverse = (
                self.path_policy_var.get()
                == "Allow Automatic Reverse"
            )

            self.plan = plan_ml_waypoints(
                model,
                self.waypoints,
                start,
                self.max_step_var.get(),
                self.max_yaw_var.get(),
                self.tolerance_var.get(),
                self.max_commands_var.get(),
                head_offset=self._head_offset(),
                turn_threshold_deg=self.turn_threshold_var.get(),
                allow_reverse=allow_reverse
            )

            cmds = self.plan["commands"]
            nf = sum(
                c["direction"] == "Forward"
                for c in cmds
            )
            nb = sum(
                c["direction"] == "Backward"
                for c in cmds
            )
            nt = sum(
                c["direction"].startswith("Turn")
                for c in cmds
            )

            self.path_var.set(
                f"ML controller plan [{model_name}] | "
                f"Route={self.route_source_var.get().replace('Route source: ', '')} | "
                f"{'ALL WAYPOINTS REACHED' if self.plan['reached'] else 'MAX COMMANDS REACHED'} | "
                f"turn={nt}, forward={nf}, backward={nb} | "
                f"HEAD final error={self.plan['final_error_m']:.4f} m"
            )

            self.active_ml_algorithm_var.set(
                f"Active ML controller: {model_name}"
            )
            self._refresh_algorithm_identity()

            self.status_var.set(
                "ML gait-control command sequence generated. "
                "Classical route generation, if used, came from the Path Planning tab."
            )

            self._update_path_plot()
            self._update_3d()
            self.update_planning_plot()

            if simulate:
                self.start_sim()

        except Exception as exc:
            messagebox.showerror(
                "ML Controller Planning",
                str(exc)
            )

    # ------------------------------------------------------------------
    # Desired vs actual metrics
    # ------------------------------------------------------------------
    def _finish_sim(self):
        super()._finish_sim()
        self._update_desired_actual_summary()

    def _update_desired_actual_summary(self):
        if not self.waypoints:
            self.desired_actual_var.set(
                "Desired vs actual: no reference route."
            )
            return

        if len(self.head_trail) < 2:
            self.desired_actual_var.set(
                "Desired vs actual: simulation has not produced an executed HEAD path yet."
            )
            return

        desired = list(self.waypoints)
        actual = list(self.head_trail)
        planned = (
            list(self.plan.get("head_path", []))
            if self.plan else []
        )

        rmse = polyline_tracking_rmse(
            actual, desired
        )
        mae = polyline_tracking_mae(
            actual, desired
        )
        final_error = euclid(
            actual[-1], desired[-1]
        )

        desired_L = path_length_2d(desired)
        actual_L = path_length_2d(actual)
        planned_L = path_length_2d(planned)

        self.desired_actual_var.set(
            f"Desired vs actual HEAD path | "
            f"Desired L={desired_L:.4f} m | "
            f"ML-planned L={planned_L:.4f} m | "
            f"Actual L={actual_L:.4f} m | "
            f"MAE={mae:.4f} m | RMSE={rmse:.4f} m | "
            f"Final error={final_error:.4f} m"
        )

    # ------------------------------------------------------------------
    # Simulation plot title now identifies BOTH algorithms.
    # ------------------------------------------------------------------
    def _update_3d(self):
        super()._update_3d()
        try:
            _m, mname = self._active_ml_model()
            route = self.route_source_var.get().replace(
                "Route source: ", ""
            )
            hx, hy = self._head_xy()
            self.ax3d.set_title(
                f"Route: {route} | ML Controller: {mname}\n"
                f"HEAD=({hx:.3f}, {hy:.3f}) m | "
                f"Yaw={math.degrees(self.base_yaw):.1f}°"
            )
            self.fig3d.tight_layout()
            self.canvas3d.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # ML × Path-Planner comparison tab
    # ------------------------------------------------------------------
    def _build_ml_planner_comparison_tab(self, parent):
        self._ensure_v20_vars()

        outer = ttk.Frame(parent, padding=8)
        outer.pack(fill="both", expand=True)

        top = ttk.Frame(outer)
        top.pack(fill="x")

        ttk.Label(
            top,
            text="ML Controller × Path-Planning Algorithm Comparison",
            style="Header.TLabel"
        ).pack(side="left")

        ttk.Label(
            top,
            text="ML Controller"
        ).pack(side="left", padx=(18, 3))
        ttk.Combobox(
            top,
            textvariable=self.comp_ml_var,
            state="readonly",
            values=(
                "Random Forest",
                "Extra Trees",
                "KNN",
                "MLP",
            ),
            width=17
        ).pack(side="left", padx=2)

        ttk.Label(
            top,
            text="Planner"
        ).pack(side="left", padx=(12, 3))
        ttk.Combobox(
            top,
            textvariable=self.comp_planner_var,
            state="readonly",
            values=(
                "Dijkstra",
                "A*",
                "RRT",
                "RRT*",
            ),
            width=12
        ).pack(side="left", padx=2)

        ttk.Button(
            top,
            text="EVALUATE 4 ML × 4 PLANNERS",
            command=self.evaluate_ml_planner_matrix
        ).pack(side="right", padx=2)

        ttk.Button(
            top,
            text="Save Figure",
            command=lambda: self.save_figure_dialog(
                self.comp_fig,
                "Fig10_ML_vs_Path_Planners"
            )
        ).pack(side="right", padx=2)

        ttk.Button(
            top,
            text="Save Comparison CSV",
            command=self.save_ml_planner_comparison_csv
        ).pack(side="right", padx=2)

        expl = ttk.LabelFrame(
            outer,
            text="What is being compared?",
            padding=7
        )
        expl.pack(fill="x", pady=(6, 4))

        ttk.Label(
            expl,
            text=(
                "Dijkstra / A* / RRT / RRT* generate collision-free HEAD routes. "
                "Random Forest / Extra Trees / KNN / MLP then act as gait controllers that "
                "follow those routes. The 4×4 matrix therefore compares complete "
                "'Planner + ML Controller' stacks under the same robot geometry, start, goal, "
                "obstacles, gait parameters and HEAD-tracking definition."
            ),
            wraplength=1500,
            justify="left"
        ).pack(anchor="w")

        ttk.Label(
            expl,
            textvariable=self.comp_status_var,
            style="Sub.TLabel",
            wraplength=1500,
            justify="left"
        ).pack(anchor="w", pady=(4, 0))

        # Result table for selected controller across all planners.
        cols = (
            "stack", "routeL", "execL", "rmse",
            "final", "commands", "ctrltime", "collision"
        )
        self.comp_tree = ttk.Treeview(
            outer, columns=cols,
            show="headings", height=5
        )

        heads = {
            "stack": "Planner + ML Controller",
            "routeL": "Planner L (m)",
            "execL": "ML-followed L (m)",
            "rmse": "Tracking RMSE (m)",
            "final": "Final Error (m)",
            "commands": "ML Commands",
            "ctrltime": "Controller Time (s)",
            "collision": "Collision-Free",
        }
        widths = {
            "stack": 260,
            "routeL": 100,
            "execL": 120,
            "rmse": 125,
            "final": 115,
            "commands": 100,
            "ctrltime": 120,
            "collision": 110,
        }

        for c in cols:
            self.comp_tree.heading(
                c, text=heads[c]
            )
            self.comp_tree.column(
                c, width=widths[c],
                anchor="center"
            )
        self.comp_tree.pack(
            fill="x", pady=(3, 6)
        )

        self.comp_fig = Figure(
            figsize=(13.2, 7.5), dpi=100
        )
        self.comp_axes = [
            self.comp_fig.add_subplot(231),
            self.comp_fig.add_subplot(232),
            self.comp_fig.add_subplot(233),
            self.comp_fig.add_subplot(234),
            self.comp_fig.add_subplot(235),
            self.comp_fig.add_subplot(236),
        ]

        self.comp_canvas = FigureCanvasTkAgg(
            self.comp_fig, master=outer
        )
        self.comp_canvas.get_tk_widget().pack(
            fill="both", expand=True
        )

        self.update_ml_planner_comparison_figure()

    def _evaluate_controller_on_planner(
        self, model_name, model, planner_name, planner_result
    ):
        if not planner_result["success"]:
            return {
                "ml": model_name,
                "planner": planner_name,
                "success": False,
                "planner_length_m": float("nan"),
                "execution_length_m": float("nan"),
                "tracking_rmse_m": float("nan"),
                "tracking_mae_m": float("nan"),
                "final_error_m": float("nan"),
                "commands": 0,
                "controller_time_s": float("nan"),
                "collision_free": False,
                "head_path": [],
            }

        route = list(planner_result["path"])

        start = (
            float(self.start_x_var.get()),
            float(self.start_y_var.get()),
            math.radians(
                float(self.start_yaw_var.get())
            ),
        )

        allow_reverse = (
            self.path_policy_var.get()
            == "Allow Automatic Reverse"
        )

        t0 = time.perf_counter()
        ctrl = plan_ml_waypoints(
            model,
            route,
            start,
            self.max_step_var.get(),
            self.max_yaw_var.get(),
            self.tolerance_var.get(),
            self.max_commands_var.get(),
            head_offset=self._head_offset(),
            turn_threshold_deg=self.turn_threshold_var.get(),
            allow_reverse=allow_reverse
        )
        ctrl_time = time.perf_counter() - t0

        head_path = list(
            ctrl.get("head_path", [])
        )

        rmse = polyline_tracking_rmse(
            head_path, route
        )
        mae = polyline_tracking_mae(
            head_path, route
        )

        collision_free = path_collision_free(
            head_path,
            self.pp_obstacles,
            self.pp_clearance_var.get()
        )

        return {
            "ml": model_name,
            "planner": planner_name,
            "success": bool(ctrl.get("reached", False)),
            "planner_length_m": path_length_2d(route),
            "execution_length_m": path_length_2d(head_path),
            "tracking_rmse_m": rmse,
            "tracking_mae_m": mae,
            "final_error_m": float(
                ctrl.get(
                    "final_error_m",
                    float("nan")
                )
            ),
            "commands": len(
                ctrl.get("commands", [])
            ),
            "controller_time_s": ctrl_time,
            "collision_free": bool(
                collision_free
            ),
            "head_path": head_path,
            "route": route,
        }

    def evaluate_ml_planner_matrix(self):
        if not self.models:
            messagebox.showinfo(
                "ML vs Planners",
                "Train the four ML algorithms first."
            )
            return

        # Guarantee all four planner results exist under the same conditions.
        self.status_var.set(
            "Evaluating 4 ML controllers × 4 path planners..."
        )
        self.update_idletasks()

        planners = (
            "Dijkstra", "A*", "RRT", "RRT*"
        )
        self.pp_results = {
            name: self._run_one_planner(name)
            for name in planners
        }
        self.pp_refresh_results_tree()
        self.pp_update_figure()

        self.comparison_matrix = {}

        for ml_name in (
            "Random Forest",
            "Extra Trees",
            "KNN",
            "MLP",
        ):
            if ml_name not in self.models:
                continue

            for planner_name in planners:
                key = (
                    ml_name,
                    planner_name
                )
                self.comparison_matrix[key] = (
                    self._evaluate_controller_on_planner(
                        ml_name,
                        self.models[ml_name],
                        planner_name,
                        self.pp_results[
                            planner_name
                        ]
                    )
                )

        self._refresh_comparison_table()
        self.update_ml_planner_comparison_figure()

        finite = [
            r for r in self.comparison_matrix.values()
            if r["success"]
            and math.isfinite(
                r["tracking_rmse_m"]
            )
            and r["collision_free"]
        ]

        if finite:
            best = min(
                finite,
                key=lambda r: (
                    r["tracking_rmse_m"],
                    r["final_error_m"],
                    r["planner_length_m"]
                )
            )
            self.comp_status_var.set(
                f"Best collision-free stack by tracking RMSE: "
                f"{best['planner']} + {best['ml']} | "
                f"RMSE={best['tracking_rmse_m']:.4f} m | "
                f"Final error={best['final_error_m']:.4f} m | "
                f"Planner length={best['planner_length_m']:.4f} m"
            )
        else:
            self.comp_status_var.set(
                "Comparison complete, but no stack satisfied all success/collision-free criteria."
            )

        self.status_var.set(
            "4×4 ML-controller / path-planner comparison complete."
        )

    def _refresh_comparison_table(self):
        if not hasattr(self, "comp_tree"):
            return

        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)

        ml = self.comp_ml_var.get()

        for planner in (
            "Dijkstra", "A*", "RRT", "RRT*"
        ):
            r = self.comparison_matrix.get(
                (ml, planner)
            )
            if not r:
                continue

            self.comp_tree.insert(
                "", "end",
                values=(
                    f"{planner} + {ml}",
                    (
                        f"{r['planner_length_m']:.4f}"
                        if math.isfinite(
                            r["planner_length_m"]
                        ) else "—"
                    ),
                    (
                        f"{r['execution_length_m']:.4f}"
                        if math.isfinite(
                            r["execution_length_m"]
                        ) else "—"
                    ),
                    (
                        f"{r['tracking_rmse_m']:.5f}"
                        if math.isfinite(
                            r["tracking_rmse_m"]
                        ) else "—"
                    ),
                    (
                        f"{r['final_error_m']:.5f}"
                        if math.isfinite(
                            r["final_error_m"]
                        ) else "—"
                    ),
                    r["commands"],
                    (
                        f"{r['controller_time_s']:.5f}"
                        if math.isfinite(
                            r["controller_time_s"]
                        ) else "—"
                    ),
                    "Yes" if r["collision_free"]
                    else "No",
                )
            )

    def update_ml_planner_comparison_figure(self):
        if not hasattr(self, "comp_axes"):
            return

        axes = self.comp_axes
        for ax in axes:
            ax.clear()

        ml_names = (
            "Random Forest",
            "Extra Trees",
            "KNN",
            "MLP",
        )
        planners = (
            "Dijkstra", "A*", "RRT", "RRT*"
        )

        # (a) selected planner + selected ML overlay
        ax = axes[0]
        start, goal = self._planner_start_goal()

        for ox, oy, rr in getattr(
            self, "pp_obstacles", []
        ):
            ax.add_patch(
                Circle(
                    (ox, oy), rr,
                    fill=False, linewidth=1.3
                )
            )

        ax.scatter(
            [start[0]], [start[1]],
            marker="s", s=55,
            label="HEAD start"
        )
        ax.scatter(
            [goal[0]], [goal[1]],
            marker="*", s=95,
            label="Goal"
        )

        selected = self.comparison_matrix.get(
            (
                self.comp_ml_var.get(),
                self.comp_planner_var.get()
            )
        )

        if selected:
            if selected.get("route"):
                P = np.asarray(
                    selected["route"],
                    dtype=float
                )
                ax.plot(
                    P[:,0], P[:,1],
                    linewidth=2.0,
                    label=(
                        f"{selected['planner']} route"
                    )
                )
            if selected.get("head_path"):
                Q = np.asarray(
                    selected["head_path"],
                    dtype=float
                )
                ax.plot(
                    Q[:,0], Q[:,1],
                    linestyle="--",
                    linewidth=2.0,
                    label=(
                        f"{selected['ml']} followed path"
                    )
                )

        if len(self.head_trail) >= 2:
            H = np.asarray(
                self.head_trail,
                dtype=float
            )
            ax.plot(
                H[:,0], H[:,1],
                linewidth=1.8,
                label="Actual simulated HEAD path"
            )

        ax.set_title(
            "(a) Planner Route vs ML-Followed / Actual Path"
        )
        ax.set_xlabel("World X (m)")
        ax.set_ylabel("World Y (m)")
        ax.axis("equal")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)

        if not self.comparison_matrix:
            for i, title in enumerate((
                "(b) Planner vs ML-Followed Path Length",
                "(c) Tracking RMSE Matrix",
                "(d) Final HEAD Error Matrix",
                "(e) ML Controller Computation Time",
                "(f) ML Commands Matrix",
            ), start=1):
                axes[i].text(
                    0.5, 0.5,
                    "Run 4×4 evaluation",
                    ha="center", va="center",
                    transform=axes[i].transAxes
                )
                axes[i].set_title(title)

            self.comp_fig.suptitle(
                "ML Controller × Path Planner Comparison",
                fontsize=11
            )
            self.comp_fig.tight_layout()
            self.comp_canvas.draw_idle()
            return

        # (b) route length vs followed length for selected ML
        selected_ml = self.comp_ml_var.get()
        route_L = []
        exec_L = []

        for p in planners:
            r = self.comparison_matrix.get(
                (selected_ml, p)
            )
            route_L.append(
                r["planner_length_m"]
                if r and math.isfinite(
                    r["planner_length_m"]
                ) else 0.0
            )
            exec_L.append(
                r["execution_length_m"]
                if r and math.isfinite(
                    r["execution_length_m"]
                ) else 0.0
            )

        x = np.arange(len(planners))
        axes[1].bar(
            x-0.18, route_L,
            width=0.36,
            label="Planner route"
        )
        axes[1].bar(
            x+0.18, exec_L,
            width=0.36,
            label="ML-followed"
        )
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(
            planners, rotation=15
        )
        axes[1].set_ylabel("Length (m)")
        axes[1].set_title(
            f"(b) Path Length — {selected_ml}"
        )
        axes[1].legend(fontsize=7)
        axes[1].grid(
            True, axis="y", alpha=0.25
        )

        def matrix(metric, fallback=np.nan):
            M = np.full(
                (len(ml_names), len(planners)),
                fallback, dtype=float
            )
            for i, ml in enumerate(ml_names):
                for j, p in enumerate(planners):
                    r = self.comparison_matrix.get(
                        (ml, p)
                    )
                    if not r:
                        continue
                    val = r.get(metric, fallback)
                    if isinstance(val, bool):
                        val = 1.0 if val else 0.0
                    try:
                        M[i, j] = float(val)
                    except Exception:
                        pass
            return M

        def heatmap(ax, M, title, fmt=".3f"):
            im = ax.imshow(
                M, aspect="auto"
            )
            ax.set_xticks(
                range(len(planners))
            )
            ax.set_xticklabels(
                planners, rotation=15
            )
            ax.set_yticks(
                range(len(ml_names))
            )
            ax.set_yticklabels(
                ml_names
            )
            ax.set_title(title)

            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    v = M[i, j]
                    if math.isfinite(v):
                        ax.text(
                            j, i,
                            format(v, fmt),
                            ha="center",
                            va="center",
                            fontsize=7
                        )
            self.comp_fig.colorbar(
                im, ax=ax,
                fraction=0.046,
                pad=0.04
            )

        heatmap(
            axes[2],
            matrix("tracking_rmse_m"),
            "(c) Tracking RMSE (m)",
            ".4f"
        )

        heatmap(
            axes[3],
            matrix("final_error_m"),
            "(d) Final HEAD Error (m)",
            ".4f"
        )

        heatmap(
            axes[4],
            matrix("controller_time_s"),
            "(e) ML Controller Computation Time (s)",
            ".4f"
        )

        heatmap(
            axes[5],
            matrix("commands"),
            "(f) ML Command Count",
            ".0f"
        )

        self.comp_fig.suptitle(
            "4 ML Controllers × 4 Classical Path Planners",
            fontsize=11
        )
        self.comp_fig.tight_layout()
        self.comp_canvas.draw_idle()

        self._refresh_comparison_table()

    def save_ml_planner_comparison_csv(self):
        if not self.comparison_matrix:
            messagebox.showinfo(
                "Comparison",
                "Run the 4×4 evaluation first."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save ML vs Planner Comparison",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")]
        )
        if not path:
            return

        with open(
            path, "w",
            newline="",
            encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow([
                "ml_controller",
                "path_planner",
                "success",
                "planner_length_m",
                "ml_followed_length_m",
                "tracking_mae_m",
                "tracking_rmse_m",
                "final_head_error_m",
                "ml_commands",
                "controller_compute_time_s",
                "collision_free",
            ])

            for ml in (
                "Random Forest",
                "Extra Trees",
                "KNN",
                "MLP",
            ):
                for p in (
                    "Dijkstra",
                    "A*",
                    "RRT",
                    "RRT*",
                ):
                    r = self.comparison_matrix.get(
                        (ml, p)
                    )
                    if not r:
                        continue
                    w.writerow([
                        ml, p,
                        int(r["success"]),
                        r["planner_length_m"],
                        r["execution_length_m"],
                        r["tracking_mae_m"],
                        r["tracking_rmse_m"],
                        r["final_error_m"],
                        r["commands"],
                        r["controller_time_s"],
                        int(r["collision_free"]),
                    ])

        self.status_var.set(
            f"ML vs planner comparison saved: {Path(path).name}"
        )

    # ------------------------------------------------------------------
    # analysis export now includes the comparison figure
    # ------------------------------------------------------------------
    def _refresh_all_analysis_graphs(self):
        super()._refresh_all_analysis_graphs()
        self.update_ml_planner_comparison_figure()

    def export_all_analysis_figures(self):
        folder = filedialog.askdirectory(
            title="Select Folder for Analysis Figures"
        )
        if not folder:
            return

        self._refresh_all_analysis_graphs()

        figs = [
            (
                "Fig01_Workspace",
                getattr(
                    self, "workspace_fig", None
                )
            ),
            (
                "Fig02_Trajectory_Desired_Planned_Actual",
                getattr(
                    self, "path_fig", None
                )
            ),
            (
                "Fig03_Trajectory_Profiles",
                getattr(
                    self, "planning_fig", None
                )
            ),
            (
                "Fig04_Live_Singularity",
                getattr(
                    self, "singularity_fig", None
                )
            ),
            (
                "Fig05_Gait_Analysis",
                getattr(
                    self, "gait_fig", None
                )
            ),
            (
                "Fig06_ML_Algorithms",
                getattr(
                    self, "score_fig", None
                )
            ),
            (
                "Fig07_Error_Analysis",
                getattr(
                    self, "error_fig", None
                )
            ),
            (
                "Fig08_Selected_Joint_Kinematics",
                getattr(
                    self,
                    "joint_kinematics_fig",
                    None
                )
            ),
            (
                "Fig09_Path_Planning_Algorithms",
                getattr(
                    self, "pp_fig", None
                )
            ),
            (
                "Fig10_ML_vs_Path_Planners",
                getattr(
                    self, "comp_fig", None
                )
            ),
        ]

        out = Path(folder)
        count = 0

        for name, fig in figs:
            if fig is None:
                continue
            fig.savefig(
                out / f"{name}.png",
                dpi=600,
                bbox_inches="tight"
            )
            fig.savefig(
                out / f"{name}.pdf",
                bbox_inches="tight"
            )
            count += 1

        self.status_var.set(
            f"Exported {count} Analysis figures as 600-dpi PNG + PDF."
        )
        messagebox.showinfo(
            "Analysis Figures",
            f"Exported {count} figure sets (PNG + PDF) to:\n{folder}"
        )


# ===========================================================================
# V2.1 — ROBUST PLANNER IDENTITY + RRT/RRT* FIXES
# ===========================================================================

def rrt_planner_robust(
    start, goal, obstacles, bounds,
    step_size=0.12,
    max_iter=3000,
    goal_sample_rate=0.15,
    clearance=0.04,
    seed=42,
    rewire=False,
    rewire_radius=0.32
):
    """
    Robust RRT / RRT* implementation.

    Fixes over V2.0:
    1. If start->goal has collision-free line of sight, return it immediately.
    2. Assumes caller has expanded bounds to contain start, goal and obstacles.
    3. RRT* rewiring avoids ancestor cycles.
    4. Goal is treated as a terminal connection, not repeatedly inserted into
       the rewired tree.
    """
    rng = np.random.default_rng(int(seed))
    t0 = time.perf_counter()
    name = "RRT*" if rewire else "RRT"

    start = (float(start[0]), float(start[1]))
    goal = (float(goal[0]), float(goal[1]))
    xmin, xmax, ymin, ymax = map(float, bounds)

    if not point_collision_free(start, obstacles, clearance):
        return {
            "name": name,
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": 0,
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": [],
            "message": "Start HEAD point is inside an obstacle / clearance zone.",
        }

    if not point_collision_free(goal, obstacles, clearance):
        return {
            "name": name,
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": 0,
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": [],
            "message": "Goal is inside an obstacle / clearance zone.",
        }

    # Guaranteed success for a simple unobstructed destination.
    if segment_collision_free(start, goal, obstacles, clearance):
        direct = [start, goal]
        return {
            "name": name,
            "success": True,
            "path": direct,
            "time_s": time.perf_counter() - t0,
            "path_length_m": euclid(start, goal),
            "nodes": 2,
            "points": 2,
            "smoothness_deg": 0.0,
            "history": [euclid(start, goal)],
            "message": "Direct collision-free start-to-goal connection.",
        }

    nodes = [start]
    parents = [-1]
    costs = [0.0]

    step_size = max(0.03, float(step_size))
    max_iter = max(100, int(max_iter))
    goal_sample_rate = min(max(float(goal_sample_rate), 0.01), 0.75)
    rewire_radius = max(step_size * 1.25, float(rewire_radius))

    best_goal_parent = None
    best_goal_cost = float("inf")
    history = []

    def ancestor_set(idx):
        ancestors = set()
        cur = idx
        guard = 0
        while cur != -1 and guard <= len(parents):
            ancestors.add(cur)
            cur = parents[cur]
            guard += 1
        return ancestors

    def update_descendant_costs(root_idx):
        # Recompute descendant costs after a rewire.
        queue = [root_idx]
        visited = set()
        while queue:
            p = queue.pop(0)
            if p in visited:
                continue
            visited.add(p)
            for child, par in enumerate(parents):
                if par != p:
                    continue
                costs[child] = costs[p] + euclid(nodes[p], nodes[child])
                queue.append(child)

    for _it in range(max_iter):
        if rng.random() < goal_sample_rate:
            sample = goal
        else:
            sample = (
                float(rng.uniform(xmin, xmax)),
                float(rng.uniform(ymin, ymax)),
            )

        dists = np.asarray([euclid(n, sample) for n in nodes], dtype=float)
        nearest = int(np.argmin(dists))
        new_pt = _steer(nodes[nearest], sample, step_size)

        if not in_bounds(new_pt, bounds):
            history.append(
                best_goal_cost if math.isfinite(best_goal_cost) else float("nan")
            )
            continue

        if not segment_collision_free(
            nodes[nearest], new_pt, obstacles, clearance
        ):
            history.append(
                best_goal_cost if math.isfinite(best_goal_cost) else float("nan")
            )
            continue

        parent = nearest
        new_cost = costs[nearest] + euclid(nodes[nearest], new_pt)

        near = []
        if rewire:
            near = [
                i for i, n in enumerate(nodes)
                if euclid(n, new_pt) <= rewire_radius
            ]

            # Choose the lowest-cost feasible parent.
            for i in near:
                if segment_collision_free(nodes[i], new_pt, obstacles, clearance):
                    cand = costs[i] + euclid(nodes[i], new_pt)
                    if cand < new_cost:
                        parent = i
                        new_cost = cand

        nodes.append(new_pt)
        parents.append(parent)
        costs.append(new_cost)
        new_idx = len(nodes) - 1

        if rewire and near:
            ancestors_of_new = ancestor_set(new_idx)

            for i in near:
                if i == 0 or i == parent or i in ancestors_of_new:
                    continue

                cand = costs[new_idx] + euclid(nodes[new_idx], nodes[i])
                if cand + 1e-12 >= costs[i]:
                    continue

                if not segment_collision_free(
                    nodes[new_idx], nodes[i], obstacles, clearance
                ):
                    continue

                parents[i] = new_idx
                costs[i] = cand
                update_descendant_costs(i)

        # Check whether this node can connect to goal.
        if (
            euclid(new_pt, goal) <= step_size * 1.35
            and segment_collision_free(new_pt, goal, obstacles, clearance)
        ):
            cand_goal_cost = costs[new_idx] + euclid(new_pt, goal)

            if cand_goal_cost < best_goal_cost:
                best_goal_cost = cand_goal_cost
                best_goal_parent = new_idx

            # Standard RRT stops on first valid goal connection.
            if not rewire:
                history.append(best_goal_cost)
                break

        history.append(
            best_goal_cost if math.isfinite(best_goal_cost) else float("nan")
        )

    if best_goal_parent is None:
        return {
            "name": name,
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": len(nodes),
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": history,
            "message": (
                "No feasible path found within the sampling limit. "
                "Increase iterations / step size or reduce obstacle clearance."
            ),
        }

    path = [goal]
    cur = best_goal_parent
    guard = 0
    while cur != -1 and guard <= len(nodes) + 2:
        path.append(nodes[cur])
        cur = parents[cur]
        guard += 1

    if guard > len(nodes) + 2:
        return {
            "name": name,
            "success": False,
            "path": [],
            "time_s": time.perf_counter() - t0,
            "path_length_m": float("nan"),
            "nodes": len(nodes),
            "points": 0,
            "smoothness_deg": float("nan"),
            "history": history,
            "message": "Tree parent-cycle detected; path reconstruction aborted.",
        }

    path.reverse()
    path[0] = start
    path[-1] = goal
    path = shorten_path(path, obstacles, clearance)

    return {
        "name": name,
        "success": True,
        "path": path,
        "time_s": time.perf_counter() - t0,
        "path_length_m": path_length_2d(path),
        "nodes": len(nodes),
        "points": len(path),
        "smoothness_deg": path_smoothness_deg(path),
        "history": history,
        "message": "OK",
    }


class MLRobotDogAppV21(MLRobotDogAppV20):
    """
    V2.1 resolves two user-facing ambiguities:

    - Path Planner is ALWAYS displayed independently from ML Controller.
    - RRT/RRT* use dynamic search bounds and robust direct-path handling.
    """

    def _ensure_v20_vars(self):
        super()._ensure_v20_vars()

        if not hasattr(self, "current_path_planner_var"):
            self.current_path_planner_var = tk.StringVar(
                value="Path Planner: NONE — Manual HEAD waypoints"
            )

    def __init__(self):
        super().__init__()

        self.title(
            "ROBOQUAD FK V2.1 — Explicit Path Planner + ML Controller / Robust RRT"
        )

        self._refresh_algorithm_identity()
        self.status_var.set(
            "V2.1 ready: Path Planner and ML Controller are independent and explicitly displayed."
        )

    # ------------------------------------------------------------------
    # Simulation identity
    # ------------------------------------------------------------------
    def _build_simulation_tab(self, parent):
        super()._build_simulation_tab(parent)

        # Add a prominent path-planner line to the algorithm stack.
        try:
            identity_frame = None

            def find_frame(w):
                nonlocal identity_frame
                for child in w.winfo_children():
                    try:
                        if (
                            isinstance(child, ttk.LabelFrame)
                            and child.cget("text") == "Simulation Algorithm Stack"
                        ):
                            identity_frame = child
                            return
                    except Exception:
                        pass
                    find_frame(child)

            find_frame(parent)

            if identity_frame is not None:
                label = ttk.Label(
                    identity_frame,
                    textvariable=self.current_path_planner_var,
                    style="Header.TLabel"
                )
                label.grid(
                    row=3, column=0, columnspan=2,
                    sticky="w", padx=5, pady=(5, 0)
                )
        except Exception:
            pass

    def _refresh_algorithm_identity(self):
        self._ensure_v20_vars()
        super()._refresh_algorithm_identity()

        _model, ml_name = self._active_ml_model()
        planner = self.current_path_planner_var.get().replace(
            "Path Planner: ", ""
        )
        route = self.route_source_var.get().replace(
            "Route source: ", ""
        )

        self.execution_stack_var.set(
            f"Execution stack: Path Planner [{planner}] → Route [{route}] → "
            f"ML Controller [{ml_name}] → Turn/Forward gait → 12-DOF FK simulation"
        )

    def _update_3d(self):
        # Let inherited drawing execute first.
        super()._update_3d()

        try:
            _model, ml_name = self._active_ml_model()
            planner = self.current_path_planner_var.get().replace(
                "Path Planner: ", ""
            )
            route = self.route_source_var.get().replace(
                "Route source: ", ""
            )
            hx, hy = self._head_xy()

            self.ax3d.set_title(
                f"Path Planner: {planner} | ML Controller: {ml_name}\n"
                f"Route: {route} | HEAD=({hx:.3f}, {hy:.3f}) m | "
                f"Yaw={math.degrees(self.base_yaw):.1f}°"
            )
            self.fig3d.tight_layout()
            self.canvas3d.draw_idle()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Manual waypoints mean NO path-planning algorithm is active.
    # ------------------------------------------------------------------
    def _mark_manual_route(self):
        super()._mark_manual_route()
        self.current_path_planner_var.set(
            "Path Planner: NONE — Manual HEAD waypoints"
        )
        self._refresh_algorithm_identity()

    # ------------------------------------------------------------------
    # Robust search domain: always contain start HEAD, goal and obstacles.
    # ------------------------------------------------------------------
    def _planner_bounds(self):
        start, goal = self._planner_start_goal()

        try:
            half = max(
                1.0,
                float(self.plane_size_var.get()) / 2.0
            )
        except Exception:
            half = 2.0

        xs = [
            -half, +half,
            float(start[0]), float(goal[0])
        ]
        ys = [
            -half, +half,
            float(start[1]), float(goal[1])
        ]

        for ox, oy, rr in self.pp_obstacles:
            rr_eff = float(rr) + max(
                0.0, float(self.pp_clearance_var.get())
            )
            xs.extend([
                float(ox) - rr_eff,
                float(ox) + rr_eff
            ])
            ys.extend([
                float(oy) - rr_eff,
                float(oy) + rr_eff
            ])

        margin = max(
            0.40,
            3.0 * float(self.pp_step_size_var.get()),
            3.0 * float(self.pp_resolution_var.get())
        )

        xmin = min(xs) - margin
        xmax = max(xs) + margin
        ymin = min(ys) - margin
        ymax = max(ys) + margin

        # Keep a non-degenerate rectangular domain.
        if xmax - xmin < 1.0:
            c = 0.5 * (xmin + xmax)
            xmin, xmax = c - 0.5, c + 0.5

        if ymax - ymin < 1.0:
            c = 0.5 * (ymin + ymax)
            ymin, ymax = c - 0.5, c + 0.5

        return (
            float(xmin), float(xmax),
            float(ymin), float(ymax)
        )

    # ------------------------------------------------------------------
    # Use robust RRT/RRT* while keeping Dijkstra/A* unchanged.
    # ------------------------------------------------------------------
    def _run_one_planner(self, name):
        start, goal = self._planner_start_goal()
        bounds = self._planner_bounds()
        clearance = float(
            self.pp_clearance_var.get()
        )

        if name == "Dijkstra":
            return grid_search_planner(
                start, goal,
                self.pp_obstacles,
                bounds,
                resolution=float(
                    self.pp_resolution_var.get()
                ),
                use_astar=False,
                clearance=clearance
            )

        if name == "A*":
            return grid_search_planner(
                start, goal,
                self.pp_obstacles,
                bounds,
                resolution=float(
                    self.pp_resolution_var.get()
                ),
                use_astar=True,
                clearance=clearance
            )

        if name == "RRT":
            return rrt_planner_robust(
                start, goal,
                self.pp_obstacles,
                bounds,
                step_size=float(
                    self.pp_step_size_var.get()
                ),
                max_iter=int(
                    self.pp_max_iter_var.get()
                ),
                goal_sample_rate=float(
                    self.pp_goal_bias_var.get()
                ),
                clearance=clearance,
                seed=int(self.seed_var.get()),
                rewire=False,
                rewire_radius=float(
                    self.pp_rewire_radius_var.get()
                )
            )

        if name == "RRT*":
            return rrt_planner_robust(
                start, goal,
                self.pp_obstacles,
                bounds,
                step_size=float(
                    self.pp_step_size_var.get()
                ),
                max_iter=int(
                    self.pp_max_iter_var.get()
                ),
                goal_sample_rate=float(
                    self.pp_goal_bias_var.get()
                ),
                clearance=clearance,
                seed=int(self.seed_var.get()),
                rewire=True,
                rewire_radius=float(
                    self.pp_rewire_radius_var.get()
                )
            )

        raise ValueError(
            f"Unknown path planner: {name}"
        )

    # ------------------------------------------------------------------
    # Explicitly activate planner only after its path is transferred.
    # ------------------------------------------------------------------
    def pp_use_selected_path_as_ml_waypoints(self):
        name = (
            self.pp_selected_path_name
            or self.pp_algo_var.get()
        )

        # Use V2.0 transfer logic.
        super().pp_use_selected_path_as_ml_waypoints()

        if (
            name
            and name in self.pp_results
            and self.pp_results[name]["success"]
        ):
            self.current_path_planner_var.set(
                f"Path Planner: {name}"
            )
            self.route_source_var.set(
                f"Route source: {name} collision-free HEAD route"
            )
            self._refresh_algorithm_identity()

            self.pp_status_var.set(
                f"{name} is now the ACTIVE path planner for simulation. "
                f"Its route was transferred to the ML controller as HEAD waypoints."
            )

    def pp_use_best_path_as_ml_waypoints(self):
        successful = [
            r for r in self.pp_results.values()
            if r["success"]
        ]

        if not successful:
            messagebox.showinfo(
                "Path Planning",
                "No successful planner path is available."
            )
            return

        best = min(
            successful,
            key=lambda r: (
                r["path_length_m"],
                r["time_s"]
            )
        )

        self.pp_selected_path_name = best["name"]
        self.pp_algo_var.set(best["name"])
        self.pp_use_selected_path_as_ml_waypoints()

    # ------------------------------------------------------------------
    # Convenience button: algorithm -> route -> ML commands -> simulate.
    # ------------------------------------------------------------------
    def _build_path_planning_tab(self, parent):
        super()._build_path_planning_tab(parent)

        try:
            target = None

            def find_actions(w):
                nonlocal target
                for child in w.winfo_children():
                    try:
                        if (
                            isinstance(child, ttk.LabelFrame)
                            and child.cget("text") == "4. Run / Transfer / Export"
                        ):
                            target = child
                            return
                    except Exception:
                        pass
                    find_actions(child)

            find_actions(parent)

            if target is not None:
                ttk.Separator(
                    target, orient="horizontal"
                ).pack(fill="x", pady=6)

                ttk.Button(
                    target,
                    text="PLAN SELECTED → TRANSFER → ML SIMULATE",
                    command=self.pp_plan_transfer_simulate
                ).pack(
                    fill="x", pady=2
                )

                ttk.Label(
                    target,
                    text=(
                        "Path planners are independent of ML. "
                        "This button first generates the classical route, then passes it "
                        "to the selected ML controller for gait execution."
                    ),
                    wraplength=390,
                    justify="left"
                ).pack(
                    fill="x", pady=(3, 0)
                )
        except Exception:
            pass

    def pp_plan_transfer_simulate(self):
        alg = self.pp_algo_var.get()

        self.status_var.set(
            f"Running {alg} path planner..."
        )
        self.update_idletasks()

        result = self._run_one_planner(alg)
        self.pp_results = {
            alg: result
        }
        self.pp_selected_path_name = alg
        self.pp_refresh_results_tree()
        self.pp_update_figure()

        if not result["success"]:
            messagebox.showerror(
                "Path Planning",
                f"{alg} failed:\n\n{result.get('message', 'No feasible path.')}"
            )
            self.pp_status_var.set(
                f"{alg} failed. {result.get('message', '')}"
            )
            return

        self.pp_use_selected_path_as_ml_waypoints()

        # Build ML controller commands and run simulation.
        self.plan_path(
            simulate=True
        )

    # ------------------------------------------------------------------
    # More explicit planner status after planning.
    # ------------------------------------------------------------------
    def pp_plan_selected(self):
        super().pp_plan_selected()

        alg = self.pp_algo_var.get()
        if (
            alg in self.pp_results
            and self.pp_results[alg]["success"]
        ):
            bounds = self._planner_bounds()
            self.pp_status_var.set(
                f"{alg}: SUCCESS | "
                f"path length={self.pp_results[alg]['path_length_m']:.4f} m | "
                f"time={self.pp_results[alg]['time_s']:.4f} s | "
                f"search bounds X=[{bounds[0]:.2f},{bounds[1]:.2f}], "
                f"Y=[{bounds[2]:.2f},{bounds[3]:.2f}]. "
                f"Planner is NOT active in simulation until its route is transferred."
            )

    # ------------------------------------------------------------------
    # Comparison figure title clarifies independent layers.
    # ------------------------------------------------------------------
    def update_ml_planner_comparison_figure(self):
        super().update_ml_planner_comparison_figure()

        try:
            self.comp_fig.suptitle(
                "Independent Classical Route Planners × ML Gait Controllers",
                fontsize=11
            )
            self.comp_fig.tight_layout()
            self.comp_canvas.draw_idle()
        except Exception:
            pass


# ===========================================================================
# V2.2 — SHARED VIA POINTS FOR ML + CLASSICAL PATH PLANNERS
# ===========================================================================

class MLRobotDogAppV22(MLRobotDogAppV21):
    """Use one master via-point sequence for ML and every path planner."""

    def __init__(self):
        self.pp_shared_vias = []
        super().__init__()
        self.title('ROBOQUAD FK V2.2 — Shared Via Points: ML + Dijkstra/A*/RRT/RRT*')
        if not self.pp_shared_vias:
            self.pp_shared_vias = [(float(x), float(y)) for x, y in self.waypoints]
        self.pp_refresh_shared_vias()
        self.status_var.set(
            'V2.2 ready: ML Control via points are the common required via points for all path planners.'
        )

    # ------------------------------------------------------------------
    # Replace the old single-goal box with the shared ML via-point list.
    # ------------------------------------------------------------------
    def _build_path_planning_tab(self, parent):
        if not self.pp_shared_vias and hasattr(self, 'waypoints'):
            self.pp_shared_vias = [(float(x), float(y)) for x, y in self.waypoints]

        super()._build_path_planning_tab(parent)

        self.pp_via_status_var = tk.StringVar(value='Shared via points: synchronized with ML Control.')
        target = None

        def find_start_goal_frame(w):
            nonlocal target
            for child in w.winfo_children():
                try:
                    if isinstance(child, ttk.LabelFrame) and child.cget('text') == '1. Start and Goal':
                        target = child
                        return
                except Exception:
                    pass
                find_start_goal_frame(child)

        find_start_goal_frame(parent)
        if target is None:
            return

        for child in target.winfo_children():
            child.destroy()
        target.configure(text='1. Shared Via Points from ML Control')

        ttk.Label(
            target,
            text=(
                'The path planners use the SAME user-entered via points as ML Control. '
                'No separate Goal X/Y is required. Planning is performed as:\n'
                'HEAD Start → V1 → V2 → V3 → ... → Vn'
            ),
            wraplength=405,
            justify='left'
        ).pack(anchor='w', pady=(0,5))

        self.pp_via_list = tk.Listbox(target, height=8, exportselection=False)
        self.pp_via_list.pack(fill='x')

        ttk.Label(
            target,
            textvariable=self.pp_via_status_var,
            wraplength=405,
            justify='left'
        ).pack(anchor='w', pady=(5,0))

        b = ttk.Frame(target)
        b.pack(fill='x', pady=(6,0))
        ttk.Button(b, text='Refresh from ML Control', command=self.pp_sync_vias_from_ml).pack(side='left', padx=2)
        ttk.Button(b, text='Restore Master Via Points', command=self.pp_restore_master_vias_to_ml).pack(side='left', padx=2)
        ttk.Button(target, text='Open ML Control to Edit Via Points', command=lambda: self.main_nb.select(0)).pack(fill='x', pady=(5,0))

        ttk.Label(
            target,
            text=(
                'The master via-point list is preserved even after a planner inserts intermediate '
                'collision-avoidance points for execution.'
            ),
            wraplength=405,
            justify='left'
        ).pack(anchor='w', pady=(5,0))

        # Rename action buttons so the multi-via behavior is explicit.
        self._rename_widget_text(parent, {
            'PLAN SELECTED ALGORITHM': 'PLAN ALL SHARED VIA POINTS',
            'COMPARE ALL ALGORITHMS': 'COMPARE ALL — SAME VIA POINTS',
            'Use Selected Path as ML Waypoints': 'Use Selected Multi-Via Route for ML Execution',
            'Use Shortest Successful Path as ML Waypoints': 'Use Best Multi-Via Route for ML Execution',
            'PLAN SELECTED → TRANSFER → ML SIMULATE': 'PLAN ALL VIAS → TRANSFER → ML SIMULATE',
        })

        self.pp_refresh_shared_vias()
        self.pp_update_figure()

    # ------------------------------------------------------------------
    # Synchronization / preservation of the user's master via points.
    # ------------------------------------------------------------------
    def pp_refresh_shared_vias(self):
        if not hasattr(self, 'pp_via_list'):
            return
        self.pp_via_list.delete(0, 'end')
        for i, (x, y) in enumerate(self.pp_shared_vias, start=1):
            self.pp_via_list.insert('end', f'V{i}: X={x:+.3f} m, Y={y:+.3f} m')
        if hasattr(self, 'pp_via_status_var'):
            self.pp_via_status_var.set(
                f'Shared via points: {len(self.pp_shared_vias)} | Master sequence from ML Control.'
            )

    def pp_sync_vias_from_ml(self):
        self.pp_shared_vias = [(float(x), float(y)) for x, y in self.waypoints]
        self.pp_refresh_shared_vias()
        self.pp_update_figure()
        self.pp_status_var.set(f'Synchronized {len(self.pp_shared_vias)} via points from ML Control.')

    def pp_restore_master_vias_to_ml(self):
        if not self.pp_shared_vias:
            messagebox.showinfo('Shared Via Points', 'No master via points are stored.')
            return
        self.waypoints = list(self.pp_shared_vias)
        self._refresh_waypoints()
        self.plan = None
        self.body_trail = []
        self.head_trail = []
        self.current_path_planner_var.set('Path Planner: NONE — Master via points restored')
        self.route_source_var.set('Route source: Shared master HEAD via points')
        self._update_path_plot()
        self._refresh_algorithm_identity()
        self.status_var.set('Master via points restored to ML Control.')

    def _mark_manual_route(self):
        super()._mark_manual_route()
        self.pp_shared_vias = [(float(x), float(y)) for x, y in self.waypoints]
        self.pp_refresh_shared_vias()
        if hasattr(self, 'pp_fig'):
            self.pp_update_figure()

    # ------------------------------------------------------------------
    # Shared via-point interpretation.
    # ------------------------------------------------------------------
    def _planner_head_start(self):
        bx = float(self.start_x_var.get())
        by = float(self.start_y_var.get())
        yaw = math.radians(float(self.start_yaw_var.get()))
        return head_xy_from_body(bx, by, yaw, self._head_offset())

    def _effective_shared_vias(self):
        vias = list(self.pp_shared_vias)
        if not vias:
            return []

        # Keep the same convention as the ML controller: if the first point is
        # the BODY start anchor, do not make the HEAD backtrack to it.
        body_start = (float(self.start_x_var.get()), float(self.start_y_var.get()))
        tol = max(0.005, float(self.tolerance_var.get()))
        if euclid(vias[0], body_start) <= tol:
            return vias[1:]
        return vias

    def _planner_start_goal(self):
        start = self._planner_head_start()
        vias = self._effective_shared_vias()
        return start, (vias[-1] if vias else start)

    def _planner_bounds(self):
        start = self._planner_head_start()
        vias = self._effective_shared_vias()
        try:
            half = max(1.0, float(self.plane_size_var.get()) / 2.0)
        except Exception:
            half = 2.0

        xs = [-half, half, float(start[0])]
        ys = [-half, half, float(start[1])]
        for x, y in vias:
            xs.append(float(x)); ys.append(float(y))
        for ox, oy, rr in self.pp_obstacles:
            re = float(rr) + max(0.0, float(self.pp_clearance_var.get()))
            xs.extend([float(ox)-re, float(ox)+re])
            ys.extend([float(oy)-re, float(oy)+re])

        margin = max(
            0.40,
            3.0 * float(self.pp_step_size_var.get()),
            3.0 * float(self.pp_resolution_var.get())
        )
        xmin, xmax = min(xs)-margin, max(xs)+margin
        ymin, ymax = min(ys)-margin, max(ys)+margin
        if xmax-xmin < 1.0:
            c = 0.5*(xmin+xmax); xmin, xmax = c-0.5, c+0.5
        if ymax-ymin < 1.0:
            c = 0.5*(ymin+ymax); ymin, ymax = c-0.5, c+0.5
        return float(xmin), float(xmax), float(ymin), float(ymax)

    # ------------------------------------------------------------------
    # One planner segment and complete multi-via route.
    # ------------------------------------------------------------------
    def _run_planner_segment(self, name, start, goal):
        bounds = self._planner_bounds()
        clearance = float(self.pp_clearance_var.get())
        if name == 'Dijkstra':
            return grid_search_planner(
                start, goal, self.pp_obstacles, bounds,
                resolution=float(self.pp_resolution_var.get()),
                use_astar=False, clearance=clearance
            )
        if name == 'A*':
            return grid_search_planner(
                start, goal, self.pp_obstacles, bounds,
                resolution=float(self.pp_resolution_var.get()),
                use_astar=True, clearance=clearance
            )
        if name == 'RRT':
            return rrt_planner_robust(
                start, goal, self.pp_obstacles, bounds,
                step_size=float(self.pp_step_size_var.get()),
                max_iter=int(self.pp_max_iter_var.get()),
                goal_sample_rate=float(self.pp_goal_bias_var.get()),
                clearance=clearance,
                seed=int(self.seed_var.get()),
                rewire=False,
                rewire_radius=float(self.pp_rewire_radius_var.get())
            )
        if name == 'RRT*':
            return rrt_planner_robust(
                start, goal, self.pp_obstacles, bounds,
                step_size=float(self.pp_step_size_var.get()),
                max_iter=int(self.pp_max_iter_var.get()),
                goal_sample_rate=float(self.pp_goal_bias_var.get()),
                clearance=clearance,
                seed=int(self.seed_var.get()),
                rewire=True,
                rewire_radius=float(self.pp_rewire_radius_var.get())
            )
        raise ValueError(f'Unknown planner: {name}')

    def _run_one_planner(self, name):
        start = self._planner_head_start()
        vias = self._effective_shared_vias()
        if not vias:
            return {
                'name': name, 'success': False, 'path': [], 'time_s': 0.0,
                'path_length_m': float('nan'), 'nodes': 0, 'points': 0,
                'smoothness_deg': float('nan'), 'history': [],
                'message': 'No required via points. Enter via points in ML Control.',
                'via_points_total': 0, 'via_points_reached': 0,
                'segment_results': []
            }

        full_path = [start]
        current = start
        total_time = 0.0
        total_nodes = 0
        all_history = []
        segment_results = []
        reached = 0

        for seg_idx, target in enumerate(vias, start=1):
            result = self._run_planner_segment(name, current, target)
            segment_results.append(result)
            total_time += float(result.get('time_s', 0.0))
            total_nodes += int(result.get('nodes', 0))
            if result.get('history'):
                all_history.extend(list(result['history']))

            if not result['success']:
                return {
                    'name': name, 'success': False, 'path': full_path,
                    'time_s': total_time,
                    'path_length_m': path_length_2d(full_path),
                    'nodes': total_nodes, 'points': len(full_path),
                    'smoothness_deg': path_smoothness_deg(full_path),
                    'history': all_history,
                    'message': (
                        f'Failed on segment {seg_idx}: {current} → {target}. '
                        f"{result.get('message','')}"
                    ),
                    'via_points_total': len(vias),
                    'via_points_reached': reached,
                    'segment_results': segment_results
                }

            seg_path = list(result['path'])
            if seg_path:
                full_path.extend(seg_path[1:])
            current = (float(target[0]), float(target[1]))
            reached += 1

        return {
            'name': name, 'success': True, 'path': full_path,
            'time_s': total_time,
            'path_length_m': path_length_2d(full_path),
            'nodes': total_nodes, 'points': len(full_path),
            'smoothness_deg': path_smoothness_deg(full_path),
            'history': all_history,
            'message': f'All {reached} required via points reached.',
            'via_points_total': len(vias),
            'via_points_reached': reached,
            'segment_results': segment_results
        }

    # ------------------------------------------------------------------
    # Planner actions/status use the full shared route.
    # ------------------------------------------------------------------
    def pp_plan_selected(self):
        alg = self.pp_algo_var.get()
        if not self._effective_shared_vias():
            messagebox.showinfo('Path Planning', 'Enter via points in ML Control first.')
            return
        self.status_var.set(f'Planning {alg} through all shared via points...')
        self.update_idletasks()
        result = self._run_one_planner(alg)
        self.pp_results = {alg: result}
        self.pp_selected_path_name = alg
        self.pp_refresh_results_tree()
        self.pp_update_figure()
        if result['success']:
            self.pp_status_var.set(
                f"{alg}: SUCCESS through {result['via_points_reached']}/{result['via_points_total']} via points | "
                f"total path={result['path_length_m']:.4f} m | time={result['time_s']:.4f} s | nodes={result['nodes']}. "
                'Transfer this full route to activate the planner in simulation.'
            )
        else:
            self.pp_status_var.set(
                f"{alg}: FAILED after {result['via_points_reached']}/{result['via_points_total']} via points | {result['message']}"
            )
        self.status_var.set(f'{alg} multi-via planning complete.')

    def pp_compare_all(self):
        if not self._effective_shared_vias():
            messagebox.showinfo('Path Planning', 'Enter via points in ML Control first.')
            return
        algs = ('Dijkstra','A*','RRT','RRT*')
        self.status_var.set('Comparing all planners through the SAME shared via points...')
        self.update_idletasks()
        self.pp_results = {a: self._run_one_planner(a) for a in algs}
        successful = [r for r in self.pp_results.values() if r['success']]
        if successful:
            best = min(successful, key=lambda r: (r['path_length_m'], r['time_s']))
            self.pp_selected_path_name = best['name']
            self.pp_algo_var.set(best['name'])
            self.pp_status_var.set(
                f"All algorithms used the same {best['via_points_total']} required via points. "
                f"Best route: {best['name']} | L={best['path_length_m']:.4f} m | T={best['time_s']:.4f} s."
            )
        else:
            self.pp_selected_path_name = None
            self.pp_status_var.set('No planner completed all shared via points.')
        self.pp_refresh_results_tree()
        self.pp_update_figure()
        self.status_var.set('Multi-via path-planning comparison complete.')

    def pp_plan_transfer_simulate(self):
        alg = self.pp_algo_var.get()
        self.pp_plan_selected()
        if alg not in self.pp_results or not self.pp_results[alg]['success']:
            return
        self.pp_use_selected_path_as_ml_waypoints()
        self.plan_path(simulate=True)

    # ------------------------------------------------------------------
    # Transfer route without overwriting the stored MASTER via sequence.
    # ------------------------------------------------------------------
    def pp_use_selected_path_as_ml_waypoints(self):
        name = self.pp_selected_path_name or self.pp_algo_var.get()
        if not name or name not in self.pp_results:
            messagebox.showinfo('Path Planning', 'Run a path planner first.')
            return
        r = self.pp_results[name]
        if not r['success'] or not r['path']:
            messagebox.showinfo('Path Planning', f'{name} has not completed all shared via points.')
            return

        # Dense collision-free planner route becomes the ML execution waypoint
        # sequence. The original pp_shared_vias stays unchanged.
        self.waypoints = list(r['path'])
        self._refresh_waypoints()
        self.plan = None
        self.body_trail = []
        self.head_trail = []
        self.current_path_planner_var.set(f'Path Planner: {name}')
        self.route_source_var.set(
            f"Route source: {name} route through {r.get('via_points_total',0)} shared via points"
        )
        self._update_path_plot()
        self._refresh_algorithm_identity()
        self.pp_refresh_shared_vias()
        self.pp_status_var.set(
            f"{name} is ACTIVE for simulation. Its complete collision-free route passes through all "
            f"{r.get('via_points_total',0)} shared via points and is now supplied to the ML controller."
        )
        self.status_var.set(f'{name} multi-via route transferred to ML execution.')

    def pp_use_best_path_as_ml_waypoints(self):
        successful = [r for r in self.pp_results.values() if r['success']]
        if not successful:
            messagebox.showinfo('Path Planning', 'No planner completed all shared via points.')
            return
        best = min(successful, key=lambda r: (r['path_length_m'], r['time_s']))
        self.pp_selected_path_name = best['name']
        self.pp_algo_var.set(best['name'])
        self.pp_use_selected_path_as_ml_waypoints()

    # ------------------------------------------------------------------
    # Annotate shared via points on the inherited analysis figure.
    # ------------------------------------------------------------------
    def pp_update_figure(self):
        MLRobotDogAppV21.pp_update_figure(self)
        try:
            ax = self.pp_axes[0]
            vias = self._effective_shared_vias()
            if vias:
                V = np.asarray(vias, dtype=float)
                ax.scatter(V[:,0], V[:,1], marker='D', s=48, label='Required shared via points')
                for i,(x,y) in enumerate(vias, start=1):
                    ax.text(x, y, f' V{i}', fontsize=8)
            hs = self._planner_head_start()
            ax.scatter([hs[0]],[hs[1]], marker='s', s=58, label='HEAD start')
            ax.set_title('(a) Same ML Via Points + Classical Planner Routes')
            h,l = ax.get_legend_handles_labels()
            unique = {}
            for hh,ll in zip(h,l):
                unique[ll] = hh
            if unique:
                ax.legend(unique.values(), unique.keys(), fontsize=7, loc='best')
            self.pp_fig.suptitle(
                'Dijkstra / A* / RRT / RRT* Through the Same ML Via Points', fontsize=11
            )
            self.pp_fig.tight_layout()
            self.pp_canvas.draw_idle()
        except Exception:
            pass

    # The inherited 4x4 ML×planner comparison automatically uses this class's
    # multi-via _run_one_planner() implementation.
    def update_ml_planner_comparison_figure(self):
        super().update_ml_planner_comparison_figure()
        try:
            n = len(self._effective_shared_vias())
            self.comp_fig.suptitle(
                f'4 ML Controllers × 4 Path Planners | Same {n} Required Via Points', fontsize=11
            )
            self.comp_fig.tight_layout()
            self.comp_canvas.draw_idle()
        except Exception:
            pass


# ===========================================================================
# V2.3 — MOUSE OBSTACLES + SIM/GIF OBSTACLES + ALGORITHM IDENTITY
# ===========================================================================

class MLRobotDogAppV23(MLRobotDogAppV22):
    """
    V2.3 corrections:
      1. Obstacle centres are selected with the mouse on the Path Planning map.
         - Left click  : add obstacle
         - Right click : remove nearest obstacle
      2. Obstacles are rendered in the 3D simulation.
      3. Obstacles are rendered in exported GIFs.
      4. GIF title explicitly stores and displays BOTH:
         - active path-planning algorithm
         - active ML controller algorithm
    """

    def __init__(self):
        # These are plain Python values and can exist before Tk initialization.
        self.last_run_planner_name = "NONE — Manual HEAD waypoints"
        self.last_run_ml_name = "—"
        self.last_run_route_name = "Manual HEAD waypoints"
        self._pp_click_cid = None

        super().__init__()

        self.title(
            "ROBOQUAD FK V2.3 — Mouse Obstacles + Visible Obstacles + GIF Algorithm Identity"
        )

        self.status_var.set(
            "V2.3 ready: left-click planner map to add obstacles; right-click to remove."
        )

    # ------------------------------------------------------------------
    # Path-planning GUI — rebuild obstacle controls around mouse selection.
    # ------------------------------------------------------------------
    def _build_path_planning_tab(self, parent):
        # Variables must exist before the inherited builder finishes.
        if not hasattr(self, "pp_mouse_obstacle_mode_var"):
            self.pp_mouse_obstacle_mode_var = tk.StringVar(value="Add")
        if not hasattr(self, "pp_obstacle_height_var"):
            self.pp_obstacle_height_var = tk.DoubleVar(value=0.22)
        if not hasattr(self, "pp_mouse_status_var"):
            self.pp_mouse_status_var = tk.StringVar(
                value="Mouse: LEFT click adds an obstacle; RIGHT click removes the nearest obstacle."
            )

        super()._build_path_planning_tab(parent)

        # Find and replace the older manual X/Y obstacle entry controls.
        obstacle_frame = None

        def find_obstacle_frame(widget):
            nonlocal obstacle_frame
            for child in widget.winfo_children():
                try:
                    if (
                        isinstance(child, ttk.LabelFrame)
                        and str(child.cget("text")).startswith("2. Obstacles")
                    ):
                        obstacle_frame = child
                        return
                except Exception:
                    pass
                find_obstacle_frame(child)

        find_obstacle_frame(parent)

        if obstacle_frame is not None:
            for child in obstacle_frame.winfo_children():
                child.destroy()

            obstacle_frame.configure(
                text="2. Obstacles — Mouse Selection on Planner Map"
            )

            ttk.Label(
                obstacle_frame,
                text=(
                    "LEFT-click anywhere inside panel (a) Path Planning Overlay to place an obstacle. "
                    "RIGHT-click near an existing obstacle to delete it. "
                    "Only the radius/visual height are entered numerically; X and Y come from the mouse pointer."
                ),
                wraplength=410,
                justify="left"
            ).pack(anchor="w", pady=(0, 6))

            self.pp_obs_list = tk.Listbox(
                obstacle_frame,
                height=7,
                exportselection=False
            )
            self.pp_obs_list.pack(fill="x")

            settings = ttk.Frame(obstacle_frame)
            settings.pack(fill="x", pady=(6, 0))

            ttk.Label(
                settings,
                text="Obstacle radius (m)"
            ).grid(row=0, column=0, sticky="w")

            ttk.Spinbox(
                settings,
                from_=0.03,
                to=2.0,
                increment=0.01,
                textvariable=self.pp_obs_r_var,
                width=9
            ).grid(row=0, column=1, padx=4)

            ttk.Label(
                settings,
                text="3D/GIF height (m)"
            ).grid(row=0, column=2, sticky="w", padx=(10, 0))

            ttk.Spinbox(
                settings,
                from_=0.03,
                to=2.0,
                increment=0.01,
                textvariable=self.pp_obstacle_height_var,
                width=9
            ).grid(row=0, column=3, padx=4)

            buttons = ttk.Frame(obstacle_frame)
            buttons.pack(fill="x", pady=(6, 0))

            ttk.Button(
                buttons,
                text="Remove Selected",
                command=self.pp_remove_obstacle
            ).pack(side="left", padx=2)

            ttk.Button(
                buttons,
                text="Undo Last",
                command=self.pp_undo_last_obstacle
            ).pack(side="left", padx=2)

            ttk.Button(
                buttons,
                text="Clear All",
                command=self.pp_clear_obstacles
            ).pack(side="left", padx=2)

            ttk.Button(
                buttons,
                text="Load Example",
                command=self.pp_load_obstacle_example
            ).pack(side="left", padx=2)

            ttk.Label(
                obstacle_frame,
                textvariable=self.pp_mouse_status_var,
                wraplength=410,
                justify="left",
                style="Sub.TLabel"
            ).pack(anchor="w", pady=(7, 0))

        self.pp_refresh_obstacle_list()

        # Connect mouse clicks to panel (a) of the planner analysis figure.
        if hasattr(self, "pp_canvas"):
            try:
                if self._pp_click_cid is not None:
                    self.pp_canvas.mpl_disconnect(self._pp_click_cid)
            except Exception:
                pass

            self._pp_click_cid = self.pp_canvas.mpl_connect(
                "button_press_event",
                self._on_planner_map_click
            )

    # ------------------------------------------------------------------
    # Mouse obstacle interaction
    # ------------------------------------------------------------------
    def _on_planner_map_click(self, event):
        if not hasattr(self, "pp_axes"):
            return
        if event.inaxes is not self.pp_axes[0]:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = float(event.xdata)
        y = float(event.ydata)

        # LEFT click = add
        if int(event.button) == 1:
            try:
                r = max(0.01, float(self.pp_obs_r_var.get()))
            except Exception:
                r = 0.18
                self.pp_obs_r_var.set(r)

            self.pp_obstacles.append((x, y, r))
            self.pp_mouse_status_var.set(
                f"Added obstacle O{len(self.pp_obstacles)} at "
                f"X={x:+.3f}, Y={y:+.3f}, R={r:.3f} m."
            )
            self._invalidate_planner_after_obstacle_change()
            self.pp_refresh_obstacle_list()
            self.pp_update_figure()
            self._update_3d()
            return

        # RIGHT click = remove nearest
        if int(event.button) == 3 and self.pp_obstacles:
            distances = [
                math.hypot(x-float(ox), y-float(oy))
                for ox, oy, _r in self.pp_obstacles
            ]
            idx = int(np.argmin(distances))
            ox, oy, rr = self.pp_obstacles[idx]

            # Generous pointer-selection radius, while still avoiding accidental
            # deletion of a distant obstacle.
            click_limit = max(
                float(rr) * 1.6,
                0.18
            )

            if distances[idx] <= click_limit:
                del self.pp_obstacles[idx]
                self.pp_mouse_status_var.set(
                    f"Removed nearest obstacle at X={ox:+.3f}, Y={oy:+.3f}."
                )
                self._invalidate_planner_after_obstacle_change()
                self.pp_refresh_obstacle_list()
                self.pp_update_figure()
                self._update_3d()
            else:
                self.pp_mouse_status_var.set(
                    "Right-click closer to an obstacle centre to remove it."
                )

    def pp_undo_last_obstacle(self):
        if not self.pp_obstacles:
            return
        ox, oy, rr = self.pp_obstacles.pop()
        self.pp_mouse_status_var.set(
            f"Undid obstacle at X={ox:+.3f}, Y={oy:+.3f}, R={rr:.3f}."
        )
        self._invalidate_planner_after_obstacle_change()
        self.pp_refresh_obstacle_list()
        self.pp_update_figure()
        self._update_3d()

    def _invalidate_planner_after_obstacle_change(self):
        """
        Existing planner routes can become invalid when the obstacle map changes.
        Clear classical planner results and mark the active route as stale.
        """
        self.pp_results = {}
        self.pp_selected_path_name = None

        try:
            self.pp_refresh_results_tree()
        except Exception:
            pass

        try:
            self.current_path_planner_var.set(
                "Path Planner: NONE — Obstacle map changed; re-plan required"
            )
            self.route_source_var.set(
                "Route source: Existing waypoints retained; planner route requires re-planning"
            )
            self._refresh_algorithm_identity()
        except Exception:
            pass

        self.pp_status_var.set(
            "Obstacle map changed. Run the selected path planner again before planner-based simulation."
        )

    # Ensure list/button-based obstacle edits also invalidate old routes.
    def pp_remove_obstacle(self):
        sel = self.pp_obs_list.curselection() if hasattr(self, "pp_obs_list") else ()
        if not sel:
            return
        idx = int(sel[0])
        if 0 <= idx < len(self.pp_obstacles):
            del self.pp_obstacles[idx]
            self._invalidate_planner_after_obstacle_change()
            self.pp_refresh_obstacle_list()
            self.pp_update_figure()
            self._update_3d()

    def pp_clear_obstacles(self):
        self.pp_obstacles = []
        self._invalidate_planner_after_obstacle_change()
        self.pp_refresh_obstacle_list()
        self.pp_update_figure()
        self._update_3d()
        if hasattr(self, "pp_mouse_status_var"):
            self.pp_mouse_status_var.set("All obstacles cleared.")

    def pp_load_obstacle_example(self):
        self.pp_obstacles = [
            (0.32, 0.18, 0.15),
            (0.72, 0.30, 0.16),
            (0.68, -0.18, 0.13),
            (1.02, 0.06, 0.14),
        ]
        self._invalidate_planner_after_obstacle_change()
        self.pp_refresh_obstacle_list()
        self.pp_update_figure()
        self._update_3d()
        if hasattr(self, "pp_mouse_status_var"):
            self.pp_mouse_status_var.set(
                "Example obstacle field loaded. You can still left-click/right-click to edit it."
            )

    # ------------------------------------------------------------------
    # Save the algorithm identity of the actual simulation run.
    # ------------------------------------------------------------------
    def start_sim(self):
        try:
            _model, ml_name = self._active_ml_model()
        except Exception:
            ml_name = "—"

        try:
            planner = self.current_path_planner_var.get().replace(
                "Path Planner: ", ""
            )
        except Exception:
            planner = "NONE — Manual HEAD waypoints"

        try:
            route = self.route_source_var.get().replace(
                "Route source: ", ""
            )
        except Exception:
            route = "Manual HEAD waypoints"

        self.last_run_ml_name = ml_name
        self.last_run_planner_name = planner
        self.last_run_route_name = route

        super().start_sim()

    # ------------------------------------------------------------------
    # 3D obstacle drawing helper
    # ------------------------------------------------------------------
    def _draw_obstacles_3d(self, ax, z0, obstacle_height=None, label=True):
        if not getattr(self, "pp_obstacles", None):
            return

        try:
            height = (
                float(self.pp_obstacle_height_var.get())
                if obstacle_height is None
                else float(obstacle_height)
            )
        except Exception:
            height = 0.22

        height = max(0.02, height)

        theta = np.linspace(0.0, 2.0*math.pi, 40)

        for idx, (ox, oy, rr) in enumerate(self.pp_obstacles, start=1):
            ox = float(ox)
            oy = float(oy)
            rr = float(rr)

            cx = ox + rr*np.cos(theta)
            cy = oy + rr*np.sin(theta)

            # Bottom and top rings.
            ax.plot(
                cx, cy,
                np.full_like(theta, z0 + 0.006),
                linewidth=2.0
            )
            ax.plot(
                cx, cy,
                np.full_like(theta, z0 + height),
                linewidth=1.7
            )

            # Sparse vertical struts to make a visible wire-cylinder.
            for a in np.linspace(0.0, 2.0*math.pi, 9)[:-1]:
                x = ox + rr*math.cos(a)
                y = oy + rr*math.sin(a)
                ax.plot(
                    [x, x],
                    [y, y],
                    [z0 + 0.006, z0 + height],
                    linewidth=1.0,
                    alpha=0.75
                )

            if label:
                ax.text(
                    ox, oy, z0 + height + 0.015,
                    f" O{idx}",
                    fontsize=8
                )

    # ------------------------------------------------------------------
    # Obstacles visible in live 3D simulation
    # ------------------------------------------------------------------
    def _update_3d(self):
        super()._update_3d()

        try:
            z0 = float(self.ground_z)
            self._draw_obstacles_3d(
                self.ax3d,
                z0,
                obstacle_height=float(self.pp_obstacle_height_var.get()),
                label=True
            )

            # Expand view bounds if an obstacle lies outside inherited route bounds.
            if self.pp_obstacles:
                xlo, xhi = self.ax3d.get_xlim()
                ylo, yhi = self.ax3d.get_ylim()

                for ox, oy, rr in self.pp_obstacles:
                    xlo = min(xlo, float(ox)-float(rr)-0.20)
                    xhi = max(xhi, float(ox)+float(rr)+0.20)
                    ylo = min(ylo, float(oy)-float(rr)-0.20)
                    yhi = max(yhi, float(oy)+float(rr)+0.20)

                self.ax3d.set_xlim(xlo, xhi)
                self.ax3d.set_ylim(ylo, yhi)

            # Keep explicit identity title after adding obstacles.
            try:
                _model, ml_name = self._active_ml_model()
            except Exception:
                ml_name = "—"

            try:
                planner = self.current_path_planner_var.get().replace(
                    "Path Planner: ", ""
                )
            except Exception:
                planner = "NONE"

            try:
                route = self.route_source_var.get().replace(
                    "Route source: ", ""
                )
            except Exception:
                route = "Manual HEAD waypoints"

            hx, hy = self._head_xy()

            self.ax3d.set_title(
                f"Path Planner: {planner} | ML Controller: {ml_name}\n"
                f"Route: {route} | HEAD=({hx:.3f}, {hy:.3f}) m | "
                f"Yaw={math.degrees(self.base_yaw):.1f}° | "
                f"Obstacles={len(self.pp_obstacles)}"
            )

            self.fig3d.tight_layout()
            self.canvas3d.draw_idle()

        except Exception:
            pass

    # ------------------------------------------------------------------
    # GIF with obstacle geometry + algorithm identity
    # ------------------------------------------------------------------
    def save_animation_gif(self):
        if Image is None:
            messagebox.showerror(
                "GIF",
                "Install Pillow: pip install pillow"
            )
            return

        if not self.sim_log:
            messagebox.showinfo(
                "GIF",
                "Run a simulation first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[("GIF", "*.gif")]
        )
        if not path:
            return

        try:
            rows = self.sim_log
            stride = max(
                1,
                math.ceil(len(rows)/240)
            )
            rows = rows[::stride]

            frames = []
            fig = Figure(
                figsize=(8.2, 6.0),
                dpi=90
            )
            ax = fig.add_subplot(
                111,
                projection="3d"
            )

            size = max(
                1.5,
                float(self.plane_size_var.get())
            )
            half = size/2.0

            path_xy = (
                np.asarray(self.plan["path"], dtype=float)
                if self.plan
                else np.array([[0.0, 0.0]])
            )

            xmin = min(
                -half,
                float(path_xy[:,0].min()) - 0.5
            )
            xmax = max(
                +half,
                float(path_xy[:,0].max()) + 0.5
            )
            ymin = min(
                -half,
                float(path_xy[:,1].min()) - 0.5
            )
            ymax = max(
                +half,
                float(path_xy[:,1].max()) + 0.5
            )

            # Include obstacles in exported GIF bounds.
            for ox, oy, rr in self.pp_obstacles:
                xmin = min(
                    xmin,
                    float(ox)-float(rr)-0.25
                )
                xmax = max(
                    xmax,
                    float(ox)+float(rr)+0.25
                )
                ymin = min(
                    ymin,
                    float(oy)-float(rr)-0.25
                )
                ymax = max(
                    ymax,
                    float(oy)+float(rr)+0.25
                )

            g = self.fk.g
            L, W, H = (
                g.body_length,
                g.body_width,
                g.body_height
            )

            planner_name = getattr(
                self,
                "last_run_planner_name",
                "NONE — Manual HEAD waypoints"
            )
            ml_name = getattr(
                self,
                "last_run_ml_name",
                "—"
            )
            route_name = getattr(
                self,
                "last_run_route_name",
                "Manual HEAD waypoints"
            )

            try:
                obstacle_height = max(
                    0.02,
                    float(self.pp_obstacle_height_var.get())
                )
            except Exception:
                obstacle_height = 0.22

            for k, row in enumerate(rows):
                ax.clear()

                bx = float(
                    row["body_x_m"]
                )
                by = float(
                    row["body_y_m"]
                )
                yaw = math.radians(
                    float(row["body_yaw_deg"])
                )

                q = {
                    leg: {
                        j: float(
                            row[
                                f"{leg}_{j}_angle_deg"
                            ]
                        )
                        for j in JOINTS
                    }
                    for leg in LEG_ORDER
                }

                pts_all = self.fk.all_points(q)

                R = rot_z(yaw)[:3, :3]
                t = np.array([
                    bx, by, 0.0
                ])

                def wp(p):
                    return (
                        R @ np.asarray(p)
                        + t
                    )

                z0 = float(self.ground_z)

                # Ground grid.
                for xv in np.arange(
                    math.floor(xmin/.5)*.5,
                    math.ceil(xmax/.5)*.5 + .1,
                    .5
                ):
                    ax.plot(
                        [xv, xv],
                        [ymin, ymax],
                        [z0, z0],
                        linewidth=.35,
                        alpha=.18
                    )

                for yv in np.arange(
                    math.floor(ymin/.5)*.5,
                    math.ceil(ymax/.5)*.5 + .1,
                    .5
                ):
                    ax.plot(
                        [xmin, xmax],
                        [yv, yv],
                        [z0, z0],
                        linewidth=.35,
                        alpha=.18
                    )

                # Route / destinations.
                if self.waypoints:
                    P = np.asarray(
                        self.waypoints,
                        dtype=float
                    )
                    ax.plot(
                        P[:,0],
                        P[:,1],
                        np.full(
                            len(P),
                            z0+.012
                        ),
                        marker="o",
                        linewidth=1.8,
                        label="Execution route"
                    )

                # Obstacles.
                self._draw_obstacles_3d(
                    ax,
                    z0,
                    obstacle_height=obstacle_height,
                    label=True
                )

                # Body.
                top_b = np.array([
                    [ L/2,  W/2, H/2],
                    [ L/2, -W/2, H/2],
                    [-L/2, -W/2, H/2],
                    [-L/2,  W/2, H/2],
                    [ L/2,  W/2, H/2]
                ])
                bot_b = top_b.copy()
                bot_b[:,2] = -H/2

                top = np.vstack([
                    wp(p) for p in top_b
                ])
                bot = np.vstack([
                    wp(p) for p in bot_b
                ])

                ax.plot(
                    top[:,0],
                    top[:,1],
                    top[:,2],
                    linewidth=2
                )
                ax.plot(
                    bot[:,0],
                    bot[:,1],
                    bot[:,2],
                    linewidth=2
                )

                for i in range(4):
                    ax.plot(
                        [top[i,0], bot[i,0]],
                        [top[i,1], bot[i,1]],
                        [top[i,2], bot[i,2]],
                        linewidth=1
                    )

                # Head.
                ht_b = np.array([
                    [L/2+.025, W*.30, H*.42],
                    [L/2+.120, W*.30, H*.42],
                    [L/2+.120,-W*.30, H*.42],
                    [L/2+.025,-W*.30, H*.42],
                    [L/2+.025, W*.30, H*.42]
                ])
                hb_b = ht_b.copy()
                hb_b[:,2] = -H*.05

                ht = np.vstack([
                    wp(p) for p in ht_b
                ])
                hb = np.vstack([
                    wp(p) for p in hb_b
                ])

                ax.plot(
                    ht[:,0],
                    ht[:,1],
                    ht[:,2],
                    linewidth=1.5
                )
                ax.plot(
                    hb[:,0],
                    hb[:,1],
                    hb[:,2],
                    linewidth=1.5
                )

                for i in range(4):
                    ax.plot(
                        [ht[i,0], hb[i,0]],
                        [ht[i,1], hb[i,1]],
                        [ht[i,2], hb[i,2]],
                        linewidth=.8
                    )

                # Tail.
                rear = wp(
                    np.array(
                        [-L/2, 0, 0]
                    )
                )
                tail = wp(
                    np.array(
                        [-L/2-.12, 0, .07]
                    )
                )
                ax.plot(
                    [rear[0], tail[0]],
                    [rear[1], tail[1]],
                    [rear[2], tail[2]],
                    linewidth=1.8
                )

                # Legs.
                for leg, (
                    h, kn, ft
                ) in pts_all.items():
                    Pleg = np.vstack([
                        wp(h),
                        wp(kn),
                        wp(ft)
                    ])
                    ax.plot(
                        Pleg[:,0],
                        Pleg[:,1],
                        Pleg[:,2],
                        marker="o",
                        linewidth=2.5,
                        markersize=3
                    )

                # Trails.
                body_trail = np.array([
                    [
                        float(r["body_x_m"]),
                        float(r["body_y_m"])
                    ]
                    for r in rows[:k+1]
                ])

                ax.plot(
                    body_trail[:,0],
                    body_trail[:,1],
                    np.full(
                        len(body_trail),
                        z0+.018
                    ),
                    linewidth=1.0,
                    alpha=.55,
                    label="Body-centre path"
                )

                if "head_x_m" in row:
                    head_trail = np.array([
                        [
                            float(r["head_x_m"]),
                            float(r["head_y_m"])
                        ]
                        for r in rows[:k+1]
                    ])

                    ax.plot(
                        head_trail[:,0],
                        head_trail[:,1],
                        np.full(
                            len(head_trail),
                            z0+.028
                        ),
                        linewidth=2.0,
                        label="Executed HEAD path"
                    )

                    ax.scatter(
                        [float(row["head_x_m"])],
                        [float(row["head_y_m"])],
                        [z0+.045],
                        s=45,
                        marker="*"
                    )

                ax.set_xlim(
                    xmin, xmax
                )
                ax.set_ylim(
                    ymin, ymax
                )
                ax.set_zlim(
                    min(
                        z0-.06,
                        -0.32
                    ),
                    max(
                        0.28,
                        z0 + obstacle_height + 0.10
                    )
                )

                ax.set_xlabel(
                    "+X Forward (m)"
                )
                ax.set_ylabel(
                    "+Y Left (m)"
                )
                ax.set_zlabel(
                    "Z (m)"
                )

                # The requested algorithm identity is permanently stamped into
                # every GIF frame.
                ax.set_title(
                    f"Path Planner: {planner_name} | ML Controller: {ml_name}\n"
                    f"Route: {route_name} | "
                    f"t={float(row['time_s']):.2f} s | "
                    f"Command={row['direction']} | "
                    f"Obstacles={len(self.pp_obstacles)}"
                )

                ax.view_init(
                    elev=25,
                    azim=-55
                )

                try:
                    ax.legend(
                        loc="upper right",
                        fontsize=6
                    )
                except Exception:
                    pass

                buf = BytesIO()
                fig.savefig(
                    buf,
                    format="png",
                    bbox_inches="tight"
                )
                buf.seek(0)

                im = Image.open(
                    buf
                ).convert(
                    "P",
                    palette=Image.ADAPTIVE
                )
                frames.append(
                    im.copy()
                )
                buf.close()

            if not frames:
                raise RuntimeError(
                    "No GIF frames were generated."
                )

            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
                optimize=False
            )

            self.status_var.set(
                f"Saved animation with obstacles and algorithm identity: {Path(path).name}"
            )

            messagebox.showinfo(
                "GIF Saved",
                f"Animation saved with:\n\n"
                f"Path Planner: {planner_name}\n"
                f"ML Controller: {ml_name}\n"
                f"Obstacles: {len(self.pp_obstacles)}\n\n"
                f"{path}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Save GIF",
                str(exc)
            )


# ===========================================================================
# V2.4 — DYNAMIC / SUDDEN OBSTACLE REPLANNING
# ===========================================================================

class MLRobotDogAppV24(MLRobotDogAppV23):
    """
    Adds dynamic obstacle insertion during an active simulation.

    Workflow
    --------
    1. User enters sudden obstacle X, Y and radius in the 3D Simulation tab.
    2. Click "ADD DYNAMIC OBSTACLE + REPLAN".
    3. Current motion is immediately paused.
    4. The obstacle is added to the planner collision map.
    5. The selected/active Dijkstra, A*, RRT or RRT* planner replans from the
       CURRENT HEAD position through the REMAINING master via points.
    6. The currently selected ML controller regenerates gait commands from the
       CURRENT body pose.
    7. Simulation resumes without resetting the executed trail or simulation
       clock.

    Dynamic obstacles are time-stamped so an exported GIF shows the obstacle
    only from the instant it appeared.
    """

    def __init__(self):
        # Plain Python values can be initialized before Tk exists.
        self.dynamic_obstacle_events = []
        self.run_initial_obstacles = []
        self.dynamic_replan_count = 0
        self._dynamic_resume_in_progress = False

        super().__init__()

        self.title(
            "ROBOQUAD FK V2.4 — Dynamic Obstacle Detection / Replanning"
        )

        self.status_var.set(
            "V2.4 ready: sudden obstacles can be entered during simulation and trigger immediate replanning."
        )

    # ------------------------------------------------------------------
    # Dynamic obstacle controls in the 3D Simulation tab
    # ------------------------------------------------------------------
    def _build_simulation_tab(self, parent):
        # Variables must exist before widgets are created.
        if not hasattr(self, "dyn_obs_x_var"):
            self.dyn_obs_x_var = tk.DoubleVar(value=0.50)
            self.dyn_obs_y_var = tk.DoubleVar(value=0.00)
            self.dyn_obs_r_var = tk.DoubleVar(value=0.15)
            self.dyn_replanner_var = tk.StringVar(value="Active Planner")
            self.dyn_status_var = tk.StringVar(
                value="Dynamic obstacle: enter X/Y/R while the robot is moving."
            )

        super()._build_simulation_tab(parent)

        dyn = ttk.LabelFrame(
            parent,
            text="Dynamic / Sudden Obstacle — Emergency Replanning",
            padding=7
        )

        # Put it above the large 3D canvas so it remains visible.
        try:
            dyn.pack(
                fill="x",
                padx=6,
                pady=(2, 4),
                before=self.canvas3d.get_tk_widget()
            )
        except Exception:
            dyn.pack(
                fill="x",
                padx=6,
                pady=(2, 4)
            )

        ttk.Label(
            dyn,
            text="Obstacle X (m)"
        ).grid(
            row=0, column=0,
            sticky="w", padx=3
        )
        ttk.Entry(
            dyn,
            textvariable=self.dyn_obs_x_var,
            width=9
        ).grid(
            row=0, column=1,
            padx=3
        )

        ttk.Label(
            dyn,
            text="Y (m)"
        ).grid(
            row=0, column=2,
            sticky="w", padx=(10, 3)
        )
        ttk.Entry(
            dyn,
            textvariable=self.dyn_obs_y_var,
            width=9
        ).grid(
            row=0, column=3,
            padx=3
        )

        ttk.Label(
            dyn,
            text="Radius (m)"
        ).grid(
            row=0, column=4,
            sticky="w", padx=(10, 3)
        )
        ttk.Entry(
            dyn,
            textvariable=self.dyn_obs_r_var,
            width=9
        ).grid(
            row=0, column=5,
            padx=3
        )

        ttk.Label(
            dyn,
            text="Replanner"
        ).grid(
            row=0, column=6,
            sticky="w", padx=(12, 3)
        )
        ttk.Combobox(
            dyn,
            textvariable=self.dyn_replanner_var,
            state="readonly",
            width=16,
            values=(
                "Active Planner",
                "Dijkstra",
                "A*",
                "RRT",
                "RRT*",
            )
        ).grid(
            row=0, column=7,
            padx=3
        )

        ttk.Button(
            dyn,
            text="ADD DYNAMIC OBSTACLE + REPLAN",
            command=self.add_dynamic_obstacle_and_replan
        ).grid(
            row=0, column=8,
            padx=(12, 3)
        )

        ttk.Button(
            dyn,
            text="Place 0.35 m Ahead of HEAD",
            command=self.place_dynamic_ahead_of_head
        ).grid(
            row=0, column=9,
            padx=3
        )

        ttk.Label(
            dyn,
            textvariable=self.dyn_status_var,
            wraplength=1450,
            justify="left",
            style="Sub.TLabel"
        ).grid(
            row=1, column=0,
            columnspan=10,
            sticky="w",
            padx=3,
            pady=(5, 0)
        )

        dyn.columnconfigure(
            8, weight=1
        )

    def place_dynamic_ahead_of_head(self):
        """
        Convenience: calculate an obstacle centre directly in front of the
        current HEAD, while still allowing X/Y to be edited before insertion.
        """
        try:
            hx, hy = self._head_xy()
            d = 0.35
            x = hx + math.cos(self.base_yaw) * d
            y = hy + math.sin(self.base_yaw) * d
            self.dyn_obs_x_var.set(round(x, 4))
            self.dyn_obs_y_var.set(round(y, 4))
            self.dyn_status_var.set(
                f"Prepared sudden obstacle 0.35 m ahead of HEAD at X={x:.3f}, Y={y:.3f}. "
                "Press ADD DYNAMIC OBSTACLE + REPLAN."
            )
        except Exception as exc:
            messagebox.showerror(
                "Dynamic Obstacle",
                str(exc)
            )

    # ------------------------------------------------------------------
    # Fresh simulation snapshot for time-accurate GIF obstacle appearance
    # ------------------------------------------------------------------
    def start_sim(self):
        if not self._dynamic_resume_in_progress:
            self.run_initial_obstacles = list(
                getattr(self, "pp_obstacles", [])
            )
            self.dynamic_obstacle_events = []
            self.dynamic_replan_count = 0

        super().start_sim()

    # ------------------------------------------------------------------
    # Determine the remaining ORIGINAL required via points
    # ------------------------------------------------------------------
    def _remaining_master_vias_from_executed_trail(self):
        """
        Sequentially determine which required master via points have already
        been reached by the executed HEAD trail.
        """
        try:
            vias = list(
                self._effective_shared_vias()
            )
        except Exception:
            vias = list(
                getattr(self, "pp_shared_vias", [])
            )

        if not vias:
            return []

        trail = list(
            getattr(self, "head_trail", [])
        )
        if not trail:
            return vias

        tol = max(
            0.05,
            float(self.tolerance_var.get()) * 1.8
        )

        via_idx = 0

        # Sequential matching preserves the required via-point order.
        for px, py in trail:
            if via_idx >= len(vias):
                break

            vx, vy = vias[via_idx]
            if math.hypot(
                float(px)-float(vx),
                float(py)-float(vy)
            ) <= tol:
                via_idx += 1

        return vias[via_idx:]

    def _resolve_dynamic_replanner(self):
        selected = self.dyn_replanner_var.get()

        if selected != "Active Planner":
            return selected

        try:
            active = self.current_path_planner_var.get().replace(
                "Path Planner: ", ""
            ).strip()
        except Exception:
            active = ""

        for name in (
            "Dijkstra",
            "A*",
            "RRT",
            "RRT*"
        ):
            if active.startswith(name):
                return name

        # No planner was active (for example, manual route). Fall back to
        # whatever the user selected in the Path Planning tab.
        try:
            return self.pp_algo_var.get()
        except Exception:
            return "A*"

    # ------------------------------------------------------------------
    # Dynamic bounds and segment planning from CURRENT HEAD
    # ------------------------------------------------------------------
    def _dynamic_planner_bounds(
        self,
        current_head,
        remaining_vias
    ):
        try:
            half = max(
                1.0,
                float(self.plane_size_var.get()) / 2.0
            )
        except Exception:
            half = 2.0

        xs = [
            -half, +half,
            float(current_head[0])
        ]
        ys = [
            -half, +half,
            float(current_head[1])
        ]

        for vx, vy in remaining_vias:
            xs.append(float(vx))
            ys.append(float(vy))

        clearance = max(
            0.0,
            float(self.pp_clearance_var.get())
        )

        for ox, oy, rr in self.pp_obstacles:
            reff = float(rr) + clearance
            xs.extend([
                float(ox)-reff,
                float(ox)+reff
            ])
            ys.extend([
                float(oy)-reff,
                float(oy)+reff
            ])

        margin = max(
            0.45,
            3.0 * float(self.pp_step_size_var.get()),
            3.0 * float(self.pp_resolution_var.get())
        )

        return (
            min(xs)-margin,
            max(xs)+margin,
            min(ys)-margin,
            max(ys)+margin
        )

    def _dynamic_plan_segment(
        self,
        planner_name,
        start,
        goal,
        bounds
    ):
        clearance = float(
            self.pp_clearance_var.get()
        )

        if planner_name == "Dijkstra":
            return grid_search_planner(
                start,
                goal,
                self.pp_obstacles,
                bounds,
                resolution=float(
                    self.pp_resolution_var.get()
                ),
                use_astar=False,
                clearance=clearance
            )

        if planner_name == "A*":
            return grid_search_planner(
                start,
                goal,
                self.pp_obstacles,
                bounds,
                resolution=float(
                    self.pp_resolution_var.get()
                ),
                use_astar=True,
                clearance=clearance
            )

        if planner_name == "RRT":
            return rrt_planner_robust(
                start,
                goal,
                self.pp_obstacles,
                bounds,
                step_size=float(
                    self.pp_step_size_var.get()
                ),
                max_iter=int(
                    self.pp_max_iter_var.get()
                ),
                goal_sample_rate=float(
                    self.pp_goal_bias_var.get()
                ),
                clearance=clearance,
                seed=int(self.seed_var.get())
                    + self.dynamic_replan_count,
                rewire=False,
                rewire_radius=float(
                    self.pp_rewire_radius_var.get()
                )
            )

        if planner_name == "RRT*":
            return rrt_planner_robust(
                start,
                goal,
                self.pp_obstacles,
                bounds,
                step_size=float(
                    self.pp_step_size_var.get()
                ),
                max_iter=int(
                    self.pp_max_iter_var.get()
                ),
                goal_sample_rate=float(
                    self.pp_goal_bias_var.get()
                ),
                clearance=clearance,
                seed=int(self.seed_var.get())
                    + self.dynamic_replan_count,
                rewire=True,
                rewire_radius=float(
                    self.pp_rewire_radius_var.get()
                )
            )

        raise ValueError(
            f"Unknown dynamic replanner: {planner_name}"
        )

    def _plan_dynamic_remaining_route(
        self,
        planner_name,
        current_head,
        remaining_vias
    ):
        if not remaining_vias:
            return {
                "success": True,
                "path": [current_head],
                "time_s": 0.0,
                "nodes": 0,
                "message": "No remaining via points.",
            }

        bounds = self._dynamic_planner_bounds(
            current_head,
            remaining_vias
        )

        full_path = [
            (
                float(current_head[0]),
                float(current_head[1])
            )
        ]
        current = full_path[0]
        total_time = 0.0
        total_nodes = 0

        for i, target in enumerate(
            remaining_vias,
            start=1
        ):
            result = self._dynamic_plan_segment(
                planner_name,
                current,
                target,
                bounds
            )

            total_time += float(
                result.get("time_s", 0.0)
            )
            total_nodes += int(
                result.get("nodes", 0)
            )

            if not result["success"]:
                return {
                    "success": False,
                    "path": full_path,
                    "time_s": total_time,
                    "nodes": total_nodes,
                    "message": (
                        f"Dynamic replan failed on remaining segment {i}: "
                        f"{current} → {target}. "
                        f"{result.get('message', '')}"
                    ),
                }

            segment = list(
                result["path"]
            )

            if segment:
                full_path.extend(
                    segment[1:]
                )

            current = (
                float(target[0]),
                float(target[1])
            )

        return {
            "success": True,
            "path": full_path,
            "time_s": total_time,
            "nodes": total_nodes,
            "message": (
                f"Dynamic route completed through {len(remaining_vias)} remaining via point(s)."
            ),
        }

    # ------------------------------------------------------------------
    # Main sudden-obstacle action
    # ------------------------------------------------------------------
    def add_dynamic_obstacle_and_replan(self):
        try:
            x = float(
                self.dyn_obs_x_var.get()
            )
            y = float(
                self.dyn_obs_y_var.get()
            )
            r = float(
                self.dyn_obs_r_var.get()
            )
        except Exception:
            messagebox.showerror(
                "Dynamic Obstacle",
                "Enter numeric X, Y and radius."
            )
            return

        if r <= 0.0:
            messagebox.showerror(
                "Dynamic Obstacle",
                "Obstacle radius must be greater than zero."
            )
            return

        # Dynamic replanning is meaningful during an active or paused run.
        if not getattr(
            self, "sim_running", False
        ):
            messagebox.showinfo(
                "Dynamic Obstacle",
                "Start the simulation first. "
                "For obstacles known before motion, use the Path Planning tab."
            )
            return

        # Freeze the current simulation immediately.
        if self.sim_job:
            try:
                self.after_cancel(
                    self.sim_job
                )
            except Exception:
                pass
            self.sim_job = None

        self.sim_paused = True

        current_body = (
            float(self.base_x),
            float(self.base_y),
            float(self.base_yaw)
        )
        current_head = self._head_xy()

        clearance = max(
            0.0,
            float(self.pp_clearance_var.get())
        )

        # Do not allow a newly reported obstacle to physically contain the
        # already occupied HEAD reference point.
        if math.hypot(
            x-current_head[0],
            y-current_head[1]
        ) <= r + clearance:
            self.sim_paused = False
            self.dyn_status_var.set(
                "Dynamic obstacle rejected: its clearance zone overlaps the current HEAD position."
            )
            self._sim_frame()
            return

        planner_name = self._resolve_dynamic_replanner()
        model, ml_name = self._active_ml_model()

        if model is None:
            self.sim_paused = False
            messagebox.showerror(
                "Dynamic Replanning",
                "No trained/loaded ML controller is available."
            )
            self._sim_frame()
            return

        remaining_vias = (
            self._remaining_master_vias_from_executed_trail()
        )

        if not remaining_vias:
            self.sim_paused = False
            self.dyn_status_var.set(
                "All required via points have already been reached; no dynamic replan is required."
            )
            self._sim_frame()
            return

        # Add obstacle to the live map without using the static-obstacle
        # invalidation routine; we are replanning immediately here.
        obstacle = (
            float(x),
            float(y),
            float(r)
        )
        self.pp_obstacles.append(
            obstacle
        )

        event = {
            "time_s": float(
                self.sim_time
            ),
            "x": float(x),
            "y": float(y),
            "r": float(r),
            "planner": planner_name,
            "ml": ml_name,
            "remaining_vias": len(
                remaining_vias
            ),
        }

        self.dynamic_obstacle_events.append(
            event
        )
        self.dynamic_replan_count += 1

        self.pp_refresh_obstacle_list()
        self.pp_update_figure()
        self._update_3d()

        self.dyn_status_var.set(
            f"Dynamic obstacle #{self.dynamic_replan_count} appeared at "
            f"t={self.sim_time:.2f} s. Replanning with {planner_name}..."
        )
        self.status_var.set(
            f"Emergency stop: dynamic obstacle detected. {planner_name} is replanning."
        )
        self.update_idletasks()

        route_result = (
            self._plan_dynamic_remaining_route(
                planner_name,
                current_head,
                remaining_vias
            )
        )

        if not route_result["success"]:
            # Keep obstacle visible and simulation paused for safety.
            self.sim_paused = True
            self.dyn_status_var.set(
                route_result["message"]
                + " Simulation remains PAUSED."
            )
            self.status_var.set(
                "Dynamic replanning failed; simulation paused."
            )
            messagebox.showerror(
                "Dynamic Replanning Failed",
                route_result["message"]
            )
            return

        new_route = list(
            route_result["path"]
        )

        allow_reverse = (
            self.path_policy_var.get()
            == "Allow Automatic Reverse"
        )

        new_ml_plan = plan_ml_waypoints(
            model,
            new_route,
            start=current_body,
            max_step=self.max_step_var.get(),
            max_yaw_deg=self.max_yaw_var.get(),
            tolerance=self.tolerance_var.get(),
            max_commands=self.max_commands_var.get(),
            head_offset=self._head_offset(),
            turn_threshold_deg=self.turn_threshold_var.get(),
            allow_reverse=allow_reverse
        )

        if not new_ml_plan.get(
            "commands"
        ):
            self.sim_paused = True
            self.dyn_status_var.set(
                "Classical replanning succeeded, but the ML controller generated no continuation commands. "
                "Simulation remains paused."
            )
            return

        # Replace only the FUTURE route/commands. Preserve master required via
        # points, past logs, trails, joint displacement reference and sim_time.
        self.waypoints = new_route
        self.plan = new_ml_plan

        self.current_path_planner_var.set(
            f"Path Planner: {planner_name}"
        )
        self.route_source_var.set(
            f"Route source: DYNAMIC {planner_name} replan through "
            f"{len(remaining_vias)} remaining master via point(s)"
        )

        self.last_run_planner_name = (
            planner_name
        )
        self.last_run_ml_name = ml_name
        self.last_run_route_name = (
            f"DYNAMIC {planner_name} replan"
        )

        self._refresh_algorithm_identity()
        self._update_path_plot()
        self.pp_update_figure()

        # Start the new command sequence from the robot's CURRENT pose.
        self.command_index = 0
        self.command_elapsed = 0.0
        self.sim_running = True
        self.sim_paused = False

        # Keep derivative continuity based on the actual current joint state.
        self.prev_joint_angles = {
            leg: {
                j: float(
                    self.joint_angles[leg][j]
                )
                for j in JOINTS
            }
            for leg in LEG_ORDER
        }

        self._prepare_command()

        self.dyn_status_var.set(
            f"Dynamic obstacle #{self.dynamic_replan_count}: "
            f"{planner_name} replanned {path_length_2d(new_route):.3f} m "
            f"through {len(remaining_vias)} remaining via point(s) in "
            f"{route_result['time_s']:.4f} s. "
            f"ML Controller={ml_name}. Simulation resumed."
        )

        self.status_var.set(
            f"Dynamic replan complete: {planner_name} + {ml_name}. Simulation resumed."
        )

        self._sim_frame()

    # ------------------------------------------------------------------
    # Add dynamic-obstacle metadata to each logged frame
    # ------------------------------------------------------------------
    def _log_frame(self, error):
        super()._log_frame(error)

        if self.sim_log:
            row = self.sim_log[-1]
            row["dynamic_replan_count"] = int(
                self.dynamic_replan_count
            )
            row["obstacle_count"] = len(
                self.pp_obstacles
            )

            if self.dynamic_obstacle_events:
                evt = self.dynamic_obstacle_events[-1]
                row["last_dynamic_obstacle_time_s"] = evt[
                    "time_s"
                ]
                row["last_dynamic_obstacle_x_m"] = evt[
                    "x"
                ]
                row["last_dynamic_obstacle_y_m"] = evt[
                    "y"
                ]
                row["last_dynamic_obstacle_r_m"] = evt[
                    "r"
                ]
            else:
                row["last_dynamic_obstacle_time_s"] = float(
                    "nan"
                )
                row["last_dynamic_obstacle_x_m"] = float(
                    "nan"
                )
                row["last_dynamic_obstacle_y_m"] = float(
                    "nan"
                )
                row["last_dynamic_obstacle_r_m"] = float(
                    "nan"
                )

    # ------------------------------------------------------------------
    # Dynamic obstacle labeling in live 3D
    # ------------------------------------------------------------------
    def _is_dynamic_obstacle(
        self,
        obstacle
    ):
        ox, oy, rr = map(
            float, obstacle
        )

        for evt in self.dynamic_obstacle_events:
            if (
                abs(ox-float(evt["x"])) < 1e-9
                and abs(oy-float(evt["y"])) < 1e-9
                and abs(rr-float(evt["r"])) < 1e-9
            ):
                return True

        return False

    def _draw_obstacle_list_3d(
        self,
        ax,
        z0,
        obstacles,
        obstacle_height=None,
        label=True
    ):
        if not obstacles:
            return

        try:
            height = (
                float(self.pp_obstacle_height_var.get())
                if obstacle_height is None
                else float(obstacle_height)
            )
        except Exception:
            height = 0.22

        height = max(
            0.02,
            height
        )

        theta = np.linspace(
            0.0,
            2.0*math.pi,
            40
        )

        for idx, obs in enumerate(
            obstacles,
            start=1
        ):
            ox, oy, rr = map(
                float, obs
            )

            cx = (
                ox
                + rr*np.cos(theta)
            )
            cy = (
                oy
                + rr*np.sin(theta)
            )

            ax.plot(
                cx, cy,
                np.full_like(
                    theta,
                    z0+0.006
                ),
                linewidth=2.0
            )

            ax.plot(
                cx, cy,
                np.full_like(
                    theta,
                    z0+height
                ),
                linewidth=1.7
            )

            for a in np.linspace(
                0.0,
                2.0*math.pi,
                9
            )[:-1]:
                xx = ox + rr*math.cos(a)
                yy = oy + rr*math.sin(a)

                ax.plot(
                    [xx, xx],
                    [yy, yy],
                    [
                        z0+0.006,
                        z0+height
                    ],
                    linewidth=1.0,
                    alpha=0.75
                )

            if label:
                prefix = (
                    "DYN"
                    if self._is_dynamic_obstacle(
                        obs
                    )
                    else "O"
                )

                ax.text(
                    ox,
                    oy,
                    z0+height+0.015,
                    f" {prefix}{idx}",
                    fontsize=8
                )

    def _draw_obstacles_3d(
        self,
        ax,
        z0,
        obstacle_height=None,
        label=True
    ):
        self._draw_obstacle_list_3d(
            ax,
            z0,
            list(
                getattr(
                    self,
                    "pp_obstacles",
                    []
                )
            ),
            obstacle_height=obstacle_height,
            label=label
        )

    # ------------------------------------------------------------------
    # Time-accurate obstacle state for GIF
    # ------------------------------------------------------------------
    def _obstacles_for_sim_time(
        self,
        sim_time
    ):
        obs = list(
            self.run_initial_obstacles
        )

        for evt in self.dynamic_obstacle_events:
            if float(evt["time_s"]) <= float(
                sim_time
            ) + 1e-12:
                obs.append(
                    (
                        float(evt["x"]),
                        float(evt["y"]),
                        float(evt["r"])
                    )
                )

        return obs

    # ------------------------------------------------------------------
    # GIF override: dynamic obstacle appears at its actual insertion time
    # ------------------------------------------------------------------
    def save_animation_gif(self):
        if Image is None:
            messagebox.showerror(
                "GIF",
                "Install Pillow: pip install pillow"
            )
            return

        if not self.sim_log:
            messagebox.showinfo(
                "GIF",
                "Run a simulation first."
            )
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".gif",
            filetypes=[
                ("GIF", "*.gif")
            ]
        )

        if not path:
            return

        try:
            rows = self.sim_log
            stride = max(
                1,
                math.ceil(
                    len(rows)/260
                )
            )
            rows = rows[::stride]

            frames = []
            fig = Figure(
                figsize=(8.4, 6.1),
                dpi=90
            )
            ax = fig.add_subplot(
                111,
                projection="3d"
            )

            # Bounds include every obstacle that ever appeared.
            all_obstacles = list(
                self.run_initial_obstacles
            ) + [
                (
                    e["x"],
                    e["y"],
                    e["r"]
                )
                for e in self.dynamic_obstacle_events
            ]

            size = max(
                1.5,
                float(
                    self.plane_size_var.get()
                )
            )
            half = size/2.0

            # Use all executed HEAD points for robust bounds even after
            # dynamic re-planning replaced self.plan.
            if self.head_trail:
                XY = np.asarray(
                    self.head_trail,
                    dtype=float
                )
            else:
                XY = np.array(
                    [[0.0, 0.0]],
                    dtype=float
                )

            xmin = min(
                -half,
                float(
                    XY[:,0].min()
                )-0.5
            )
            xmax = max(
                +half,
                float(
                    XY[:,0].max()
                )+0.5
            )
            ymin = min(
                -half,
                float(
                    XY[:,1].min()
                )-0.5
            )
            ymax = max(
                +half,
                float(
                    XY[:,1].max()
                )+0.5
            )

            for ox, oy, rr in all_obstacles:
                xmin = min(
                    xmin,
                    float(ox)-float(rr)-0.25
                )
                xmax = max(
                    xmax,
                    float(ox)+float(rr)+0.25
                )
                ymin = min(
                    ymin,
                    float(oy)-float(rr)-0.25
                )
                ymax = max(
                    ymax,
                    float(oy)+float(rr)+0.25
                )

            g = self.fk.g
            L, W, H = (
                g.body_length,
                g.body_width,
                g.body_height
            )

            planner_name = getattr(
                self,
                "last_run_planner_name",
                "NONE"
            )
            ml_name = getattr(
                self,
                "last_run_ml_name",
                "—"
            )

            try:
                obstacle_height = max(
                    0.02,
                    float(
                        self.pp_obstacle_height_var.get()
                    )
                )
            except Exception:
                obstacle_height = 0.22

            for k, row in enumerate(
                rows
            ):
                ax.clear()

                frame_time = float(
                    row["time_s"]
                )

                bx = float(
                    row["body_x_m"]
                )
                by = float(
                    row["body_y_m"]
                )
                yaw = math.radians(
                    float(
                        row["body_yaw_deg"]
                    )
                )

                q = {
                    leg: {
                        j: float(
                            row[
                                f"{leg}_{j}_angle_deg"
                            ]
                        )
                        for j in JOINTS
                    }
                    for leg in LEG_ORDER
                }

                pts_all = self.fk.all_points(
                    q
                )

                R = rot_z(
                    yaw
                )[:3, :3]

                t = np.array(
                    [
                        bx,
                        by,
                        0.0
                    ]
                )

                def wp(p):
                    return (
                        R
                        @ np.asarray(p)
                        + t
                    )

                z0 = float(
                    self.ground_z
                )

                # Ground grid.
                for xv in np.arange(
                    math.floor(xmin/.5)*.5,
                    math.ceil(xmax/.5)*.5 + .1,
                    .5
                ):
                    ax.plot(
                        [xv, xv],
                        [ymin, ymax],
                        [z0, z0],
                        linewidth=.35,
                        alpha=.18
                    )

                for yv in np.arange(
                    math.floor(ymin/.5)*.5,
                    math.ceil(ymax/.5)*.5 + .1,
                    .5
                ):
                    ax.plot(
                        [xmin, xmax],
                        [yv, yv],
                        [z0, z0],
                        linewidth=.35,
                        alpha=.18
                    )

                # Master required vias stay meaningful even after dynamic
                # planner-generated waypoints replace self.waypoints.
                master_vias = list(
                    getattr(
                        self,
                        "pp_shared_vias",
                        []
                    )
                )

                if master_vias:
                    P = np.asarray(
                        master_vias,
                        dtype=float
                    )
                    ax.plot(
                        P[:,0],
                        P[:,1],
                        np.full(
                            len(P),
                            z0+.012
                        ),
                        marker="o",
                        linewidth=1.3,
                        label="Required master vias"
                    )

                # Time-accurate obstacles.
                frame_obstacles = (
                    self._obstacles_for_sim_time(
                        frame_time
                    )
                )

                self._draw_obstacle_list_3d(
                    ax,
                    z0,
                    frame_obstacles,
                    obstacle_height=obstacle_height,
                    label=True
                )

                # Body box.
                top_b = np.array([
                    [ L/2,  W/2, H/2],
                    [ L/2, -W/2, H/2],
                    [-L/2, -W/2, H/2],
                    [-L/2,  W/2, H/2],
                    [ L/2,  W/2, H/2]
                ])

                bot_b = top_b.copy()
                bot_b[:,2] = -H/2

                top = np.vstack([
                    wp(p)
                    for p in top_b
                ])
                bot = np.vstack([
                    wp(p)
                    for p in bot_b
                ])

                ax.plot(
                    top[:,0],
                    top[:,1],
                    top[:,2],
                    linewidth=2
                )
                ax.plot(
                    bot[:,0],
                    bot[:,1],
                    bot[:,2],
                    linewidth=2
                )

                for i in range(4):
                    ax.plot(
                        [
                            top[i,0],
                            bot[i,0]
                        ],
                        [
                            top[i,1],
                            bot[i,1]
                        ],
                        [
                            top[i,2],
                            bot[i,2]
                        ],
                        linewidth=1
                    )

                # Head block.
                ht_b = np.array([
                    [
                        L/2+.025,
                        W*.30,
                        H*.42
                    ],
                    [
                        L/2+.120,
                        W*.30,
                        H*.42
                    ],
                    [
                        L/2+.120,
                        -W*.30,
                        H*.42
                    ],
                    [
                        L/2+.025,
                        -W*.30,
                        H*.42
                    ],
                    [
                        L/2+.025,
                        W*.30,
                        H*.42
                    ]
                ])

                hb_b = ht_b.copy()
                hb_b[:,2] = -H*.05

                ht = np.vstack([
                    wp(p)
                    for p in ht_b
                ])
                hb = np.vstack([
                    wp(p)
                    for p in hb_b
                ])

                ax.plot(
                    ht[:,0],
                    ht[:,1],
                    ht[:,2],
                    linewidth=1.5
                )
                ax.plot(
                    hb[:,0],
                    hb[:,1],
                    hb[:,2],
                    linewidth=1.5
                )

                for i in range(4):
                    ax.plot(
                        [
                            ht[i,0],
                            hb[i,0]
                        ],
                        [
                            ht[i,1],
                            hb[i,1]
                        ],
                        [
                            ht[i,2],
                            hb[i,2]
                        ],
                        linewidth=.8
                    )

                # Tail.
                rear = wp(
                    np.array(
                        [
                            -L/2,
                            0,
                            0
                        ]
                    )
                )
                tail = wp(
                    np.array(
                        [
                            -L/2-.12,
                            0,
                            .07
                        ]
                    )
                )

                ax.plot(
                    [
                        rear[0],
                        tail[0]
                    ],
                    [
                        rear[1],
                        tail[1]
                    ],
                    [
                        rear[2],
                        tail[2]
                    ],
                    linewidth=1.8
                )

                # Legs.
                for leg, (
                    h,
                    kn,
                    ft
                ) in pts_all.items():
                    Pleg = np.vstack([
                        wp(h),
                        wp(kn),
                        wp(ft)
                    ])

                    ax.plot(
                        Pleg[:,0],
                        Pleg[:,1],
                        Pleg[:,2],
                        marker="o",
                        linewidth=2.5,
                        markersize=3
                    )

                # Executed trails up to this frame.
                prefix_rows = rows[:k+1]

                body_trail = np.asarray([
                    [
                        float(rw["body_x_m"]),
                        float(rw["body_y_m"])
                    ]
                    for rw in prefix_rows
                ])

                ax.plot(
                    body_trail[:,0],
                    body_trail[:,1],
                    np.full(
                        len(body_trail),
                        z0+.018
                    ),
                    linewidth=1.0,
                    alpha=.55,
                    label="Executed body path"
                )

                if "head_x_m" in row:
                    head_trail = np.asarray([
                        [
                            float(rw["head_x_m"]),
                            float(rw["head_y_m"])
                        ]
                        for rw in prefix_rows
                    ])

                    ax.plot(
                        head_trail[:,0],
                        head_trail[:,1],
                        np.full(
                            len(head_trail),
                            z0+.028
                        ),
                        linewidth=2.0,
                        label="Executed HEAD path"
                    )

                    ax.scatter(
                        [
                            float(
                                row["head_x_m"]
                            )
                        ],
                        [
                            float(
                                row["head_y_m"]
                            )
                        ],
                        [z0+.045],
                        s=45,
                        marker="*"
                    )

                ax.set_xlim(
                    xmin,
                    xmax
                )
                ax.set_ylim(
                    ymin,
                    ymax
                )
                ax.set_zlim(
                    min(
                        z0-.06,
                        -0.32
                    ),
                    max(
                        0.28,
                        z0+obstacle_height+0.10
                    )
                )

                ax.set_xlabel(
                    "+X Forward (m)"
                )
                ax.set_ylabel(
                    "+Y Left (m)"
                )
                ax.set_zlabel(
                    "Z (m)"
                )

                dyn_count = sum(
                    1
                    for evt in self.dynamic_obstacle_events
                    if float(evt["time_s"])
                    <= frame_time + 1e-12
                )

                ax.set_title(
                    f"Path Planner: {planner_name} | ML Controller: {ml_name}\n"
                    f"t={frame_time:.2f} s | "
                    f"Command={row['direction']} | "
                    f"Static obstacles={len(self.run_initial_obstacles)} | "
                    f"Dynamic obstacles appeared={dyn_count}"
                )

                ax.view_init(
                    elev=25,
                    azim=-55
                )

                try:
                    ax.legend(
                        loc="upper right",
                        fontsize=6
                    )
                except Exception:
                    pass

                buf = BytesIO()

                fig.savefig(
                    buf,
                    format="png",
                    bbox_inches="tight"
                )

                buf.seek(0)

                im = Image.open(
                    buf
                ).convert(
                    "P",
                    palette=Image.ADAPTIVE
                )

                frames.append(
                    im.copy()
                )

                buf.close()

            if not frames:
                raise RuntimeError(
                    "No GIF frames were generated."
                )

            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=50,
                loop=0,
                optimize=False
            )

            self.status_var.set(
                f"Saved dynamic-obstacle animation: {Path(path).name}"
            )

            messagebox.showinfo(
                "GIF Saved",
                f"Animation saved with time-accurate dynamic obstacles.\n\n"
                f"Path Planner: {planner_name}\n"
                f"ML Controller: {ml_name}\n"
                f"Dynamic replans: {self.dynamic_replan_count}\n\n"
                f"{path}"
            )

        except Exception as exc:
            messagebox.showerror(
                "Save GIF",
                str(exc)
            )


# ===========================================================================
# V2.5 — QUADRUPED GAIT-AWARE A* (QGA*)
# ===========================================================================

def qga_path_min_clearance(path, obstacles):
    """Minimum point-to-obstacle-boundary clearance along a 2D path."""
    if not path:
        return float("nan")
    if not obstacles:
        return float("inf")

    best = float("inf")
    for px, py in path:
        for ox, oy, rr in obstacles:
            d = math.hypot(
                float(px) - float(ox),
                float(py) - float(oy)
            ) - float(rr)
            best = min(best, d)
    return float(best)


def qga_compress_collinear(path, eps=1e-10):
    """Remove only collinear intermediate grid points; preserve QGA* turns."""
    if not path or len(path) <= 2:
        return list(path or [])

    out = [path[0]]
    for i in range(1, len(path)-1):
        a = np.asarray(out[-1], dtype=float)
        b = np.asarray(path[i], dtype=float)
        c = np.asarray(path[i+1], dtype=float)

        ab = b - a
        bc = c - b
        cross = float(ab[0]*bc[1] - ab[1]*bc[0])

        # Keep a point whenever the direction changes.
        if abs(cross) > eps:
            out.append(path[i])

    out.append(path[-1])
    return out


class MLRobotDogAppV25(MLRobotDogAppV24):
    """
    V2.5 adds Quadruped Gait-Aware A* (QGA*).

    Implemented evaluation:

      f(n) = g(n)
             + w_h   h(n)
             + w_psi C_turn
             + w_o   C_clearance
             + w_q   C_joint
             + w_s   C_singularity
             + w_d   C_dynamic

    State:
        n = (x, y, psi)

    QGA* is therefore heading-aware. It also inflates obstacle clearance by
    the robot half-width so the cost/occupancy check is not based on an
    infinitesimal point robot.

    Notes on implementation:
      - C_turn uses normalized heading change.
      - C_clearance is an inverse-clearance cost, normalized for numerical
        stability.
      - C_joint is estimated from the simulator's actual 12-DOF gait equations.
      - C_singularity uses the minimum singular value of the numerical leg
        Jacobian across representative gait phases.
      - C_dynamic is activated by obstacles inserted through the dynamic
        obstacle mechanism.
    """

    QGA_PLANNERS = (
        "Dijkstra",
        "A*",
        "RRT",
        "RRT*",
        "QGA*",
    )

    ML_CONTROLLERS = (
        "Random Forest",
        "Extra Trees",
        "KNN",
        "MLP",
    )

    def __init__(self):
        self._qga_motion_cache = {}
        super().__init__()

        self.title(
            "ROBOQUAD-X Studio — Intelligent Quadruped Navigation & Motion Lab"
        )

        self.status_var.set(
            "V2.5 ready: QGA* added beside Dijkstra, A*, RRT and RRT*."
        )

    # ------------------------------------------------------------------
    # QGA* GUI variables
    # ------------------------------------------------------------------
    def _ensure_qga_vars(self):
        if hasattr(self, "qga_wh_var"):
            return

        self.qga_wh_var = tk.DoubleVar(value=1.00)
        self.qga_wturn_var = tk.DoubleVar(value=0.45)
        self.qga_wclear_var = tk.DoubleVar(value=0.35)
        self.qga_wjoint_var = tk.DoubleVar(value=0.15)
        self.qga_wsing_var = tk.DoubleVar(value=0.08)
        self.qga_wdynamic_var = tk.DoubleVar(value=0.40)
        self.qga_body_margin_var = tk.DoubleVar(value=0.02)
        self.qga_eps_var = tk.DoubleVar(value=0.02)

        self.qga_status_var = tk.StringVar(
            value=(
                "QGA*: heading + body clearance + gait/joint + singularity "
                "+ dynamic-obstacle aware."
            )
        )

    def reset_qga_weights(self):
        self.qga_wh_var.set(1.00)
        self.qga_wturn_var.set(0.45)
        self.qga_wclear_var.set(0.35)
        self.qga_wjoint_var.set(0.15)
        self.qga_wsing_var.set(0.08)
        self.qga_wdynamic_var.set(0.40)
        self.qga_body_margin_var.set(0.02)
        self.qga_eps_var.set(0.02)
        self._qga_motion_cache = {}
        self.qga_status_var.set("QGA* weights restored to recommended defaults.")

    # ------------------------------------------------------------------
    # Add QGA* to all existing planner selectors, keeping the rest unchanged
    # ------------------------------------------------------------------
    def _set_combobox_values_by_var(self, root, variable, values):
        target_name = str(variable)

        def walk(w):
            for child in w.winfo_children():
                if isinstance(child, ttk.Combobox):
                    try:
                        if str(child.cget("textvariable")) == target_name:
                            child.configure(values=values)
                    except Exception:
                        pass
                walk(child)

        walk(root)

    def _build_path_planning_tab(self, parent):
        self._ensure_qga_vars()
        super()._build_path_planning_tab(parent)

        self._set_combobox_values_by_var(
            parent,
            self.pp_algo_var,
            self.QGA_PLANNERS
        )

        # Find the existing Planner Settings panel and append only a QGA*
        # configuration panel below it.
        planner_frame = None

        def find_planner_frame(w):
            nonlocal planner_frame
            for child in w.winfo_children():
                try:
                    if (
                        isinstance(child, ttk.LabelFrame)
                        and child.cget("text") == "3. Planner Settings"
                    ):
                        planner_frame = child
                        return
                except Exception:
                    pass
                find_planner_frame(child)

        find_planner_frame(parent)

        if planner_frame is not None:
            qbox = ttk.LabelFrame(
                planner_frame.master,
                text="QGA* — Quadruped Gait-Aware Modified A*",
                padding=8
            )
            qbox.pack(
                fill="x",
                padx=8,
                pady=4,
                after=planner_frame
            )

            ttk.Label(
                qbox,
                text=(
                    "f(n) = g(n) + w_h h(n) + wψ Cturn + wo Cclearance "
                    "+ wq Cjoint + ws Csingularity + wd Cdynamic"
                ),
                style="Sub.TLabel",
                wraplength=410,
                justify="left"
            ).grid(
                row=0, column=0,
                columnspan=4,
                sticky="w",
                pady=(0, 5)
            )

            ttk.Label(
                qbox,
                text=(
                    "State: n=(x,y,ψ). QGA* uses the actual robot body width "
                    "and representative 12-DOF gait/Jacobian calculations."
                ),
                wraplength=410,
                justify="left"
            ).grid(
                row=1, column=0,
                columnspan=4,
                sticky="w",
                pady=(0, 6)
            )

            rows = (
                ("w_h  heuristic", self.qga_wh_var),
                ("wψ  turning", self.qga_wturn_var),
                ("wo  clearance", self.qga_wclear_var),
                ("wq  joint/gait", self.qga_wjoint_var),
                ("ws  singularity", self.qga_wsing_var),
                ("wd  dynamic obstacle", self.qga_wdynamic_var),
                ("Extra body margin (m)", self.qga_body_margin_var),
                ("ε normalization", self.qga_eps_var),
            )

            for r, (label, var) in enumerate(rows, start=2):
                ttk.Label(
                    qbox, text=label
                ).grid(
                    row=r, column=0,
                    sticky="w", pady=2
                )
                ttk.Entry(
                    qbox,
                    textvariable=var,
                    width=10
                ).grid(
                    row=r, column=1,
                    sticky="w",
                    padx=4,
                    pady=2
                )

            ttk.Button(
                qbox,
                text="Reset QGA* Weights",
                command=self.reset_qga_weights
            ).grid(
                row=2,
                column=2,
                rowspan=2,
                padx=8,
                sticky="nsew"
            )

            ttk.Label(
                qbox,
                textvariable=self.qga_status_var,
                wraplength=405,
                justify="left"
            ).grid(
                row=10,
                column=0,
                columnspan=4,
                sticky="w",
                pady=(6, 0)
            )

        # Correct explanatory titles to include QGA*.
        try:
            self.pp_fig.suptitle(
                "Dijkstra / A* / RRT / RRT* / QGA* — Same Shared Via Points",
                fontsize=11
            )
        except Exception:
            pass

    def _build_simulation_tab(self, parent):
        self._ensure_qga_vars()
        super()._build_simulation_tab(parent)

        # Dynamic replanner dropdown now also supports QGA*.
        self._set_combobox_values_by_var(
            parent,
            self.dyn_replanner_var,
            (
                "Active Planner",
                "Dijkstra",
                "A*",
                "RRT",
                "RRT*",
                "QGA*",
            )
        )

    def _build_ml_planner_comparison_tab(self, parent):
        self._ensure_qga_vars()
        super()._build_ml_planner_comparison_tab(parent)

        self._set_combobox_values_by_var(
            parent,
            self.comp_planner_var,
            self.QGA_PLANNERS
        )

        self._rename_widget_text(
            parent,
            {
                "EVALUATE 4 ML × 4 PLANNERS":
                    "EVALUATE 4 ML × 5 PLANNERS",
            }
        )

        # Add a concise interpretation without restructuring the working tab.
        try:
            outer = parent.winfo_children()[0]
            ttk.Label(
                outer,
                text=(
                    "QGA* is compared as a route planner. Random Forest / Extra Trees / "
                    "KNN / MLP remain the independent gait controllers."
                ),
                wraplength=1450,
                justify="left"
            ).pack(fill="x", pady=(0, 4))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # QGA* gait and singularity terms
    # ------------------------------------------------------------------
    def _qga_geometry_signature(self):
        g = self.fk.g
        return tuple(
            round(float(v), 6)
            for v in (
                g.body_length,
                g.body_width,
                g.body_height,
                g.FL_upper, g.FL_lower,
                g.FR_upper, g.FR_lower,
                g.RL_upper, g.RL_lower,
                g.RR_upper, g.RR_lower,
                float(self.duty_var.get()),
            )
        )

    def _qga_motion_terms(self, delta_heading_rad):
        """
        Compute gait/joint and singularity terms from actual simulator models.
        Cached because a grid search repeatedly uses the same 45-degree turn
        categories.
        """
        turn_deg = abs(math.degrees(float(delta_heading_rad)))
        bucket_deg = int(round(turn_deg / 45.0) * 45)
        bucket_deg = min(max(bucket_deg, 0), 180)

        direction = "Forward"
        if bucket_deg > 0:
            direction = (
                "Turn Left"
                if delta_heading_rad > 0
                else "Turn Right"
            )

        key = (
            direction,
            bucket_deg,
            self._qga_geometry_signature()
        )

        if key in self._qga_motion_cache:
            return self._qga_motion_cache[key]

        # More severe heading changes request a slightly larger representative
        # hip sweep. This is still bounded inside the simulator's gait range.
        turn_norm = min(
            abs(float(delta_heading_rad)) / math.pi,
            1.0
        )
        q2_amp = 18.0 + 6.0 * turn_norm
        q3_lift = 28.0

        phases = np.linspace(
            0.0, 1.0, 8, endpoint=False
        )

        configs = [
            self.gait_angles(
                float(ph),
                direction,
                q2_amp,
                q3_lift
            )
            for ph in phases
        ]

        # Total 12-DOF angular excursion over one representative cycle.
        total_excursion = 0.0
        previous = {
            leg: dict(STAND[leg])
            for leg in LEG_ORDER
        }

        sigma_min = float("inf")

        for q_all in configs:
            for leg in LEG_ORDER:
                for joint in JOINTS:
                    total_excursion += abs(
                        float(q_all[leg][joint])
                        - float(previous[leg][joint])
                    )

            metrics = self.fk.all_singularity_metrics(
                q_all
            )
            sigma_min = min(
                sigma_min,
                min(
                    float(metrics[leg]["sigma_min"])
                    for leg in LEG_ORDER
                )
            )

            previous = {
                leg: dict(q_all[leg])
                for leg in LEG_ORDER
            }

        # Close the gait cycle back to phase zero.
        if configs:
            first = configs[0]
            for leg in LEG_ORDER:
                for joint in JOINTS:
                    total_excursion += abs(
                        float(first[leg][joint])
                        - float(previous[leg][joint])
                    )

        # Dimensionless joint effort. Turning is intentionally more expensive.
        joint_cost = (
            total_excursion
            / max(
                1.0,
                len(configs) * 12.0 * 30.0
            )
        ) * (1.0 + 0.75 * turn_norm)

        eps = max(
            1e-6,
            float(self.qga_eps_var.get())
        )

        # Dimensionless inverse singularity term.
        # 0.02 m/rad is used only as a normalization reference.
        singularity_cost = min(
            8.0,
            0.02 / max(
                sigma_min + eps,
                1e-8
            )
        )

        result = {
            "joint_cost": float(joint_cost),
            "singularity_cost": float(singularity_cost),
            "sigma_min": float(sigma_min),
            "direction": direction,
        }

        self._qga_motion_cache[key] = result
        return result

    def _qga_dynamic_obstacles(self):
        dyn = []
        for evt in getattr(
            self,
            "dynamic_obstacle_events",
            []
        ):
            dyn.append(
                (
                    float(evt["x"]),
                    float(evt["y"]),
                    float(evt["r"])
                )
            )
        return dyn

    def _qga_inverse_clearance_cost(
        self,
        point,
        obstacles,
        effective_extra_clearance,
        resolution
    ):
        if not obstacles:
            return 0.0, float("inf")

        x, y = map(float, point)
        min_edge = float("inf")

        for ox, oy, rr in obstacles:
            edge = (
                math.hypot(
                    x-float(ox),
                    y-float(oy)
                )
                - (
                    float(rr)
                    + float(effective_extra_clearance)
                )
            )
            min_edge = min(
                min_edge,
                edge
            )

        eps = max(
            1e-6,
            float(self.qga_eps_var.get())
        )

        # Numerically stable normalized inverse distance:
        # behaves like 1/(d+eps), but remains well scaled relative to metre
        # path costs.
        c = (
            float(resolution)
            / max(
                min_edge
                + float(resolution)
                + eps,
                eps
            )
        )

        return float(min(c, 10.0)), float(min_edge)

    # ------------------------------------------------------------------
    # QGA* — heading-aware modified A*
    # ------------------------------------------------------------------
    def _qga_star_segment(
        self,
        start,
        goal,
        bounds,
        start_heading_rad=None
    ):
        t0 = time.perf_counter()

        start = (
            float(start[0]),
            float(start[1])
        )
        goal = (
            float(goal[0]),
            float(goal[1])
        )

        if start_heading_rad is None:
            start_heading_rad = math.radians(
                float(self.start_yaw_var.get())
            )

        resolution = max(
            0.03,
            float(self.pp_resolution_var.get())
        )

        xmin, xmax, ymin, ymax = map(
            float,
            bounds
        )

        # QGA* protects the finite-width body rather than treating the dog as
        # an infinitesimal point.
        robot_half_width = (
            float(self.fk.g.body_width) / 2.0
        )
        body_margin = max(
            0.0,
            float(self.qga_body_margin_var.get())
        )
        user_clearance = max(
            0.0,
            float(self.pp_clearance_var.get())
        )
        effective_clearance = (
            user_clearance
            + robot_half_width
            + body_margin
        )

        if not point_collision_free(
            start,
            self.pp_obstacles,
            effective_clearance
        ):
            return {
                "name": "QGA*",
                "success": False,
                "path": [],
                "time_s": time.perf_counter() - t0,
                "path_length_m": float("nan"),
                "nodes": 0,
                "points": 0,
                "smoothness_deg": float("nan"),
                "history": [],
                "message": (
                    "QGA* start HEAD position violates robot-body obstacle clearance."
                ),
                "qga_cost": float("nan"),
                "min_clearance_m": float("nan"),
                "final_heading_rad": float(start_heading_rad),
                "min_sigma": float("nan"),
            }

        if not point_collision_free(
            goal,
            self.pp_obstacles,
            effective_clearance
        ):
            return {
                "name": "QGA*",
                "success": False,
                "path": [],
                "time_s": time.perf_counter() - t0,
                "path_length_m": float("nan"),
                "nodes": 0,
                "points": 0,
                "smoothness_deg": float("nan"),
                "history": [],
                "message": (
                    "QGA* destination/via point violates robot-body obstacle clearance."
                ),
                "qga_cost": float("nan"),
                "min_clearance_m": float("nan"),
                "final_heading_rad": float(start_heading_rad),
                "min_sigma": float("nan"),
            }

        xs = np.arange(
            xmin,
            xmax + 0.5*resolution,
            resolution
        )
        ys = np.arange(
            ymin,
            ymax + 0.5*resolution,
            resolution
        )

        nx, ny = len(xs), len(ys)

        # Occupancy includes body width + requested safety margin.
        occ = np.zeros(
            (nx, ny),
            dtype=bool
        )

        for i, x in enumerate(xs):
            for j, y in enumerate(ys):
                if not point_collision_free(
                    (x, y),
                    self.pp_obstacles,
                    effective_clearance
                ):
                    occ[i, j] = True

        si = int(
            np.argmin(
                np.abs(xs-start[0])
            )
        )
        sj = int(
            np.argmin(
                np.abs(ys-start[1])
            )
        )
        gi = int(
            np.argmin(
                np.abs(xs-goal[0])
            )
        )
        gj = int(
            np.argmin(
                np.abs(ys-goal[1])
            )
        )

        if occ[si, sj] or occ[gi, gj]:
            return {
                "name": "QGA*",
                "success": False,
                "path": [],
                "time_s": time.perf_counter() - t0,
                "path_length_m": float("nan"),
                "nodes": 0,
                "points": 0,
                "smoothness_deg": float("nan"),
                "history": [],
                "message": (
                    "QGA* grid start/goal is occupied after body-clearance inflation."
                ),
                "qga_cost": float("nan"),
                "min_clearance_m": float("nan"),
                "final_heading_rad": float(start_heading_rad),
                "min_sigma": float("nan"),
            }

        # 8 heading-aware motion primitives.
        moves = (
            (+1,  0, 0.0),
            (+1, +1, math.pi/4),
            ( 0, +1, math.pi/2),
            (-1, +1, 3*math.pi/4),
            (-1,  0, math.pi),
            (-1, -1, -3*math.pi/4),
            ( 0, -1, -math.pi/2),
            (+1, -1, -math.pi/4),
        )

        heading_angles = [
            m[2]
            for m in moves
        ]

        def wrap(a):
            return (
                float(a) + math.pi
            ) % (
                2.0*math.pi
            ) - math.pi

        start_hidx = int(
            np.argmin(
                [
                    abs(
                        wrap(
                            a-start_heading_rad
                        )
                    )
                    for a in heading_angles
                ]
            )
        )

        start_state = (
            si,
            sj,
            start_hidx
        )

        # User-editable QGA* weights.
        wh = max(
            0.0,
            float(self.qga_wh_var.get())
        )
        wt = max(
            0.0,
            float(self.qga_wturn_var.get())
        )
        wo = max(
            0.0,
            float(self.qga_wclear_var.get())
        )
        wq = max(
            0.0,
            float(self.qga_wjoint_var.get())
        )
        ws = max(
            0.0,
            float(self.qga_wsing_var.get())
        )
        wd = max(
            0.0,
            float(self.qga_wdynamic_var.get())
        )

        dynamic_obstacles = (
            self._qga_dynamic_obstacles()
        )

        def heuristic(i, j):
            return math.hypot(
                goal[0]-float(xs[i]),
                goal[1]-float(ys[j])
            )

        pq = [
            (
                wh*heuristic(si, sj),
                0.0,
                start_state
            )
        ]

        parent = {
            start_state: None
        }
        g_cost = {
            start_state: 0.0
        }
        visited = set()

        expanded = 0
        goal_state = None
        min_sigma_on_search = float("inf")
        best_f_history = []

        while pq:
            fcur, gcur, state = (
                heapq.heappop(pq)
            )

            if state in visited:
                continue

            visited.add(state)
            expanded += 1

            i, j, hidx = state

            if (
                i == gi
                and j == gj
            ):
                goal_state = state
                break

            current_heading = (
                heading_angles[hidx]
            )

            for new_hidx, (
                di,
                dj,
                new_heading
            ) in enumerate(moves):

                ni = i + di
                nj = j + dj

                if not (
                    0 <= ni < nx
                    and 0 <= nj < ny
                ):
                    continue

                if occ[ni, nj]:
                    continue

                p0 = (
                    float(xs[i]),
                    float(ys[j])
                )
                p1 = (
                    float(xs[ni]),
                    float(ys[nj])
                )

                if not segment_collision_free(
                    p0,
                    p1,
                    self.pp_obstacles,
                    effective_clearance,
                    step=max(
                        0.02,
                        resolution/3.0
                    )
                ):
                    continue

                step_distance = euclid(
                    p0,
                    p1
                )

                delta_heading = wrap(
                    new_heading
                    - current_heading
                )

                c_turn = min(
                    abs(delta_heading)
                    / math.pi,
                    1.0
                )

                c_clear, _edge = (
                    self._qga_inverse_clearance_cost(
                        p1,
                        self.pp_obstacles,
                        effective_clearance,
                        resolution
                    )
                )

                motion = (
                    self._qga_motion_terms(
                        delta_heading
                    )
                )

                c_joint = float(
                    motion["joint_cost"]
                )
                c_sing = float(
                    motion["singularity_cost"]
                )

                min_sigma_on_search = min(
                    min_sigma_on_search,
                    float(
                        motion["sigma_min"]
                    )
                )

                c_dynamic = 0.0

                if dynamic_obstacles:
                    c_dynamic, _ = (
                        self._qga_inverse_clearance_cost(
                            p1,
                            dynamic_obstacles,
                            effective_clearance,
                            resolution
                        )
                    )

                transition_cost = (
                    step_distance
                    + wt*c_turn
                    + wo*c_clear
                    + wq*c_joint
                    + ws*c_sing
                    + wd*c_dynamic
                )

                ng = (
                    gcur
                    + transition_cost
                )

                nstate = (
                    ni,
                    nj,
                    new_hidx
                )

                if ng + 1e-12 < g_cost.get(
                    nstate,
                    float("inf")
                ):
                    g_cost[nstate] = ng
                    parent[nstate] = state

                    nf = (
                        ng
                        + wh*heuristic(
                            ni,
                            nj
                        )
                    )

                    heapq.heappush(
                        pq,
                        (
                            nf,
                            ng,
                            nstate
                        )
                    )

            if expanded % 25 == 0:
                best_f_history.append(
                    float(fcur)
                )

        if goal_state is None:
            return {
                "name": "QGA*",
                "success": False,
                "path": [],
                "time_s": time.perf_counter()-t0,
                "path_length_m": float("nan"),
                "nodes": expanded,
                "points": 0,
                "smoothness_deg": float("nan"),
                "history": best_f_history,
                "message": (
                    "QGA* found no feasible heading/body-clearance-aware path."
                ),
                "qga_cost": float("nan"),
                "min_clearance_m": float("nan"),
                "final_heading_rad": float(start_heading_rad),
                "min_sigma": (
                    min_sigma_on_search
                    if math.isfinite(
                        min_sigma_on_search
                    )
                    else float("nan")
                ),
            }

        states = []
        cur = goal_state

        while cur is not None:
            states.append(cur)
            cur = parent[cur]

        states.reverse()

        path = [
            (
                float(xs[s[0]]),
                float(ys[s[1]])
            )
            for s in states
        ]

        if path:
            path[0] = start
            path[-1] = goal

        path = qga_compress_collinear(
            path
        )

        final_heading = (
            heading_angles[
                goal_state[2]
            ]
        )

        raw_min_clearance = (
            qga_path_min_clearance(
                path,
                self.pp_obstacles
            )
        )

        elapsed = (
            time.perf_counter()-t0
        )

        return {
            "name": "QGA*",
            "success": True,
            "path": path,
            "time_s": elapsed,
            "path_length_m": path_length_2d(
                path
            ),
            "nodes": expanded,
            "points": len(path),
            "smoothness_deg": path_smoothness_deg(
                path
            ),
            "history": best_f_history,
            "message": (
                "QGA* heading/body-clearance/gait/singularity-aware route found."
            ),
            "qga_cost": float(
                g_cost[goal_state]
            ),
            "min_clearance_m": float(
                raw_min_clearance
            ),
            "effective_robot_clearance_m": float(
                effective_clearance
            ),
            "final_heading_rad": float(
                final_heading
            ),
            "min_sigma": (
                float(
                    min_sigma_on_search
                )
                if math.isfinite(
                    min_sigma_on_search
                )
                else float("nan")
            ),
        }

    # ------------------------------------------------------------------
    # Planner dispatch
    # ------------------------------------------------------------------
    def _run_planner_segment(
        self,
        name,
        start,
        goal
    ):
        if name != "QGA*":
            return super()._run_planner_segment(
                name,
                start,
                goal
            )

        return self._qga_star_segment(
            start,
            goal,
            self._planner_bounds(),
            start_heading_rad=math.radians(
                float(self.start_yaw_var.get())
            )
        )

    def _run_one_planner(
        self,
        name
    ):
        if name != "QGA*":
            result = super()._run_one_planner(
                name
            )

            if result.get(
                "success"
            ):
                result[
                    "min_clearance_m"
                ] = qga_path_min_clearance(
                    result.get(
                        "path", []
                    ),
                    self.pp_obstacles
                )

            return result

        start = (
            self._planner_head_start()
        )
        vias = (
            self._effective_shared_vias()
        )

        if not vias:
            return {
                "name": "QGA*",
                "success": False,
                "path": [],
                "time_s": 0.0,
                "path_length_m": float("nan"),
                "nodes": 0,
                "points": 0,
                "smoothness_deg": float("nan"),
                "history": [],
                "message": (
                    "No required via points. Enter via points in ML Control."
                ),
                "via_points_total": 0,
                "via_points_reached": 0,
                "segment_results": [],
                "qga_cost": float("nan"),
                "min_clearance_m": float("nan"),
                "min_sigma": float("nan"),
            }

        full_path = [
            start
        ]
        current = start

        current_heading = math.radians(
            float(
                self.start_yaw_var.get()
            )
        )

        total_time = 0.0
        total_nodes = 0
        total_qga_cost = 0.0
        all_history = []
        segment_results = []
        reached = 0
        min_sigma = float("inf")

        bounds = (
            self._planner_bounds()
        )

        for seg_idx, target in enumerate(
            vias,
            start=1
        ):
            result = (
                self._qga_star_segment(
                    current,
                    target,
                    bounds,
                    start_heading_rad=current_heading
                )
            )

            segment_results.append(
                result
            )

            total_time += float(
                result.get(
                    "time_s", 0.0
                )
            )
            total_nodes += int(
                result.get(
                    "nodes", 0
                )
            )

            if result.get(
                "history"
            ):
                all_history.extend(
                    result["history"]
                )

            if math.isfinite(
                float(
                    result.get(
                        "qga_cost",
                        float("nan")
                    )
                )
            ):
                total_qga_cost += float(
                    result["qga_cost"]
                )

            if math.isfinite(
                float(
                    result.get(
                        "min_sigma",
                        float("nan")
                    )
                )
            ):
                min_sigma = min(
                    min_sigma,
                    float(
                        result["min_sigma"]
                    )
                )

            if not result[
                "success"
            ]:
                return {
                    "name": "QGA*",
                    "success": False,
                    "path": full_path,
                    "time_s": total_time,
                    "path_length_m": path_length_2d(
                        full_path
                    ),
                    "nodes": total_nodes,
                    "points": len(
                        full_path
                    ),
                    "smoothness_deg": path_smoothness_deg(
                        full_path
                    ),
                    "history": all_history,
                    "message": (
                        f"QGA* failed on segment {seg_idx}: "
                        f"{current} → {target}. "
                        f"{result.get('message','')}"
                    ),
                    "via_points_total": len(
                        vias
                    ),
                    "via_points_reached": reached,
                    "segment_results": segment_results,
                    "qga_cost": total_qga_cost,
                    "min_clearance_m": qga_path_min_clearance(
                        full_path,
                        self.pp_obstacles
                    ),
                    "min_sigma": (
                        min_sigma
                        if math.isfinite(
                            min_sigma
                        )
                        else float("nan")
                    ),
                }

            seg_path = list(
                result["path"]
            )

            if seg_path:
                full_path.extend(
                    seg_path[1:]
                )

            current = (
                float(target[0]),
                float(target[1])
            )
            current_heading = float(
                result.get(
                    "final_heading_rad",
                    current_heading
                )
            )
            reached += 1

        result = {
            "name": "QGA*",
            "success": True,
            "path": full_path,
            "time_s": total_time,
            "path_length_m": path_length_2d(
                full_path
            ),
            "nodes": total_nodes,
            "points": len(
                full_path
            ),
            "smoothness_deg": path_smoothness_deg(
                full_path
            ),
            "history": all_history,
            "message": (
                f"QGA* reached all {reached} required via points."
            ),
            "via_points_total": len(
                vias
            ),
            "via_points_reached": reached,
            "segment_results": segment_results,
            "qga_cost": float(
                total_qga_cost
            ),
            "min_clearance_m": qga_path_min_clearance(
                full_path,
                self.pp_obstacles
            ),
            "min_sigma": (
                min_sigma
                if math.isfinite(
                    min_sigma
                )
                else float("nan")
            ),
            "final_heading_rad": float(
                current_heading
            ),
        }

        return result

    # ------------------------------------------------------------------
    # Path planner actions / comparison with QGA*
    # ------------------------------------------------------------------
    def pp_plan_selected(self):
        super().pp_plan_selected()

        if self.pp_algo_var.get() == "QGA*":
            r = self.pp_results.get(
                "QGA*"
            )

            if r and r.get(
                "success"
            ):
                self.qga_status_var.set(
                    f"QGA* success | Composite cost={r.get('qga_cost',float('nan')):.4f} | "
                    f"Path={r['path_length_m']:.4f} m | "
                    f"Min obstacle clearance={r.get('min_clearance_m',float('nan')):.4f} m | "
                    f"Min σ={r.get('min_sigma',float('nan')):.5f}"
                )

                self.pp_status_var.set(
                    self.pp_status_var.get()
                    + (
                        f" | QGA cost={r.get('qga_cost',float('nan')):.4f}"
                        f" | min clearance={r.get('min_clearance_m',float('nan')):.4f} m"
                        f" | min σ={r.get('min_sigma',float('nan')):.5f}"
                    )
                )

    def pp_compare_all(self):
        if not self._effective_shared_vias():
            messagebox.showinfo(
                "Path Planning",
                "Enter via points in ML Control first."
            )
            return

        self.status_var.set(
            "Comparing Dijkstra, A*, RRT, RRT* and QGA* through the SAME via points..."
        )
        self.update_idletasks()

        self.pp_results = {
            name: self._run_one_planner(
                name
            )
            for name in self.QGA_PLANNERS
        }

        successful = [
            r
            for r in self.pp_results.values()
            if r.get("success")
        ]

        if successful:
            best = min(
                successful,
                key=lambda r: (
                    r["path_length_m"],
                    r["time_s"]
                )
            )

            self.pp_selected_path_name = (
                best["name"]
            )
            self.pp_algo_var.set(
                best["name"]
            )

            qga = self.pp_results.get(
                "QGA*"
            )

            qga_text = ""
            if qga and qga.get(
                "success"
            ):
                qga_text = (
                    f" | QGA*: cost={qga.get('qga_cost',float('nan')):.3f}, "
                    f"clearance={qga.get('min_clearance_m',float('nan')):.3f} m"
                )

            self.pp_status_var.set(
                f"Five planners used the same shared via points. "
                f"Shortest successful route={best['name']} | "
                f"L={best['path_length_m']:.4f} m | "
                f"T={best['time_s']:.4f} s"
                f"{qga_text}"
            )
        else:
            self.pp_selected_path_name = None
            self.pp_status_var.set(
                "No planner completed all shared via points."
            )

        self.pp_refresh_results_tree()
        self.pp_update_figure()

        self.status_var.set(
            "Five-planner comparison complete."
        )

    def pp_refresh_results_tree(self):
        if not hasattr(
            self,
            "pp_tree"
        ):
            return

        for item in self.pp_tree.get_children():
            self.pp_tree.delete(
                item
            )

        for name in self.QGA_PLANNERS:
            if name not in self.pp_results:
                continue

            r = self.pp_results[
                name
            ]

            self.pp_tree.insert(
                "",
                "end",
                iid=name,
                values=(
                    name,
                    (
                        "Yes"
                        if r.get(
                            "success"
                        )
                        else "No"
                    ),
                    f"{float(r.get('time_s',0.0)):.4f}",
                    (
                        f"{float(r.get('path_length_m',float('nan'))):.4f}"
                        if math.isfinite(
                            float(
                                r.get(
                                    "path_length_m",
                                    float("nan")
                                )
                            )
                        )
                        else "—"
                    ),
                    int(
                        r.get(
                            "nodes", 0
                        )
                    ),
                    int(
                        r.get(
                            "points", 0
                        )
                    ),
                    (
                        f"{r.get('via_points_reached',0)}/"
                        f"{r.get('via_points_total',0)}"
                    ),
                    (
                        f"{float(r.get('smoothness_deg',float('nan'))):.2f}"
                        if math.isfinite(
                            float(
                                r.get(
                                    "smoothness_deg",
                                    float("nan")
                                )
                            )
                        )
                        else "—"
                    ),
                )
            )

        if (
            self.pp_selected_path_name
            and self.pp_selected_path_name
            in self.pp_tree.get_children()
        ):
            self.pp_tree.selection_set(
                self.pp_selected_path_name
            )

    def pp_update_figure(self):
        if not hasattr(
            self,
            "pp_axes"
        ):
            return

        axes = self.pp_axes

        for ax in axes:
            ax.clear()

        start = (
            self._planner_head_start()
        )
        vias = (
            self._effective_shared_vias()
        )
        bounds = (
            self._planner_bounds()
        )

        # (a) routes
        ax = axes[0]

        for ox, oy, rr in self.pp_obstacles:
            ax.add_patch(
                Circle(
                    (ox, oy),
                    rr,
                    fill=False,
                    linewidth=1.5
                )
            )

        ax.scatter(
            [start[0]],
            [start[1]],
            marker="s",
            s=60,
            label="HEAD start"
        )

        if vias:
            V = np.asarray(
                vias,
                dtype=float
            )

            ax.scatter(
                V[:,0],
                V[:,1],
                marker="D",
                s=45,
                label="Required shared via points"
            )

            for idx, (
                vx,
                vy
            ) in enumerate(
                vias,
                start=1
            ):
                ax.text(
                    vx,
                    vy,
                    f" V{idx}",
                    fontsize=8
                )

        for name in self.QGA_PLANNERS:
            r = self.pp_results.get(
                name
            )

            if (
                r
                and r.get("success")
                and r.get("path")
            ):
                P = np.asarray(
                    r["path"],
                    dtype=float
                )

                lw = (
                    2.8
                    if name
                    == self.pp_selected_path_name
                    else 1.5
                )

                ax.plot(
                    P[:,0],
                    P[:,1],
                    linewidth=lw,
                    label=name
                )

        ax.set_xlim(
            bounds[0],
            bounds[1]
        )
        ax.set_ylim(
            bounds[2],
            bounds[3]
        )
        ax.set_aspect(
            "equal"
        )
        ax.set_xlabel(
            "World X (m)"
        )
        ax.set_ylabel(
            "World Y (m)"
        )
        ax.set_title(
            "(a) Shared Via Points + Planner Routes"
        )
        ax.grid(
            True,
            alpha=0.25
        )

        handles, labels = (
            ax.get_legend_handles_labels()
        )

        if handles:
            ax.legend(
                handles,
                labels,
                fontsize=7,
                loc="best"
            )

        names = [
            n
            for n in self.QGA_PLANNERS
            if n in self.pp_results
        ]

        if names:
            times = [
                float(
                    self.pp_results[n].get(
                        "time_s",
                        0.0
                    )
                )
                for n in names
            ]

            lengths = [
                (
                    float(
                        self.pp_results[n].get(
                            "path_length_m",
                            0.0
                        )
                    )
                    if self.pp_results[n].get(
                        "success"
                    )
                    else 0.0
                )
                for n in names
            ]

            nodes = [
                int(
                    self.pp_results[n].get(
                        "nodes",
                        0
                    )
                )
                for n in names
            ]

            smooth = [
                (
                    float(
                        self.pp_results[n].get(
                            "smoothness_deg",
                            0.0
                        )
                    )
                    if self.pp_results[n].get(
                        "success"
                    )
                    else 0.0
                )
                for n in names
            ]

            clearances = [
                (
                    float(
                        self.pp_results[n].get(
                            "min_clearance_m",
                            qga_path_min_clearance(
                                self.pp_results[n].get(
                                    "path",
                                    []
                                ),
                                self.pp_obstacles
                            )
                        )
                    )
                    if self.pp_results[n].get(
                        "success"
                    )
                    else 0.0
                )
                for n in names
            ]

            plots = (
                (
                    axes[1],
                    times,
                    "(b) Planning Time",
                    "Time (s)"
                ),
                (
                    axes[2],
                    lengths,
                    "(c) Path Length",
                    "Length (m)"
                ),
                (
                    axes[3],
                    nodes,
                    "(d) Nodes / Expansions",
                    "Count"
                ),
                (
                    axes[4],
                    smooth,
                    "(e) Total Heading Change",
                    "Σ|Δψ| (deg)"
                ),
                (
                    axes[5],
                    clearances,
                    "(f) Minimum Obstacle Clearance",
                    "Clearance (m)"
                ),
            )

            for pax, vals, title, ylabel in plots:
                pax.bar(
                    names,
                    vals
                )
                pax.set_title(
                    title
                )
                pax.set_ylabel(
                    ylabel
                )
                pax.tick_params(
                    axis="x",
                    rotation=18
                )
                pax.grid(
                    True,
                    axis="y",
                    alpha=0.25
                )

        else:
            titles = (
                "(b) Planning Time",
                "(c) Path Length",
                "(d) Nodes / Expansions",
                "(e) Total Heading Change",
                "(f) Minimum Obstacle Clearance",
            )

            for pax, title in zip(
                axes[1:],
                titles
            ):
                pax.text(
                    0.5,
                    0.5,
                    "Run a planner to populate this graph.",
                    ha="center",
                    va="center",
                    transform=pax.transAxes
                )
                pax.set_title(
                    title
                )

        self.pp_fig.suptitle(
            "Dijkstra / A* / RRT / RRT* / QGA* — Same Via Points",
            fontsize=11
        )
        self.pp_fig.tight_layout()
        self.pp_canvas.draw_idle()

    # ------------------------------------------------------------------
    # Dynamic QGA* replanning
    # ------------------------------------------------------------------
    def _resolve_dynamic_replanner(self):
        selected = (
            self.dyn_replanner_var.get()
        )

        if selected != "Active Planner":
            return selected

        try:
            active = (
                self.current_path_planner_var.get()
                .replace(
                    "Path Planner: ",
                    ""
                )
                .strip()
            )
        except Exception:
            active = ""

        # Test QGA* before A* to avoid any ambiguous string interpretation.
        for name in (
            "QGA*",
            "Dijkstra",
            "A*",
            "RRT*",
            "RRT",
        ):
            if active.startswith(
                name
            ):
                return name

        try:
            return (
                self.pp_algo_var.get()
            )
        except Exception:
            return "QGA*"

    def _dynamic_plan_segment(
        self,
        planner_name,
        start,
        goal,
        bounds
    ):
        if planner_name != "QGA*":
            return super()._dynamic_plan_segment(
                planner_name,
                start,
                goal,
                bounds
            )

        return self._qga_star_segment(
            start,
            goal,
            bounds,
            start_heading_rad=float(
                self.base_yaw
            )
        )

    def _plan_dynamic_remaining_route(
        self,
        planner_name,
        current_head,
        remaining_vias
    ):
        if planner_name != "QGA*":
            return super()._plan_dynamic_remaining_route(
                planner_name,
                current_head,
                remaining_vias
            )

        if not remaining_vias:
            return {
                "success": True,
                "path": [
                    current_head
                ],
                "time_s": 0.0,
                "nodes": 0,
                "message": (
                    "No remaining via points."
                ),
            }

        bounds = (
            self._dynamic_planner_bounds(
                current_head,
                remaining_vias
            )
        )

        full_path = [
            (
                float(
                    current_head[0]
                ),
                float(
                    current_head[1]
                )
            )
        ]

        current = (
            full_path[0]
        )
        current_heading = float(
            self.base_yaw
        )

        total_time = 0.0
        total_nodes = 0

        for idx, target in enumerate(
            remaining_vias,
            start=1
        ):
            result = (
                self._qga_star_segment(
                    current,
                    target,
                    bounds,
                    start_heading_rad=current_heading
                )
            )

            total_time += float(
                result.get(
                    "time_s",
                    0.0
                )
            )
            total_nodes += int(
                result.get(
                    "nodes",
                    0
                )
            )

            if not result.get(
                "success"
            ):
                return {
                    "success": False,
                    "path": full_path,
                    "time_s": total_time,
                    "nodes": total_nodes,
                    "message": (
                        f"Dynamic QGA* failed on remaining segment {idx}: "
                        f"{current} → {target}. "
                        f"{result.get('message','')}"
                    ),
                }

            seg = list(
                result["path"]
            )

            if seg:
                full_path.extend(
                    seg[1:]
                )

            current = (
                float(target[0]),
                float(target[1])
            )

            current_heading = float(
                result.get(
                    "final_heading_rad",
                    current_heading
                )
            )

        return {
            "success": True,
            "path": full_path,
            "time_s": total_time,
            "nodes": total_nodes,
            "message": (
                f"Dynamic QGA* route completed through "
                f"{len(remaining_vias)} remaining via point(s)."
            ),
        }

    # ------------------------------------------------------------------
    # 4 ML × 5 planner comparison
    # ------------------------------------------------------------------
    def evaluate_ml_planner_matrix(self):
        if not self.models:
            messagebox.showinfo(
                "ML vs Planners",
                "Train the four ML algorithms first."
            )
            return

        self.status_var.set(
            "Evaluating 4 ML controllers × 5 path planners..."
        )
        self.update_idletasks()

        self.pp_results = {
            name: self._run_one_planner(
                name
            )
            for name in self.QGA_PLANNERS
        }

        self.pp_refresh_results_tree()
        self.pp_update_figure()

        self.comparison_matrix = {}

        for ml_name in self.ML_CONTROLLERS:
            if ml_name not in self.models:
                continue

            for planner_name in self.QGA_PLANNERS:
                key = (
                    ml_name,
                    planner_name
                )

                self.comparison_matrix[
                    key
                ] = (
                    self._evaluate_controller_on_planner(
                        ml_name,
                        self.models[
                            ml_name
                        ],
                        planner_name,
                        self.pp_results[
                            planner_name
                        ]
                    )
                )

        self._refresh_comparison_table()
        self.update_ml_planner_comparison_figure()

        finite = [
            r
            for r in self.comparison_matrix.values()
            if (
                r.get("success")
                and math.isfinite(
                    r.get(
                        "tracking_rmse_m",
                        float("nan")
                    )
                )
                and r.get(
                    "collision_free"
                )
            )
        ]

        if finite:
            best = min(
                finite,
                key=lambda r: (
                    r["tracking_rmse_m"],
                    r["final_error_m"],
                    r["planner_length_m"]
                )
            )

            self.comp_status_var.set(
                f"Best collision-free stack: "
                f"{best['planner']} + {best['ml']} | "
                f"Tracking RMSE={best['tracking_rmse_m']:.4f} m | "
                f"Final error={best['final_error_m']:.4f} m | "
                f"Planner length={best['planner_length_m']:.4f} m"
            )
        else:
            self.comp_status_var.set(
                "Comparison complete, but no stack satisfied all success/collision-free criteria."
            )

        self.status_var.set(
            "4×5 ML-controller / path-planner comparison complete."
        )

    def _refresh_comparison_table(self):
        if not hasattr(
            self,
            "comp_tree"
        ):
            return

        for item in self.comp_tree.get_children():
            self.comp_tree.delete(
                item
            )

        ml = (
            self.comp_ml_var.get()
        )

        for planner in self.QGA_PLANNERS:
            r = self.comparison_matrix.get(
                (
                    ml,
                    planner
                )
            )

            if not r:
                continue

            self.comp_tree.insert(
                "",
                "end",
                values=(
                    f"{planner} + {ml}",
                    (
                        f"{r['planner_length_m']:.4f}"
                        if math.isfinite(
                            r[
                                "planner_length_m"
                            ]
                        )
                        else "—"
                    ),
                    (
                        f"{r['execution_length_m']:.4f}"
                        if math.isfinite(
                            r[
                                "execution_length_m"
                            ]
                        )
                        else "—"
                    ),
                    (
                        f"{r['tracking_rmse_m']:.5f}"
                        if math.isfinite(
                            r[
                                "tracking_rmse_m"
                            ]
                        )
                        else "—"
                    ),
                    (
                        f"{r['final_error_m']:.5f}"
                        if math.isfinite(
                            r[
                                "final_error_m"
                            ]
                        )
                        else "—"
                    ),
                    r["commands"],
                    (
                        f"{r['controller_time_s']:.5f}"
                        if math.isfinite(
                            r[
                                "controller_time_s"
                            ]
                        )
                        else "—"
                    ),
                    (
                        "Yes"
                        if r[
                            "collision_free"
                        ]
                        else "No"
                    ),
                )
            )

    def update_ml_planner_comparison_figure(
        self
    ):
        if not hasattr(
            self,
            "comp_axes"
        ):
            return

        axes = (
            self.comp_axes
        )

        for ax in axes:
            ax.clear()

        ml_names = (
            self.ML_CONTROLLERS
        )
        planners = (
            self.QGA_PLANNERS
        )

        # (a) selected planner route + selected ML followed path.
        ax = axes[0]
        start, goal = (
            self._planner_start_goal()
        )

        for ox, oy, rr in getattr(
            self,
            "pp_obstacles",
            []
        ):
            ax.add_patch(
                Circle(
                    (ox, oy),
                    rr,
                    fill=False,
                    linewidth=1.3
                )
            )

        ax.scatter(
            [start[0]],
            [start[1]],
            marker="s",
            s=55,
            label="HEAD start"
        )

        ax.scatter(
            [goal[0]],
            [goal[1]],
            marker="*",
            s=95,
            label="Final via / goal"
        )

        selected = (
            self.comparison_matrix.get(
                (
                    self.comp_ml_var.get(),
                    self.comp_planner_var.get()
                )
            )
        )

        if selected:
            route = selected.get(
                "route",
                []
            )

            if route:
                P = np.asarray(
                    route,
                    dtype=float
                )

                ax.plot(
                    P[:,0],
                    P[:,1],
                    linewidth=2.0,
                    label=(
                        f"{selected['planner']} route"
                    )
                )

            followed = selected.get(
                "head_path",
                []
            )

            if followed:
                Q = np.asarray(
                    followed,
                    dtype=float
                )

                ax.plot(
                    Q[:,0],
                    Q[:,1],
                    linestyle="--",
                    linewidth=2.0,
                    label=(
                        f"{selected['ml']} followed path"
                    )
                )

        if len(
            self.head_trail
        ) >= 2:
            H = np.asarray(
                self.head_trail,
                dtype=float
            )

            ax.plot(
                H[:,0],
                H[:,1],
                linewidth=1.6,
                label="Actual simulated HEAD path"
            )

        ax.set_title(
            "(a) Planner Route vs ML-Followed / Actual"
        )
        ax.set_xlabel(
            "World X (m)"
        )
        ax.set_ylabel(
            "World Y (m)"
        )
        ax.axis(
            "equal"
        )
        ax.grid(
            True,
            alpha=0.25
        )
        ax.legend(
            fontsize=7
        )

        if not self.comparison_matrix:
            titles = (
                "(b) Planner vs ML-Followed Path Length",
                "(c) Tracking RMSE Matrix",
                "(d) Final HEAD Error Matrix",
                "(e) ML Controller Computation Time",
                "(f) ML Commands Matrix",
            )

            for idx, title in enumerate(
                titles,
                start=1
            ):
                axes[idx].text(
                    0.5,
                    0.5,
                    "Run 4×5 evaluation",
                    ha="center",
                    va="center",
                    transform=axes[idx].transAxes
                )
                axes[idx].set_title(
                    title
                )

            self.comp_fig.suptitle(
                "4 ML Controllers × 5 Path Planners — Including QGA*",
                fontsize=11
            )
            self.comp_fig.tight_layout()
            self.comp_canvas.draw_idle()
            return

        selected_ml = (
            self.comp_ml_var.get()
        )

        route_lengths = []
        execution_lengths = []

        for planner in planners:
            r = self.comparison_matrix.get(
                (
                    selected_ml,
                    planner
                )
            )

            route_lengths.append(
                (
                    r["planner_length_m"]
                    if (
                        r
                        and math.isfinite(
                            r[
                                "planner_length_m"
                            ]
                        )
                    )
                    else 0.0
                )
            )

            execution_lengths.append(
                (
                    r["execution_length_m"]
                    if (
                        r
                        and math.isfinite(
                            r[
                                "execution_length_m"
                            ]
                        )
                    )
                    else 0.0
                )
            )

        x = np.arange(
            len(planners)
        )

        axes[1].bar(
            x-0.18,
            route_lengths,
            width=0.36,
            label="Planner route"
        )

        axes[1].bar(
            x+0.18,
            execution_lengths,
            width=0.36,
            label="ML-followed"
        )

        axes[1].set_xticks(
            x
        )
        axes[1].set_xticklabels(
            planners,
            rotation=18
        )
        axes[1].set_ylabel(
            "Length (m)"
        )
        axes[1].set_title(
            f"(b) Path Length — {selected_ml}"
        )
        axes[1].legend(
            fontsize=7
        )
        axes[1].grid(
            True,
            axis="y",
            alpha=0.25
        )

        def matrix(
            metric,
            fallback=np.nan
        ):
            M = np.full(
                (
                    len(ml_names),
                    len(planners)
                ),
                fallback,
                dtype=float
            )

            for i, ml in enumerate(
                ml_names
            ):
                for j, planner in enumerate(
                    planners
                ):
                    r = self.comparison_matrix.get(
                        (
                            ml,
                            planner
                        )
                    )

                    if not r:
                        continue

                    try:
                        M[i,j] = float(
                            r.get(
                                metric,
                                fallback
                            )
                        )
                    except Exception:
                        pass

            return M

        def heatmap(
            axh,
            M,
            title,
            fmt=".3f"
        ):
            im = axh.imshow(
                M,
                aspect="auto"
            )

            axh.set_xticks(
                range(
                    len(planners)
                )
            )
            axh.set_xticklabels(
                planners,
                rotation=18
            )
            axh.set_yticks(
                range(
                    len(ml_names)
                )
            )
            axh.set_yticklabels(
                ml_names
            )
            axh.set_title(
                title
            )

            for i in range(
                M.shape[0]
            ):
                for j in range(
                    M.shape[1]
                ):
                    val = M[i,j]
                    if math.isfinite(
                        val
                    ):
                        axh.text(
                            j,
                            i,
                            format(
                                val,
                                fmt
                            ),
                            ha="center",
                            va="center",
                            fontsize=6.8
                        )

            self.comp_fig.colorbar(
                im,
                ax=axh,
                fraction=0.046,
                pad=0.04
            )

        heatmap(
            axes[2],
            matrix(
                "tracking_rmse_m"
            ),
            "(c) Tracking RMSE (m)",
            ".4f"
        )

        heatmap(
            axes[3],
            matrix(
                "final_error_m"
            ),
            "(d) Final HEAD Error (m)",
            ".4f"
        )

        heatmap(
            axes[4],
            matrix(
                "controller_time_s"
            ),
            "(e) ML Controller Time (s)",
            ".4f"
        )

        heatmap(
            axes[5],
            matrix(
                "commands"
            ),
            "(f) ML Command Count",
            ".0f"
        )

        self.comp_fig.suptitle(
            "4 ML Controllers × 5 Path Planners — QGA* Included",
            fontsize=11
        )
        self.comp_fig.tight_layout()
        self.comp_canvas.draw_idle()

        self._refresh_comparison_table()

    def save_ml_planner_comparison_csv(
        self
    ):
        if not self.comparison_matrix:
            messagebox.showinfo(
                "Comparison",
                "Run the 4×5 evaluation first."
            )
            return

        path = filedialog.asksaveasfilename(
            title="Save ML vs Planner Comparison",
            defaultextension=".csv",
            filetypes=[
                (
                    "CSV",
                    "*.csv"
                )
            ]
        )

        if not path:
            return

        with open(
            path,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:
            writer = csv.writer(
                f
            )

            writer.writerow([
                "ml_controller",
                "path_planner",
                "success",
                "planner_length_m",
                "ml_followed_length_m",
                "tracking_mae_m",
                "tracking_rmse_m",
                "final_head_error_m",
                "ml_commands",
                "controller_compute_time_s",
                "collision_free",
            ])

            for ml in self.ML_CONTROLLERS:
                for planner in self.QGA_PLANNERS:
                    r = (
                        self.comparison_matrix.get(
                            (
                                ml,
                                planner
                            )
                        )
                    )

                    if not r:
                        continue

                    writer.writerow([
                        ml,
                        planner,
                        int(
                            r["success"]
                        ),
                        r[
                            "planner_length_m"
                        ],
                        r[
                            "execution_length_m"
                        ],
                        r[
                            "tracking_mae_m"
                        ],
                        r[
                            "tracking_rmse_m"
                        ],
                        r[
                            "final_error_m"
                        ],
                        r[
                            "commands"
                        ],
                        r[
                            "controller_time_s"
                        ],
                        int(
                            r[
                                "collision_free"
                            ]
                        ),
                    ])

        self.status_var.set(
            f"4×5 ML vs planner comparison saved: {Path(path).name}"
        )

    # ------------------------------------------------------------------
    # Keep Analysis export unchanged; Fig09/Fig10 now contain QGA* automatically.
    # ------------------------------------------------------------------
    def _refresh_all_analysis_graphs(
        self
    ):
        super()._refresh_all_analysis_graphs()
        self.pp_update_figure()
        self.update_ml_planner_comparison_figure()


# ===========================================================================
# V2.6 — LONG-ROUTE ML FOLLOWER
# ===========================================================================
#
# Problem addressed:
# A fixed global ML-command limit can terminate a long planner route before
# the final via point is reached. In addition, very distant controller targets
# can push ML features outside the range encountered during training.
#
# Refinements:
#   1. Automatic command budget from route length + heading complexity.
#   2. Polyline look-ahead target so ML sees a local target rather than a very
#      distant destination.
#   3. Minimum progress floor for long aligned segments if the ML model predicts
#      an excessively small travel command.
#   4. Conservative removal of duplicate / exactly-collinear route points.
#   5. Mandatory route endpoints and via-point order are preserved.
#
# The path-planning algorithms themselves are unchanged.
# ===========================================================================


def _rx_route_length(points):
    if not points or len(points) < 2:
        return 0.0
    return float(sum(
        math.hypot(
            float(points[i+1][0]) - float(points[i][0]),
            float(points[i+1][1]) - float(points[i][1])
        )
        for i in range(len(points)-1)
    ))


def _rx_total_heading_change_deg(points):
    if not points or len(points) < 3:
        return 0.0

    headings = []
    for i in range(len(points)-1):
        dx = float(points[i+1][0]) - float(points[i][0])
        dy = float(points[i+1][1]) - float(points[i][1])
        if math.hypot(dx, dy) > 1e-12:
            headings.append(math.atan2(dy, dx))

    total = 0.0
    for i in range(len(headings)-1):
        total += abs(math.degrees(
            wrap_angle(headings[i+1] - headings[i])
        ))
    return float(total)


def _rx_clean_route(points, eps=1e-10):
    """
    Remove duplicate and exactly-collinear intermediate points only.

    This is intentionally conservative: obstacle-avoidance bends are not
    shortcut merely to reduce the number of ML targets.
    """
    if not points:
        return []

    pts = [
        (float(p[0]), float(p[1]))
        for p in points
    ]

    # Remove consecutive duplicates.
    dedup = [pts[0]]
    for p in pts[1:]:
        if math.hypot(
            p[0]-dedup[-1][0],
            p[1]-dedup[-1][1]
        ) > 1e-9:
            dedup.append(p)

    if len(dedup) <= 2:
        return dedup

    out = [dedup[0]]
    for i in range(1, len(dedup)-1):
        a = out[-1]
        b = dedup[i]
        c = dedup[i+1]

        abx = b[0]-a[0]
        aby = b[1]-a[1]
        bcx = c[0]-b[0]
        bcy = c[1]-b[1]

        cross = abx*bcy - aby*bcx
        dot = abx*bcx + aby*bcy

        # Remove only when both pieces continue in the same direction and are
        # essentially collinear.
        if abs(cross) <= eps and dot >= 0.0:
            continue

        out.append(b)

    out.append(dedup[-1])
    return out


def _rx_segment_lookahead_target(head_xy, seg_start, seg_end, lookahead):
    """
    Pure-pursuit style target constrained to the current planner segment.

    The HEAD is projected onto the current segment; the target is then placed
    'lookahead' metres farther along that same segment. This avoids presenting
    a long-distance target to the ML model while retaining planner geometry.
    """
    hx, hy = map(float, head_xy)
    ax, ay = map(float, seg_start)
    bx, by = map(float, seg_end)

    vx = bx-ax
    vy = by-ay
    L2 = vx*vx + vy*vy

    if L2 <= 1e-15:
        return (bx, by), 1.0

    seg_len = math.sqrt(L2)

    # Projection parameter of current HEAD on the segment.
    u = ((hx-ax)*vx + (hy-ay)*vy) / L2
    u = min(max(u, 0.0), 1.0)

    du = max(0.0, float(lookahead)) / max(seg_len, 1e-12)
    ut = min(1.0, u + du)

    tx = ax + ut*vx
    ty = ay + ut*vy
    return (tx, ty), u


def _rx_auto_command_budget(
    route,
    max_step,
    max_yaw_deg,
    user_floor=160,
    progress_floor_ratio=0.25,
    safety_factor=1.65,
    reserve=80
):
    """
    Compute a conservative route-dependent command allowance.

    Distance commands are estimated using a conservative effective progress
    significantly below max_step. Turning commands are estimated from total
    route heading change.
    """
    L = _rx_route_length(route)
    heading_deg = _rx_total_heading_change_deg(route)

    max_step = max(0.01, float(max_step))
    max_yaw_deg = max(1.0, float(max_yaw_deg))

    effective_step = max(
        0.012,
        max_step * max(0.15, float(progress_floor_ratio))
    )

    n_translation = L / effective_step
    n_turn = heading_deg / max_yaw_deg

    # Additional allowance for target transitions / local re-alignment.
    n_transition = max(0, len(route)-1) * 2.0

    estimate = math.ceil(
        (n_translation + n_turn + n_transition)
        * max(1.1, float(safety_factor))
        + max(20, int(reserve))
    )

    # No arbitrary small global cap; user value is treated as a minimum.
    return int(max(
        int(user_floor),
        estimate
    ))


# ---------------------------------------------------------------------------
# Refined ML path follower.
#
# This intentionally replaces the global function name used throughout the
# existing GUI. Therefore standard simulation, dynamic replanning and the
# ML × planner comparison all benefit from the long-route correction.
# ---------------------------------------------------------------------------
def plan_ml_waypoints(
    model,
    waypoints,
    start=(0.0, 0.0, 0.0),
    max_step=0.08,
    max_yaw_deg=12.0,
    tolerance=0.035,
    max_commands=160,
    head_offset=0.40,
    turn_threshold_deg=5.0,
    allow_reverse=False,
    lookahead_distance=0.24,
    auto_command_budget=True,
    progress_floor_ratio=0.25,
    budget_safety_factor=1.65,
    budget_reserve=80
):
    """
    Long-route, turn-first, HEAD-tracked ML route follower.

    Key distinction:
        Planner path      = global geometric route.
        Look-ahead target = local ML controller target.

    The local target lies ON the active planner segment, so the ML model is
    not asked to predict from a far-away destination on long routes.
    """
    x, y, yaw = map(float, start)

    route = _rx_clean_route(waypoints)

    commands = []
    body_path = [(x, y)]
    hx, hy = head_xy_from_body(
        x, y, yaw, head_offset
    )
    head_path = [(hx, hy)]

    max_step = max(0.01, float(max_step))
    max_yaw_deg = max(1.0, float(max_yaw_deg))
    tolerance = max(0.005, float(tolerance))
    head_offset = max(0.01, float(head_offset))
    lookahead_distance = max(
        tolerance * 2.0,
        float(lookahead_distance)
    )
    progress_floor_ratio = min(
        max(float(progress_floor_ratio), 0.0),
        0.95
    )

    turn_threshold = math.radians(
        max(0.5, float(turn_threshold_deg))
    )

    # A fixed user value is retained as a minimum budget, not as a hard route-
    # independent ceiling.
    if auto_command_budget:
        command_budget = _rx_auto_command_budget(
            route,
            max_step=max_step,
            max_yaw_deg=max_yaw_deg,
            user_floor=max_commands,
            progress_floor_ratio=progress_floor_ratio,
            safety_factor=budget_safety_factor,
            reserve=budget_reserve
        )
    else:
        command_budget = max(
            1, int(max_commands)
        )

    # If route starts at the body centre (legacy/manual convention), it is only
    # an anchor; do not make the HEAD backtrack to it.
    wp_index = 0
    if route:
        if math.hypot(
            route[0][0]-x,
            route[0][1]-y
        ) <= tolerance:
            wp_index = 1

    # For segment-constrained look-ahead, the preceding route point is the
    # segment start. If the first true target has no predecessor, current HEAD
    # is used as the segment start.
    while (
        wp_index < len(route)
        and len(commands) < command_budget
    ):
        endpoint_x, endpoint_y = route[
            wp_index
        ]

        hx, hy = head_xy_from_body(
            x, y, yaw, head_offset
        )

        endpoint_error = math.hypot(
            endpoint_x-hx,
            endpoint_y-hy
        )

        # Required planner point reached.
        if endpoint_error <= tolerance:
            wp_index += 1
            continue

        if wp_index > 0:
            seg_start = route[
                wp_index-1
            ]
        else:
            seg_start = (
                hx, hy
            )

        pursuit_target, projection_u = (
            _rx_segment_lookahead_target(
                (hx, hy),
                seg_start,
                (endpoint_x, endpoint_y),
                lookahead_distance
            )
        )

        tx, ty = pursuit_target

        # If the HEAD has progressed to the end of a segment, explicitly target
        # the route endpoint so mandatory via points remain respected.
        if projection_u >= 0.985:
            tx, ty = (
                endpoint_x,
                endpoint_y
            )

        local_error = math.hypot(
            tx-hx,
            ty-hy
        )

        # Controller features remain local even on a multi-metre global route.
        desired_yaw = math.atan2(
            ty-hy,
            tx-hx
        )
        yaw_error = wrap_angle(
            desired_yaw-yaw
        )

        f, _, _ = forward_only_features(
            x, y, yaw, tx, ty
        )

        pred = np.asarray(
            model.predict(
                f.reshape(1, -1)
            )
        ).reshape(-1)

        predicted_travel = float(
            np.clip(
                pred[0],
                0.005,
                max_step
            )
        )

        # Prevent a poorly conditioned / out-of-distribution prediction from
        # spending hundreds of commands on millimetre-scale progress.
        progress_floor = min(
            max_step,
            max(
                0.012,
                max_step
                * progress_floor_ratio
            )
        )

        if local_error > max(
            2.5*tolerance,
            progress_floor
        ):
            ml_travel = max(
                predicted_travel,
                progress_floor
            )
        else:
            ml_travel = predicted_travel

        q2_amp = float(
            np.clip(
                pred[2],
                8.0,
                32.0
            )
        )
        q3_lift = float(
            np.clip(
                pred[3],
                16.0,
                40.0
            )
        )

        # ---------------------------------------------------------------
        # Optional reverse mode retained for experimental compatibility.
        # ---------------------------------------------------------------
        if allow_reverse:
            c = math.cos(yaw)
            s = math.sin(yaw)
            ex_h = (
                c*(tx-hx)
                + s*(ty-hy)
            )

            if (
                ex_h < 0.0
                and abs(yaw_error)
                > math.pi/2
            ):
                travel = min(
                    ml_travel,
                    local_error
                )

                nx = (
                    x
                    - math.cos(yaw)*travel
                )
                ny = (
                    y
                    - math.sin(yaw)*travel
                )
                nyaw = yaw

                commands.append({
                    "target_index": wp_index,
                    "direction": "Backward",
                    "travel_m": travel,
                    "yaw_step_deg": 0.0,
                    "q2_amp_deg": q2_amp,
                    "q3_lift_deg": q3_lift,
                    "start_x": x,
                    "start_y": y,
                    "start_yaw": yaw,
                    "end_x": nx,
                    "end_y": ny,
                    "end_yaw": nyaw,
                    "target_x": tx,
                    "target_y": ty,
                    "route_endpoint_x": endpoint_x,
                    "route_endpoint_y": endpoint_y,
                    "error_before_m": endpoint_error,
                    "local_target_error_m": local_error,
                    "head_start_x": hx,
                    "head_start_y": hy,
                    "pivot_x": hx,
                    "pivot_y": hy,
                    "lookahead_target": 1,
                })

                x, y, yaw = nx, ny, nyaw
                body_path.append(
                    (x, y)
                )
                head_path.append(
                    head_xy_from_body(
                        x, y, yaw,
                        head_offset
                    )
                )
                continue

        # ---------------------------------------------------------------
        # TURN FIRST
        # ---------------------------------------------------------------
        if abs(yaw_error) > turn_threshold:
            yaw_step_deg = float(
                np.clip(
                    math.degrees(yaw_error),
                    -max_yaw_deg,
                    +max_yaw_deg
                )
            )

            nyaw = wrap_angle(
                yaw
                + math.radians(
                    yaw_step_deg
                )
            )

            # Pivot around current HEAD.
            nx = (
                hx
                - math.cos(nyaw)
                * head_offset
            )
            ny = (
                hy
                - math.sin(nyaw)
                * head_offset
            )

            direction = (
                "Turn Left"
                if yaw_step_deg > 0.0
                else "Turn Right"
            )

            commands.append({
                "target_index": wp_index,
                "direction": direction,
                "travel_m": 0.0,
                "yaw_step_deg": yaw_step_deg,
                "q2_amp_deg": max(
                    12.0,
                    q2_amp*0.80
                ),
                "q3_lift_deg": q3_lift,
                "start_x": x,
                "start_y": y,
                "start_yaw": yaw,
                "end_x": nx,
                "end_y": ny,
                "end_yaw": nyaw,
                "target_x": tx,
                "target_y": ty,
                "route_endpoint_x": endpoint_x,
                "route_endpoint_y": endpoint_y,
                "error_before_m": endpoint_error,
                "local_target_error_m": local_error,
                "head_start_x": hx,
                "head_start_y": hy,
                "pivot_x": hx,
                "pivot_y": hy,
                "lookahead_target": 1,
            })

            x, y, yaw = (
                nx, ny, nyaw
            )
            body_path.append(
                (x, y)
            )
            head_path.append(
                (hx, hy)
            )
            continue

        # ---------------------------------------------------------------
        # FORWARD PROGRESS
        # ---------------------------------------------------------------
        travel = min(
            ml_travel,
            local_error
        )

        # If the pursuit target is extremely close while the actual planner
        # endpoint remains far away, do not emit a near-zero translation.
        if (
            travel < 0.006
            and endpoint_error
            > tolerance*2.0
        ):
            travel = min(
                progress_floor,
                endpoint_error
            )

        nx = (
            x
            + math.cos(yaw)*travel
        )
        ny = (
            y
            + math.sin(yaw)*travel
        )
        nyaw = yaw

        commands.append({
            "target_index": wp_index,
            "direction": "Forward",
            "travel_m": travel,
            "yaw_step_deg": 0.0,
            "q2_amp_deg": q2_amp,
            "q3_lift_deg": q3_lift,
            "start_x": x,
            "start_y": y,
            "start_yaw": yaw,
            "end_x": nx,
            "end_y": ny,
            "end_yaw": nyaw,
            "target_x": tx,
            "target_y": ty,
            "route_endpoint_x": endpoint_x,
            "route_endpoint_y": endpoint_y,
            "error_before_m": endpoint_error,
            "local_target_error_m": local_error,
            "head_start_x": hx,
            "head_start_y": hy,
            "pivot_x": hx,
            "pivot_y": hy,
            "lookahead_target": 1,
        })

        x, y, yaw = (
            nx, ny, nyaw
        )
        body_path.append(
            (x, y)
        )
        head_path.append(
            head_xy_from_body(
                x, y, yaw,
                head_offset
            )
        )

    reached = (
        wp_index >= len(route)
    )

    hx, hy = head_xy_from_body(
        x, y, yaw,
        head_offset
    )

    final_error = 0.0
    if route:
        final_error = math.hypot(
            route[-1][0]-hx,
            route[-1][1]-hy
        )

    return {
        "commands": commands,
        "path": head_path,
        "head_path": head_path,
        "body_path": body_path,
        "reached": reached,
        "final_error_m": final_error,
        "final_pose": (
            x, y, yaw
        ),
        "final_head": (
            hx, hy
        ),
        "waypoints_reached": wp_index,
        "head_offset_m": head_offset,
        "controller_route": route,
        "route_length_m": _rx_route_length(
            route
        ),
        "route_heading_change_deg":
            _rx_total_heading_change_deg(
                route
            ),
        "command_budget": int(
            command_budget
        ),
        "commands_used": len(
            commands
        ),
        "lookahead_distance_m":
            float(
                lookahead_distance
            ),
        "auto_command_budget":
            bool(
                auto_command_budget
            ),
        "progress_floor_m":
            float(
                max(
                    0.012,
                    max_step
                    * progress_floor_ratio
                )
            ),
    }


class MLRobotDogAppV26(MLRobotDogAppV25):
    """
    ROBOQUAD-X Studio V2.6

    Long-route refinement without changing:
      - Dijkstra / A* / RRT / RRT* / QGA*
      - ML algorithm set
      - dynamic obstacle system
      - joint / gait / singularity / workspace analytics
    """

    def _ensure_long_route_vars(self):
        if hasattr(
            self,
            "long_route_auto_budget_var"
        ):
            return

        self.long_route_auto_budget_var = (
            tk.BooleanVar(
                value=True
            )
        )
        self.long_route_lookahead_var = (
            tk.DoubleVar(
                value=0.24
            )
        )
        self.long_route_progress_floor_var = (
            tk.DoubleVar(
                value=0.25
            )
        )
        self.long_route_safety_var = (
            tk.DoubleVar(
                value=1.65
            )
        )
        self.long_route_reserve_var = (
            tk.IntVar(
                value=80
            )
        )
        self.long_route_status_var = (
            tk.StringVar(
                value=(
                    "Long-route mode: automatic command budget + "
                    "segment look-ahead enabled."
                )
            )
        )

    def __init__(self):
        super().__init__()

        self.title(
            "ROBOQUAD-X Studio V2.6 — Long-Route Intelligent Quadruped Navigation"
        )

        self.status_var.set(
            "V2.6 ready: long planner routes use adaptive ML command budgeting and local look-ahead."
        )

    # ------------------------------------------------------------------
    # Add controls while preserving the existing horizontal-tab GUI.
    # ------------------------------------------------------------------
    def _build_control_tab(
        self,
        parent
    ):
        self._ensure_long_route_vars()

        super()._build_control_tab(
            parent
        )

        root = None
        try:
            scroll = parent.winfo_children()[0]
            root = scroll.inner
        except Exception:
            root = parent

        box = ttk.LabelFrame(
            root,
            text="Long-Route ML Follower",
            padding=8
        )
        box.pack(
            fill="x",
            padx=8,
            pady=4
        )

        ttk.Checkbutton(
            box,
            text=(
                "Automatic command budget "
                "(recommended)"
            ),
            variable=(
                self.long_route_auto_budget_var
            )
        ).grid(
            row=0,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(0, 5)
        )

        fields = (
            (
                "Look-ahead distance (m)",
                self.long_route_lookahead_var
            ),
            (
                "Minimum progress ratio",
                self.long_route_progress_floor_var
            ),
            (
                "Budget safety factor",
                self.long_route_safety_var
            ),
            (
                "Extra command reserve",
                self.long_route_reserve_var
            ),
        )

        for r, (
            label,
            var
        ) in enumerate(
            fields,
            start=1
        ):
            ttk.Label(
                box,
                text=label
            ).grid(
                row=r,
                column=0,
                sticky="w",
                pady=2
            )
            ttk.Entry(
                box,
                textvariable=var,
                width=10
            ).grid(
                row=r,
                column=1,
                sticky="w",
                padx=4,
                pady=2
            )

        ttk.Label(
            box,
            text=(
                "The existing 'Max ML commands' value is now a minimum floor "
                "when automatic budgeting is enabled. Long routes receive the "
                "commands they require instead of stopping at 160."
            ),
            wraplength=700,
            justify="left"
        ).grid(
            row=1,
            column=2,
            rowspan=3,
            sticky="nw",
            padx=(16, 0)
        )

        ttk.Label(
            box,
            textvariable=(
                self.long_route_status_var
            ),
            style="Sub.TLabel",
            wraplength=1000,
            justify="left"
        ).grid(
            row=5,
            column=0,
            columnspan=3,
            sticky="w",
            pady=(6, 0)
        )

        box.columnconfigure(
            2,
            weight=1
        )

    # ------------------------------------------------------------------
    # Use the user-selectable long-route parameters in ordinary planning.
    # Other existing calls (dynamic replanning and comparison) automatically
    # use the refined global function defaults.
    # ------------------------------------------------------------------
    def plan_path(
        self,
        simulate=False
    ):
        model, model_name = (
            self._active_ml_model()
        )

        if model is None:
            messagebox.showinfo(
                "ML Controller",
                "Train the ML algorithms or load a model first."
            )
            return

        if not self.waypoints:
            messagebox.showerror(
                "ML Controller",
                "Define or transfer at least one HEAD waypoint."
            )
            return

        try:
            start = (
                float(
                    self.start_x_var.get()
                ),
                float(
                    self.start_y_var.get()
                ),
                math.radians(
                    float(
                        self.start_yaw_var.get()
                    )
                ),
            )

            allow_reverse = (
                self.path_policy_var.get()
                ==
                "Allow Automatic Reverse"
            )

            self.plan = plan_ml_waypoints(
                model,
                self.waypoints,
                start,
                self.max_step_var.get(),
                self.max_yaw_var.get(),
                self.tolerance_var.get(),
                self.max_commands_var.get(),
                head_offset=(
                    self._head_offset()
                ),
                turn_threshold_deg=(
                    self.turn_threshold_var.get()
                ),
                allow_reverse=(
                    allow_reverse
                ),
                lookahead_distance=(
                    self.long_route_lookahead_var.get()
                ),
                auto_command_budget=(
                    self.long_route_auto_budget_var.get()
                ),
                progress_floor_ratio=(
                    self.long_route_progress_floor_var.get()
                ),
                budget_safety_factor=(
                    self.long_route_safety_var.get()
                ),
                budget_reserve=(
                    self.long_route_reserve_var.get()
                )
            )

            cmds = self.plan[
                "commands"
            ]

            nf = sum(
                c["direction"]
                == "Forward"
                for c in cmds
            )
            nb = sum(
                c["direction"]
                == "Backward"
                for c in cmds
            )
            nt = sum(
                c["direction"].startswith(
                    "Turn"
                )
                for c in cmds
            )

            budget = self.plan[
                "command_budget"
            ]
            used = self.plan[
                "commands_used"
            ]

            completion = (
                100.0
                if self.plan[
                    "reached"
                ]
                else (
                    100.0
                    * self.plan[
                        "waypoints_reached"
                    ]
                    / max(
                        1,
                        len(
                            self.plan[
                                "controller_route"
                            ]
                        )
                    )
                )
            )

            if self.plan[
                "reached"
            ]:
                termination = (
                    "FULL ROUTE REACHED"
                )
            elif used >= budget:
                termination = (
                    "ADAPTIVE BUDGET EXHAUSTED"
                )
            else:
                termination = (
                    "ROUTE INCOMPLETE"
                )

            self.path_var.set(
                f"ML controller [{model_name}] | "
                f"Route={self.route_source_var.get().replace('Route source: ', '')} | "
                f"{termination} | "
                f"commands={used}/{budget} | "
                f"turn={nt}, forward={nf}, backward={nb} | "
                f"HEAD final error={self.plan['final_error_m']:.4f} m"
            )

            self.long_route_status_var.set(
                f"Route length={self.plan['route_length_m']:.3f} m | "
                f"Heading change={self.plan['route_heading_change_deg']:.1f}° | "
                f"Look-ahead={self.plan['lookahead_distance_m']:.3f} m | "
                f"Progress floor={self.plan['progress_floor_m']:.3f} m | "
                f"Command budget={budget} | "
                f"Used={used} | "
                f"Completion={completion:.1f}%"
            )

            self.active_ml_algorithm_var.set(
                f"Active ML controller: {model_name}"
            )
            self._refresh_algorithm_identity()

            self.status_var.set(
                (
                    "Long-route ML command sequence generated successfully."
                    if self.plan["reached"]
                    else
                    "Long-route ML sequence generated but the route remains incomplete; "
                    "inspect the adaptive-budget status."
                )
            )

            self._update_path_plot()
            self._update_3d()
            self.update_planning_plot()

            if simulate:
                self.start_sim()

        except Exception as exc:
            messagebox.showerror(
                "Long-Route ML Controller",
                str(exc)
            )


if __name__ == "__main__":
    app = MLRobotDogAppV26()
    app.mainloop()
