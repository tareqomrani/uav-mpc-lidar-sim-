# app.py — UAV LiDAR + MPC Lab (Full + Playback + Optimized + Dynamic Suggestions)
# - MPC dynamics, wind/gusts, moving obstacles
# - LiDAR beams + fused hits + COVERAGE ALONG BEAMS (new)
# - Battery/RTB, arrival speed & time-to-goal
# - Loiter at goal (circle) or damped hover (early stop)
# - Playback mode (step-by-step) + Fast mode + LineCollection beams
# - Metrics (latency, min separation, speeds) + Dynamic AI Suggestions (new)
# - Exports: PNG / CSVs / JSON / GeoJSON / ZIP

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import pandas as pd
import time, io, json, zipfile
from datetime import datetime

# ------------------------------
# Utils
# ------------------------------
def clip_norm(vec, max_mag):
    n = np.linalg.norm(vec)
    if n > max_mag and n > 1e-12:
        return (vec / n) * max_mag
    return vec

def rand_color(i):
    colors = ["tab:red","tab:blue","tab:green","tab:orange","tab:purple",
              "tab:brown","tab:pink","tab:olive","tab:cyan","tab:gray"]
    return colors[i % len(colors)]

def min_separation(drones):
    vals = []
    for i in range(len(drones)):
        for j in range(i+1, len(drones)):
            vals.append(np.linalg.norm(drones[i].p - drones[j].p))
    return (min(vals) if vals else None)

def to_geojson(drones):
    feats = []
    for d in drones:
        coords = [[float(x), float(y)] for (x, y) in d.path]
        feats.append({
            "type": "Feature",
            "properties": {"id": int(d.id)},
            "geometry": {"type": "LineString", "coordinates": coords}
        })
    return json.dumps({"type": "FeatureCollection", "features": feats}, indent=2)

# ------------------------------
# Moving Obstacles
# ------------------------------
class MovingObstacle:
    def __init__(self, x, y, vx, vy, size=1.0):
        self.p = np.array([float(x), float(y)], dtype=float)
        self.v = np.array([float(vx), float(vy)], dtype=float)
        self.size = float(size)

    def step(self, dt, bounds):
        self.p += self.v * dt
        half = self.size / 2
        for k in (0, 1):
            if self.p[k] < half:
                self.p[k] = half
                self.v[k] *= -1
            if self.p[k] > bounds - half:
                self.p[k] = bounds - half
                self.v[k] *= -1

# ------------------------------
# UAV (MPC + Hover/Loiter + Battery/RTB + Arrival metrics)
# ------------------------------
class UAV:
    def __init__(self, x, y, goal, uid, color, dt, max_speed, max_accel, goal_radius,
                 energy_Wh, home):
        self.p = np.array([float(x), float(y)], dtype=float)
        self.v = np.zeros(2, dtype=float)
        self.goal = np.array(goal, dtype=float)
        self.home = np.array(home, dtype=float)
        self.id = uid
        self.color = color
        self.dt = dt
        self.max_speed = max_speed
        self.max_accel = max_accel
        self.goal_radius = goal_radius
        self.path = [tuple(self.p)]
        self.vel_hist = [tuple(self.v)]
        self.speed_hist = [0.0]
        self.energy_Wh = float(energy_Wh)
        self.rtb = False
        self.at_goal = False

        # Arrival metrics
        self.time_to_goal = None
        self.speed_at_arrival = None
        self._arrived_recorded = False

    def lidar_ray_cast(self, obstacles, max_range, n_rays=24):
        """Return list of beam segments [(a,b), ...] and list of hit points."""
        hits, beams = [], []
        angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        step = 0.35  # beam marching step (perf/quality trade)
        for ang in angles:
            dvec = np.array([np.cos(ang), np.sin(ang)], float)
            r = 0.0
            hit_point = None
            while r < max_range:
                sample = self.p + dvec * r
                for ob in obstacles:
                    half = ob.size/2
                    if (abs(sample[0]-ob.p[0]) <= half) and (abs(sample[1]-ob.p[1]) <= half):
                        hit_point = sample.copy()
                        break
                if hit_point is not None:
                    break
                r += step
            end = self.p + dvec * min(r, max_range)
            beams.append((self.p.copy(), end))
            if hit_point is not None:
                hits.append(hit_point)
        return beams, hits

    def _rollout(self, a, others, obstacles, H, lidar_radius, wind):
        dt = self.dt
        p = self.p.copy()
        v = self.v.copy()
        cost = 0.0
        for _ in range(H):
            v = clip_norm(v + a*dt, self.max_speed)
            p = p + (v + wind) * dt
            # goal tracking
            cost += np.linalg.norm(p - self.goal)**2
            # obstacles
            for ob in obstacles:
                outside = np.maximum(np.abs(p - ob.p) - ob.size/2, 0.0)
                dist = np.linalg.norm(outside)
                if dist < 0.5:        cost += 400.0
                elif dist < lidar_radius: cost += 2.0 / (dist + 1e-3)
            # inter-drone separation
            for od in others:
                if od.id == self.id: continue
                d = np.linalg.norm(p - od.p)
                if d < 0.8:        cost += 400.0
                elif d < 2.5:      cost += 7.0 / (d + 1e-3)
        # control effort
        cost += 0.1 * np.dot(a, a)
        return cost

    def mpc_step(self, drones, obstacles, H=3, lidar_radius=6.0, damping=0.6, wind=np.zeros(2),
                 loiter=False, loiter_speed=0.6, loiter_radius=1.5):

        # RTB trigger at 15% of initial energy
        if (not self.rtb) and self.energy_Wh <= 0.15 * self.energy_Wh_init:
            self.rtb = True
            self.goal = self.home.copy()

        # within goal radius: hover or loiter
        if np.linalg.norm(self.p - self.goal) <= self.goal_radius:
            self.at_goal = True

            # record arrival metrics ONCE, before damping/loiter
            if not self._arrived_recorded:
                self._arrived_recorded = True
                self.time_to_goal = getattr(self, "_global_time", 0.0)
                self.speed_at_arrival = float(np.linalg.norm(self.v))

            if loiter:
                # Loiter: orbit goal with tangential velocity + radial correction
                r = self.p - self.goal
                r_norm = np.linalg.norm(r)
                if r_norm < 1e-6:
                    r = np.array([loiter_radius, 0.0], dtype=float)
                    r_norm = loiter_radius
                t_hat = np.array([-r[1], r[0]], dtype=float) / r_norm  # CCW tangent
                v_tangent = loiter_speed * t_hat
                radial_err = r_norm - loiter_radius
                k_r = 1.5
                v_rad = -k_r * radial_err * (r / r_norm)
                v_des = v_tangent + v_rad
                self.v = clip_norm(0.5 * self.v + 0.5 * v_des, self.max_speed)
                self.p = self.p + (self.v + wind) * self.dt
            else:
                # Damped hover
                self.v *= float(damping)
                self.p = self.p + (self.v + wind) * self.dt

            self._log_state()
            return

        # MPC step
        acc = self.max_accel
        candidates = [np.array([ax, ay], float) for ax in [-acc, 0.0, acc] for ay in [-acc, 0.0, acc]]
        best_a, best_cost = np.zeros(2), float("inf")
        for a in candidates:
            c = self._rollout(a, drones, obstacles, H, lidar_radius, wind)
            if c < best_cost:
                best_cost, best_a = c, a

        self.v = clip_norm(self.v + best_a * self.dt, self.max_speed)
        self.p = self.p + (self.v + wind) * self.dt
        self._log_state()

    def _log_state(self):
        self.path.append(tuple(self.p))
        self.vel_hist.append(tuple(self.v))
        sp = float(np.linalg.norm(self.v))
        self.speed_hist.append(sp)
        power_W = 20.0 + 8.0 * (sp ** 3)  # simple power model
        self.energy_Wh = max(0.0, self.energy_Wh - power_W * self.dt / 3600.0)

    def set_energy_init(self):
        self.energy_Wh_init = self.energy_Wh

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="UAV LiDAR + MPC Lab", layout="wide")
st.title("🛰️ UAV LiDAR + MPC Lab")

# Presets
preset = st.sidebar.selectbox("Scenario preset", ["None", "Warehouse Scan", "Urban Canyon", "Disaster Mapping"])
st.sidebar.write("---")

# Performance
fast_mode = st.sidebar.checkbox("Fast mode (optimize plotting)", True)
max_hit_points = st.sidebar.slider("Max plotted LiDAR hits", 1000, 20000, 6000, 500)

# Playback
playback = st.sidebar.checkbox("Enable Playback (step-by-step)", False)
play_delay = st.sidebar.slider("Playback delay per frame (s)", 0.01, 0.5, 0.08, 0.01)

# Controls
grid_size = st.sidebar.slider("World Size", 20, 100, 40)
steps = st.sidebar.slider("Simulation Steps (max)", 20, 600, 200)
dt = st.sidebar.slider("Δt (s)", 0.05, 1.0, 0.2, 0.05)
horizon = st.sidebar.slider("MPC Horizon H", 1, 8, 3)
max_speed = st.sidebar.slider("Max Speed (u/s)", 0.5, 12.0, 4.0, 0.1)
max_accel = st.sidebar.slider("Max Accel (u/s²)", 0.2, 8.0, 2.0, 0.1)
goal_radius = st.sidebar.slider("Goal Radius ε (u)", 0.1, 3.0, 0.7, 0.1)
damping = st.sidebar.slider("Hover Damping (0..1)", 0.2, 0.95, 0.6, 0.05)

wx = st.sidebar.slider("Wind X (u/s)", -3.0, 3.0, 0.0, 0.1)
wy = st.sidebar.slider("Wind Y (u/s)", -3.0, 3.0, 0.0, 0.1)
gust = st.sidebar.slider("Gust σ (u/s)", 0.0, 2.0, 0.3, 0.1)

lidar_radius = st.sidebar.slider("LiDAR Range (u)", 2.0, 20.0, 8.0, 0.5)
lidar_rays = st.sidebar.slider("LiDAR Rays / Drone", 8, 96, 36, 1)
show_beams = st.sidebar.checkbox("Show LiDAR beams", True)

num_drones = st.sidebar.slider("Drones", 1, 10, 4)
num_obstacles = st.sidebar.slider("Moving Obstacles", 0, 30, 10)
ob_size = st.sidebar.slider("Obstacle Size (side)", 0.6, 5.0, 1.2, 0.1)
ob_speed = st.sidebar.slider("Obstacle Speed (u/s)", 0.0, 5.0, 1.3, 0.1)
energy_Wh = st.sidebar.slider("Energy per drone (Wh)", 10.0, 200.0, 80.0, 1.0)

coverage_cell = st.sidebar.slider("Coverage cell size (u)", 0.5, 4.0, 1.0, 0.1)
seed = st.sidebar.number_input("Random seed", value=7, step=1)

# Loiter mode
loiter_enabled = st.sidebar.checkbox("Loiter at goal (circle)", True)
loiter_speed = st.sidebar.slider("Loiter speed at goal (u/s)", 0.1, 3.0, 0.6, 0.1)
loiter_radius = st.sidebar.slider("Loiter radius at goal (u)", 0.5, 5.0, 1.5, 0.1)

rng = np.random.default_rng(int(seed))

# Scenario preset obstacles
preset_obstacles = []
if preset == "Warehouse Scan":
    num_obstacles = max(num_obstacles, 14)
    ob_size = max(ob_size, 1.6)
    for x in np.linspace(8, grid_size - 8, 6):
        preset_obstacles.append(MovingObstacle(x, grid_size / 2, 0, 0, ob_size))
elif preset == "Urban Canyon":
    num_obstacles = max(num_obstacles, 16)
    ob_size = max(ob_size, 2.4)
    for x in np.linspace(6, grid_size - 6, 5):
        preset_obstacles.append(MovingObstacle(x, grid_size * 0.35, 0, 0, ob_size))
        preset_obstacles.append(MovingObstacle(x, grid_size * 0.65, 0, 0, ob_size))
elif preset == "Disaster Mapping":
    num_obstacles = max(num_obstacles, 10)
    ob_speed = max(ob_speed, 1.6)

# Obstacles (preset + random movers)
obstacles = preset_obstacles[:]
for _ in range(max(0, num_obstacles - len(preset_obstacles))):
    x = float(rng.uniform(0.5, grid_size - 0.5))
    y = float(rng.uniform(0.5, grid_size - 0.5))
    ang = rng.uniform(0, 2 * np.pi)
    vx, vy = ob_speed * np.cos(ang), ob_speed * np.sin(ang)
    obstacles.append(MovingObstacle(x, y, vx, vy, size=ob_size))

# Starts & goals
starts, goals = [], []
for _ in range(num_drones):
    starts.append((float(rng.uniform(1.0, grid_size - 1.0)), float(rng.uniform(1.0, grid_size - 1.0))))
    goals.append((float(rng.uniform(1.0, grid_size - 1.0)), float(rng.uniform(1.0, grid_size - 1.0))))

# Greedy task allocation (assign nearest unassigned goal)
assigned = set()
assigned_goals = []
for s in starts:
    dists = [(j, np.linalg.norm(np.array(s) - np.array(goals[j]))) for j in range(len(goals)) if j not in assigned]
    jmin = min(dists, key=lambda x: x[1])[0]
    assigned.add(jmin)
    assigned_goals.append(goals[jmin])
goals = assigned_goals

# Drones
drones = []
for i in range(num_drones):
    u = UAV(starts[i][0], starts[i][1], goals[i], i, rand_color(i), dt,
            max_speed, max_accel, goal_radius, energy_Wh, home=starts[i])
    u.set_energy_init()
    drones.append(u)

# Coverage grid setup
cols = int(np.ceil(grid_size / coverage_cell))
rows = int(np.ceil(grid_size / coverage_cell))
covered = set()

# Logs
fused_points = []     # (x, y, step, time)
traj_rows = []        # state time series
mpc_times = []        # per-drone per-step latency
sim_time = 0.0
post_arrival_grace = 5  # steps of loiter after all arrived (for playback)

# Plot container for playback
frame_container = st.empty()

def render_frame():
    """Draw current frame using fast-mode decimation and batched beams."""
    fig, ax = plt.subplots(figsize=(8.2, 8.2))
    ax.set_xlim(0, grid_size); ax.set_ylim(0, grid_size)
    ax.set_title("LiDAR + MPC — Loiter / Hover")

    # Fused hits (decimated)
    if len(fused_points) > 0:
        if fast_mode and len(fused_points) > max_hit_points:
            idx = np.random.choice(len(fused_points), size=max_hit_points, replace=False)
            fp = np.array([fused_points[i] for i in idx])
        else:
            fp = np.array(fused_points)
        ax.scatter(fp[:, 0], fp[:, 1], s=(3 if fast_mode else 6),
                   alpha=(0.18 if fast_mode else 0.22), marker='.', label="Fused LiDAR hits")

    # Obstacles
    for ob in obstacles:
        ax.add_patch(patches.Rectangle((ob.p[0] - ob.size/2, ob.p[1] - ob.size/2),
                                       ob.size, ob.size, color="black"))

    # Batched beams (faster)
    if show_beams and any(hasattr(d, "_last_beams") for d in drones):
        segments, colors = [], []
        for d in drones:
            if hasattr(d, "_last_beams"):
                for a, b in d._last_beams:
                    segments.append([(a[0], a[1]), (b[0], b[1])])
                    colors.append(d.color)
        if segments:
            lc = LineCollection(segments, linewidths=0.6, alpha=0.22, colors=colors)
            ax.add_collection(lc)

    # Drones & paths
    for d in drones:
        xs, ys = zip(*d.path)
        if fast_mode and len(xs) > 400:
            step_dec = max(1, len(xs) // 400)
            xs = xs[::step_dec]; ys = ys[::step_dec]
        ax.plot(xs, ys, linestyle="--", color=d.color, alpha=0.9)
        ax.scatter(d.p[0], d.p[1], color=d.color, s=90, marker="o", label=f"Drone {d.id}")
        ax.scatter(d.goal[0], d.goal[1], color=d.color, marker="*", s=180, edgecolor="k")
        circ = patches.Circle((d.goal[0], d.goal[1]), d.goal_radius, fill=False, linestyle=":", alpha=0.6)
        ax.add_patch(circ)

    ax.legend(loc="upper right")
    return fig

# ------------------------------
# Simulation loop (supports playback)
# ------------------------------
for t in range(steps):
    wind = np.array([wx, wy]) + np.random.normal(0, gust, size=2)

    # LiDAR & coverage (before move)
    for d in drones:
        beams, hits = d.lidar_ray_cast(obstacles, max_range=lidar_radius, n_rays=lidar_rays)
        d._last_beams, d._last_hits = beams, hits

        # NEW: mark coverage ALONG EVERY BEAM (not just obstacle hits)
        for a, b in beams:
            seg = np.array(b) - np.array(a)
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                pts = [np.array(a)]
            else:
                step_len = max(coverage_cell * 0.5, 0.3)
                nsteps = int(np.ceil(seg_len / step_len))
                tvals = np.linspace(0.0, 1.0, nsteps + 1)
                pts = [np.array(a) + t_ * seg for t_ in tvals]
            for pnt in pts:
                xh, yh = float(pnt[0]), float(pnt[1])
                if 0.0 <= xh <= grid_size and 0.0 <= yh <= grid_size:
                    cx = int(np.clip(xh // coverage_cell, 0, cols - 1))
                    cy = int(np.clip(yh // coverage_cell, 0, rows - 1))
                    covered.add((cx, cy))

        # keep fused map of obstacle hits (for viz/export)
        for h in hits:
            fused_points.append((float(h[0]), float(h[1]), t, sim_time))

    # MPC + dynamics (with latency profiling)
    for d in drones:
        d._global_time = sim_time  # for arrival capture
        t0 = time.perf_counter()
        d.mpc_step(drones, obstacles, H=horizon, lidar_radius=lidar_radius,
                   damping=damping, wind=wind,
                   loiter=loiter_enabled, loiter_speed=loiter_speed, loiter_radius=loiter_radius)
        t1 = time.perf_counter()
        mpc_times.append({"step": t, "drone_id": d.id, "latency_s": t1 - t0})

        x, y = d.path[-1]
        vx, vy = d.vel_hist[-1]
        traj_rows.append({
            "step": t, "time_s": sim_time + dt, "drone_id": d.id,
            "x": float(x), "y": float(y), "vx": float(vx), "vy": float(vy),
            "speed": d.speed_hist[-1], "energy_Wh": d.energy_Wh, "rtb": d.rtb
        })

    # Move obstacles
    for ob in obstacles:
        ob.step(dt, grid_size)

    sim_time += dt

    # Early stop behavior
    if all(dr.at_goal for dr in drones):
        if not playback and not loiter_enabled:
            break  # normal early stop
        # playback or loiter enabled: brief grace window then stop
        post_arrival_grace -= 1
        if post_arrival_grace <= 0:
            break

    # Playback rendering
    if playback:
        fig = render_frame()
        frame_container.pyplot(fig)
        time.sleep(play_delay)

coverage_pct = 100.0 * len(covered) / max(1, rows * cols)

# Final still (ensures one figure even if playback already showed frames)
final_fig = render_frame()
st.pyplot(final_fig)

# ------------------------------
# Metrics
# ------------------------------
st.subheader("📊 Metrics")
lat_df = pd.DataFrame(mpc_times)
ms = min_separation(drones)

st.write(f"**Total simulated time:** {sim_time:.2f} s  (Δt = {dt:.3f} s)")
if not lat_df.empty:
    st.write(
        f"**MPC decision latency:** avg = {lat_df['latency_s'].mean()*1000:.2f} ms, "
        f"p95 = {lat_df['latency_s'].quantile(0.95)*1000:.2f} ms, "
        f"max = {lat_df['latency_s'].max()*1000:.2f} ms"
    )
st.write(f"**Coverage:** {coverage_pct:.1f}%  (cell={coverage_cell:.2f} u)")
if ms is not None:
    st.write(f"**Min separation (pairwise):** {ms:.2f} u")

# Per-drone summary
metrics_list = []
for d in drones:
    final_dist = float(np.linalg.norm(d.p - d.goal))
    avg_spd = float(np.mean(d.speed_hist))
    peak_spd = float(np.max(d.speed_hist))
    batt_pct = 100.0 * d.energy_Wh / max(1e-9, d.energy_Wh_init)
    status = "Reached (within ε)" if d.at_goal else ("RTB" if d.rtb else "In transit")
    arrival = f"{d.speed_at_arrival:.2f} u/s" if d.speed_at_arrival is not None else "—"
    tgoal   = f"{d.time_to_goal:.2f} s"        if d.time_to_goal is not None else "—"
    st.write(
        f"Drone {d.id}: Final Dist = {final_dist:.2f} | "
        f"Speed(now) = {d.speed_hist[-1]:.2f} | Arrival = {arrival} @ {tgoal} | "
        f"Avg = {avg_spd:.2f} | Peak = {peak_spd:.2f} | "
        f"Battery = {batt_pct:.1f}% | Status = {status}"
    )
    metrics_list.append({
        "drone_id": d.id,
        "final_dist": final_dist,
        "avg_speed": avg_spd,
        "peak_speed": peak_spd,
        "battery_pct": batt_pct,
        "status": status
    })

# ------------------------------
# 🤖 AI Suggestions (dynamic)
# ------------------------------
st.subheader("🤖 AI Suggestions")
sugs = []

# Coverage bands
if   coverage_pct < 30:
    sugs.append("Coverage is very low (<30%). Increase LiDAR range/rays, add drones, or shrink world size.")
elif coverage_pct < 60:
    sugs.append("Coverage is moderate (30–60%). Consider more rays or a smaller cell size for finer mapping.")
elif coverage_pct > 90:
    sugs.append("Coverage is very high (>90%). You can reduce rays/range or drone count to save compute.")

# Latency bands relative to workload (H × rays)
if not lat_df.empty:
    avg_ms = float(lat_df["latency_s"].mean() * 1000.0)
    expected_ms = 6.0 + 0.15 * horizon * lidar_rays  # rough target curve
    if avg_ms > max(200.0, 1.5 * expected_ms):
        sugs.append("MPC latency is high. Reduce horizon or LiDAR rays, or enable Fast mode.")
    elif avg_ms < 0.5 * expected_ms:
        sugs.append("MPC latency is very low. Consider longer horizon or more rays for richer behavior.")

# Separation scaled to map size
if ms is not None:
    if ms < max(0.8, 0.02 * grid_size):
        sugs.append("Collision risk: minimum separation is tight. Increase avoidance penalties or slow max speed.")
    elif ms > 0.20 * grid_size:
        sugs.append("Fleet is widely dispersed. Tighten formation goals or reduce world size.")

# Battery – one tip only
avg_batt = np.mean([100.0 * d.energy_Wh / max(1e-9, d.energy_Wh_init) for d in drones])
if avg_batt < 25:
    sugs.append("Batteries are low on average (<25%). Reduce mission time or increase initial energy.")
elif avg_batt > 80 and sim_time < steps * dt * 0.75:
    sugs.append("Batteries are largely unused (>80%). Extend flight time, add tasks, or reduce initial energy.")

# Speed utilization (avg & peak)
avg_speeds = [float(np.mean(d.speed_hist)) for d in drones]
peak_speeds = [float(np.max(d.speed_hist)) for d in drones]
if np.mean(avg_speeds) < 0.2 * max_speed and np.mean(peak_speeds) < 0.5 * max_speed:
    sugs.append("Speeds are low. Increase max speed/accel or reduce damping/loiter radius.")
elif np.mean(avg_speeds) > 0.8 * max_speed:
    sugs.append("Drones are speed-saturated. Lower penalties or raise max speed cautiously.")

# Show unique suggestions only
shown = set()
if sugs:
    for s in sugs:
        if s not in shown:
            st.write("- " + s); shown.add(s)
else:
    st.write("No critical suggestions — parameters look balanced.")

# ------------------------------
# Exports (CSV/JSON/PNG/GeoJSON + ZIP)
# ------------------------------
st.subheader("📦 Export Results")

traj_df = pd.DataFrame(traj_rows)
metrics_df = pd.DataFrame([{
    "drone_id": d.id,
    "final_distance": float(np.linalg.norm(d.p - d.goal)),
    "speed_now": d.speed_hist[-1],
    "arrival_speed": d.speed_at_arrival,
    "time_to_goal_s": d.time_to_goal,
    "avg_speed": float(np.mean(d.speed_hist)),
    "peak_speed": float(np.max(d.speed_hist)),
    "battery_pct": 100.0 * d.energy_Wh / max(1e-9, d.energy_Wh_init),
    "status": "Reached (within ε)" if d.at_goal else ("RTB" if d.rtb else "In transit")
} for d in drones])

fused_df = pd.DataFrame(fused_points, columns=["x", "y", "step", "time_s"]) if len(fused_points) > 0 \
           else pd.DataFrame(columns=["x", "y", "step", "time_s"])

run_summary = {
    "timestamp": datetime.utcnow().isoformat() + "Z",
    "preset": preset,
    "seed": int(seed),
    "world": {"grid_size": grid_size, "coverage_cell": coverage_cell, "coverage_pct": coverage_pct},
    "obstacles": {"count": len(obstacles), "size": ob_size, "speed": ob_speed},
    "agents": {"num_drones": num_drones},
    "timing": {"steps_max": steps, "dt": dt, "total_time_s": sim_time, "horizon": horizon},
    "limits": {"max_speed": max_speed, "max_accel": max_accel, "goal_radius": goal_radius, "damping": damping},
    "lidar": {"range": lidar_radius, "rays_per_drone": lidar_rays, "show_beams": bool(show_beams)},
    "wind": {"wx": wx, "wy": wy, "gust_sigma": gust},
    "loiter": {"enabled": bool(loiter_enabled), "speed": loiter_speed, "radius": loiter_radius},
    "latency_ms": {
        "avg": float(lat_df["latency_s"].mean()*1000) if not lat_df.empty else None,
        "p95": float(lat_df["latency_s"].quantile(0.95)*1000) if not lat_df.empty else None,
        "max": float(lat_df["latency_s"].max()*1000) if not lat_df.empty else None
    },
    "safety_min_separation": float(ms) if ms is not None else None,
    "suggestions": sugs
}

# Individual downloads
png_buf = io.BytesIO()
final_fig.savefig(png_buf, format="png", dpi=(120 if fast_mode else 180), bbox_inches="tight")
st.download_button("🖼️ Plot (PNG)", png_buf.getvalue(), "plot.png", "image/png")
st.download_button("📄 Trajectories (CSV)", traj_df.to_csv(index=False).encode("utf-8"),
                   "trajectories.csv", "text/csv")
lat_df_out = (pd.DataFrame() if lat_df.empty else lat_df)
st.download_button("📄 Latency (CSV)", lat_df_out.to_csv(index=False).encode("utf-8"),
                   "latency.csv", "text/csv")
st.download_button("📄 Metrics (CSV)", metrics_df.to_csv(index=False).encode("utf-8"),
                   "metrics.csv", "text/csv")
st.download_button("📄 Fused LiDAR Hits (CSV)", fused_df.to_csv(index=False).encode("utf-8"),
                   "fused_map.csv", "text/csv")
st.download_button("🧾 Run Summary (JSON)", json.dumps(run_summary, indent=2).encode("utf-8"),
                   "run_summary.json", "application/json")
geojson_bytes = to_geojson(drones).encode("utf-8")
st.download_button("🌍 Trajectories (GeoJSON)", geojson_bytes, "trajectories.geojson", "application/geo+json")

# ZIP bundle (all files)
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w", zipfile.ZIP_DEFLATED) as zf:
    zf.writestr("plot.png", png_buf.getvalue())
    zf.writestr("trajectories.csv", traj_df.to_csv(index=False).encode("utf-8"))
    zf.writestr("latency.csv", lat_df_out.to_csv(index=False).encode("utf-8"))
    zf.writestr("metrics.csv", metrics_df.to_csv(index=False).encode("utf-8"))
    zf.writestr("fused_map.csv", fused_df.to_csv(index=False).encode("utf-8"))
    zf.writestr("run_summary.json", json.dumps(run_summary, indent=2).encode("utf-8"))
    zf.writestr("trajectories.geojson", geojson_bytes)

st.download_button("📦 Download ALL (ZIP)", zip_buf.getvalue(), "uav_lidar_mpc_run.zip", "application/zip")
