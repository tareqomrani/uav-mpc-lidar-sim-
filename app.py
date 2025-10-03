# app.py — Multi-Agent LiDAR + MPC (Maximal Build)
# Includes: Timed dynamics, MPC, LiDAR, wind/gusts, moving obstacles, coverage grid,
# battery/RTB, task allocation, scenario presets, metrics, AI Suggestions, full exports.

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
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
        feats.append({"type":"Feature",
                      "properties":{"id": int(d.id)},
                      "geometry":{"type":"LineString","coordinates":coords}})
    return json.dumps({"type":"FeatureCollection","features":feats}, indent=2)

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
        half = self.size/2
        for k in (0,1):
            if self.p[k] < half:
                self.p[k] = half
                self.v[k] *= -1
            if self.p[k] > bounds - half:
                self.p[k] = bounds - half
                self.v[k] *= -1

# ------------------------------
# UAV (MPC + Damped Hover + Battery/RTB)
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

    def lidar_ray_cast(self, obstacles, max_range, n_rays=24):
        hits, beams = [], []
        angles = np.linspace(0, 2*np.pi, n_rays, endpoint=False)
        for ang in angles:
            dvec = np.array([np.cos(ang), np.sin(ang)], float)
            r, step = 0.0, 0.25
            hit_point = None
            while r < max_range:
                sample = self.p + dvec * r
                collided = False
                for ob in obstacles:
                    half = ob.size/2
                    if (abs(sample[0]-ob.p[0]) <= half) and (abs(sample[1]-ob.p[1]) <= half):
                        collided = True
                        hit_point = sample.copy()
                        break
                if collided: break
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
            p = p + (v + wind)*dt
            cost += np.linalg.norm(p - self.goal)**2
            for ob in obstacles:
                outside = np.maximum(np.abs(p - ob.p) - ob.size/2, 0.0)
                dist = np.linalg.norm(outside)
                if dist < 0.5: cost += 400.0
                elif dist < lidar_radius: cost += 2.0/(dist+1e-3)
            for od in others:
                if od.id == self.id: continue
                d = np.linalg.norm(p - od.p)
                if d < 0.8: cost += 400.0
                elif d < 2.5: cost += 7.0/(d+1e-3)
        cost += 0.1*np.dot(a,a)
        return cost

    def mpc_step(self, drones, obstacles, H=3, lidar_radius=6.0, damping=0.6, wind=np.zeros(2)):
        if (not self.rtb) and self.energy_Wh <= 0.15 * self.energy_Wh_init:
            self.rtb = True
            self.goal = self.home.copy()
        if np.linalg.norm(self.p - self.goal) <= self.goal_radius:
            self.at_goal = True
            self.v *= float(damping)
            self.p = self.p + (self.v + wind) * self.dt
            self._log_state()
            return
        acc = self.max_accel
        candidates = [np.array([ax, ay], float) for ax in [-acc,0,acc] for ay in [-acc,0,acc]]
        best_a, best_cost = np.zeros(2), float("inf")
        for a in candidates:
            c = self._rollout(a, drones, obstacles, H, lidar_radius, wind)
            if c < best_cost:
                best_cost, best_a = c, a
        self.v = clip_norm(self.v + best_a*self.dt, self.max_speed)
        self.p = self.p + (self.v + wind)*self.dt
        self._log_state()

    def _log_state(self):
        self.path.append(tuple(self.p))
        self.vel_hist.append(tuple(self.v))
        self.speed_hist.append(float(np.linalg.norm(self.v)))
        power_W = 20.0 + 8.0*(np.linalg.norm(self.v)**3)
        self.energy_Wh = max(0.0, self.energy_Wh - power_W*self.dt/3600.0)

    def set_energy_init(self):
        self.energy_Wh_init = self.energy_Wh

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Multi-Agent LiDAR + MPC (Maximal Lab)", layout="wide")
st.title("🛰️ Multi-Agent UAV LiDAR + MPC — Maximal Lab")

# Presets
preset = st.sidebar.selectbox("Scenario preset", ["None","Warehouse Scan","Urban Canyon","Disaster Mapping"])
st.sidebar.write("---")

# Controls
grid_size = st.sidebar.slider("World Size", 20, 100, 40)
steps = st.sidebar.slider("Simulation Steps", 20, 600, 200)
dt = st.sidebar.slider("Δt (s)", 0.05, 1.0, 0.2, 0.05)
horizon = st.sidebar.slider("MPC Horizon H", 1, 8, 3)
max_speed = st.sidebar.slider("Max Speed", 0.5, 12.0, 4.0, 0.1)
max_accel = st.sidebar.slider("Max Accel", 0.2, 8.0, 2.0, 0.1)
goal_radius = st.sidebar.slider("Goal Radius ε", 0.1, 3.0, 0.7, 0.1)
damping = st.sidebar.slider("Hover Damping", 0.2, 0.95, 0.6, 0.05)
wx = st.sidebar.slider("Wind X", -3.0, 3.0, 0.0, 0.1)
wy = st.sidebar.slider("Wind Y", -3.0, 3.0, 0.0, 0.1)
gust = st.sidebar.slider("Gust σ", 0.0, 2.0, 0.3, 0.1)
lidar_radius = st.sidebar.slider("LiDAR Range", 2.0, 20.0, 8.0, 0.5)
lidar_rays = st.sidebar.slider("LiDAR Rays", 8, 96, 36, 1)
show_beams = st.sidebar.checkbox("Show beams", True)
num_drones = st.sidebar.slider("Drones", 1, 10, 4)
num_obstacles = st.sidebar.slider("Obstacles", 0, 30, 10)
ob_size = st.sidebar.slider("Obstacle Size", 0.6, 5.0, 1.2, 0.1)
ob_speed = st.sidebar.slider("Obstacle Speed", 0.0, 5.0, 1.3, 0.1)
energy_Wh = st.sidebar.slider("Energy (Wh)", 10.0, 200.0, 80.0, 1.0)
coverage_cell = st.sidebar.slider("Coverage cell size", 0.5, 4.0, 1.0, 0.1)
seed = st.sidebar.number_input("Seed", value=7, step=1)

rng = np.random.default_rng(int(seed))

# Preset obstacle placement
preset_obstacles = []
if preset == "Warehouse Scan":
    num_obstacles = max(num_obstacles, 14)
    ob_size = max(ob_size, 1.6)
    for x in np.linspace(8, grid_size-8, 6):
        preset_obstacles.append(MovingObstacle(x, grid_size/2, 0, 0, ob_size))
elif preset == "Urban Canyon":
    num_obstacles = max(num_obstacles, 16)
    ob_size = max(ob_size, 2.4)
    for x in np.linspace(6, grid_size-6, 5):
        preset_obstacles.append(MovingObstacle(x, grid_size*0.35, 0, 0, ob_size))
        preset_obstacles.append(MovingObstacle(x, grid_size*0.65, 0, 0, ob_size))
elif preset == "Disaster Mapping":
    num_obstacles = max(num_obstacles, 10)
    ob_speed = max(ob_speed, 1.6)

# Obstacles
obstacles = preset_obstacles[:]
for _ in range(max(0, num_obstacles - len(preset_obstacles))):
    x = float(rng.uniform(0.5, grid_size-0.5))
    y = float(rng.uniform(0.5, grid_size-0.5))
    ang = rng.uniform(0, 2*np.pi)
    vx, vy = ob_speed*np.cos(ang), ob_speed*np.sin(ang)
    obstacles.append(MovingObstacle(x, y, vx, vy, size=ob_size))

# Drones + greedy goal assignment
drones = []
starts, goals = [], []
for i in range(num_drones):
    starts.append((float(rng.uniform(1,grid_size-1)), float(rng.uniform(1,grid_size-1))))
    goals.append((float(rng.uniform(1,grid_size-1)), float(rng.uniform(1,grid_size-1))))

# Assign nearest goals
assigned = set()
assigned_goals = []
for s in starts:
    dists = [(j, np.linalg.norm(np.array(s)-np.array(goals[j]))) for j in range(len(goals)) if j not in assigned]
    jmin = min(dists, key=lambda x:x[1])[0]
    assigned.add(jmin)
    assigned_goals.append(goals[jmin])
goals = assigned_goals

for i in range(num_drones):
    u = UAV(starts[i][0], starts[i][1], goals[i], i, rand_color(i), dt,
            max_speed, max_accel, goal_radius, energy_Wh, home=starts[i])
    u.set_energy_init()
    drones.append(u)

# Coverage grid
cols = int(np.ceil(grid_size/coverage_cell))
rows = int(np.ceil(grid_size/coverage_cell))
covered=set()

# Logs
fused_points=[]; traj_rows=[]; mpc_times=[]
sim_time=0.0
for t in range(steps):
    wind = np.array([wx,wy]) + np.random.normal(0,gust,size=2)
    for d in drones:
        beams,hits = d.lidar_ray_cast(obstacles, max_range=lidar_radius, n_rays=lidar_rays)
        d._last_beams,d._last_hits = beams,hits
        for h in hits:
            fused_points.append((float(h[0]), float(h[1]), t, sim_time))
            cx,cy = int(h[0]//coverage_cell), int(h[1]//coverage_cell)
            covered.add((cx,cy))
    for d in drones:
        t0=time.perf_counter()
        d.mpc_step(drones, obstacles, H=horizon, lidar_radius=lidar_radius, damping=damping, wind=wind)
        t1=time.perf_counter()
        mpc_times.append({"step":t,"drone_id":d.id,"latency_s":t1-t0})
        x,y=d.path[-1]; vx,vy=d.vel_hist[-1]
        traj_rows.append({"step":t,"time_s":sim_time+dt,"drone_id":d.id,
                          "x":x,"y":y,"vx":vx,"vy":vy,"speed":d.speed_hist[-1],
                          "energy_Wh":d.energy_Wh,"rtb":d.rtb})
    for ob in obstacles: ob.step(dt,grid_size)
    sim_time+=dt

coverage_pct=100*len(covered)/max(1,rows*cols)

# Plot
fig,ax=plt.subplots(figsize=(8,8))
ax.set_xlim(0,grid_size); ax.set_ylim(0,grid_size)
if show_beams:
    for d in drones:
        for a,b in getattr(d,"_last_beams",[]):
            ax.plot([a[0],b[0]],[a[1],b[1]],color=d.color,alpha=0.2,linewidth=0.8)
for ob in obstacles:
    ax.add_patch(patches.Rectangle((ob.p[0]-ob.size/2,ob.p[1]-ob.size/2),
                                   ob.size,ob.size,color="black"))
for d in drones:
    xs,ys=zip(*d.path)
    ax.plot(xs,ys,"--",color=d.color)
    ax.scatter(d.p[0],d.p[1],color=d.color,s=90,marker="o",label=f"Drone {d.id}")
    ax.scatter(d.goal[0],d.goal[1],color=d.color,marker="*",s=180,edgecolor="k")
ax.legend(); st.pyplot(fig)

# Metrics
st.subheader("📊 Metrics")
lat_df=pd.DataFrame(mpc_times)
ms=min_separation(drones)
st.write(f"Total time={sim_time:.1f}s, steps={steps}, Δt={dt}")
if not lat_df.empty:
    st.write(f"Latency avg={lat_df['latency_s'].mean()*1000:.2f}ms, "
             f"p95={lat_df['latency_s'].quantile(0.95)*1000:.2f}ms")
st.write(f"Coverage={coverage_pct:.1f}%")
if ms: st.write(f"Min separation={ms:.2f}u")
for d in drones:
    st.write(f"Drone {d.id}: dist={np.linalg.norm(d.p-d.goal):.2f}, "
             f"avg_spd={np.mean(d.speed_hist):.2f}, peak={np.max(d.speed_hist):.2f}, "
             f"battery={(100*d.energy_Wh/d.energy_Wh_init):.1f}%, "
             f"status={'RTB' if d.rtb else ('Reached' if d.at_goal else 'In transit')}")

# AI Suggestions
st.subheader("🤖 AI Suggestions")
sugs=[]
if coverage_pct<50: sugs.append("Coverage low (<50%). Increase LiDAR range/rays or add drones.")
elif coverage_pct>90: sugs.append("Coverage excellent (>90%). Reduce rays to save compute.")
if not lat_df.empty:
    avg_lat_ms=lat_df["latency_s"].mean()*1000
    if avg_lat_ms>50: sugs.append("High latency (>50ms). Reduce horizon or rays.")
    elif avg_lat_ms<10: sugs.append("Latency low (<10ms). You may increase horizon or rays.")
if ms and ms<1: sugs.append("Collision risk (min sep <1u). Increase avoidance
