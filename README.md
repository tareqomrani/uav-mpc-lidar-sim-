✈️ UAV LiDAR + MPC Lab

A lightweight, interactive multi-agent UAV simulator that blends LiDAR sensing with Model Predictive Control (MPC) for real-time obstacle avoidance, tasking, and coordination. Runs in a browser via Streamlit.



⸻

✨ Highlights
	•	Multi-drone 2-D world with bouncing moving obstacles
	•	LiDAR ray-casting with coverage along beams (not just hits)
	•	MPC navigation with separation penalties and wind/gusts
	•	Goal behaviors: damped hover or circular Loiter Mode
	•	Arrival metrics: time-to-goal and speed at first arrival
	•	Playback mode: step-by-step animation or instant final snapshot
	•	Performance tools: Fast Mode, decimated plotting, batched beam drawing
	•	AI Suggestions: dynamic, data-driven tips based on coverage, latency, separation, battery, and speeds
	•	One-click exports: CSVs, JSON, PNG, GeoJSON, and full ZIP bundle

⸻

🧩 Features at a Glance
	•	Scenarios (presets): Warehouse Scan · Urban Canyon · Disaster Mapping
	•	LiDAR: configurable range & rays; fused map of hits; coverage grid %
	•	Wind & Gusts: constant vector plus zero-mean Gaussian gusts
	•	Energy/RTB: simple power model; automatic Return-To-Base when low
	•	Tasking: greedy nearest-goal assignment
	•	Safety: minimum inter-drone separation metric
	•	Metrics: latency (avg/p95/max), speeds (avg/peak/now), battery %, final distance
🕹️ Using the App
	•	Open the sidebar to configure the world (size, steps, Δt), MPC (horizon, speed/accel), LiDAR (range & rays), wind/gusts, drones/obstacles, energy, coverage cell size, and RNG seed.
	•	Choose Hover (damped) or Loiter (circle) at goals.
	•	Toggle Playback to watch the simulation step-by-step; adjust the delay slider for speed.
Disable Playback for a faster, compute-only run with a final snapshot.
	•	Use Fast Mode if you’re on mobile or see lag; it decimates plot data and batches beam drawing.

Key Toggles
	•	Loiter at goal: circles around the goal with configurable speed & radius.
	•	If Loiter is off, the sim ends early once all drones are within ε.
	•	If Loiter is on, the sim runs a short “grace” period after all drones arrived.
	•	Show LiDAR beams: visualizes all rays (batched for speed); turn off for maximum FPS.

⸻

📊 Metrics & Suggestions
	•	Arrival: records time-to-goal and speed at the instant the goal is first reached.
	•	Coverage: computed by tracing each beam across the grid (so range/rays/cell size matter).
	•	Latency: per-drone MPC timing (avg / p95 / max).
	•	AI Suggestions (dynamic):
	•	Coverage bands: <30%, 30–60%, >90%
	•	Latency relative to workload (H × rays)
	•	Separation scaled to map size
	•	Battery (average % and mission duration context)
	•	Speed utilization (avg & peak vs max)

⸻

📦 Exports

Buttons at the bottom let you download:
	•	plot.png – final figure
	•	trajectories.csv – time series (step, t, id, x, y, vx, vy, speed, energy, rtb)
	•	latency.csv – per-step MPC timings
	•	metrics.csv – per-drone summary (final distance, speeds, battery, status, arrival metrics)
	•	fused_map.csv – fused LiDAR hits (x, y, step, time)
	•	run_summary.json – all parameters, metrics, suggestions
	•	trajectories.geojson – LineStrings for each drone
	•	uav_lidar_mpc_run.zip – one-click ZIP bundle of everything above
⚙️ Parameters That Matter
	•	LiDAR Range / Rays ↑ → higher coverage & cost; adjust with Fast Mode if needed.
	•	Coverage Cell Size ↓ → finer coverage % resolution; more cells to fill.
	•	Horizon (H) ↑ → richer MPC lookahead but higher latency.
	•	Max Speed / Accel → agility; combine with separation penalties (built into MPC cost).
	•	Damping / Loiter radius & speed → arrival behavior realism and “Speed(now)” after arrival.

⸻

🐢 Performance Tips
	•	Turn on Fast Mode (recommended on mobile).
	•	Lower LiDAR rays or disable Show Beams.
	•	Reduce World Size and/or Steps during tuning.
	•	PNG export uses lower DPI in Fast Mode to speed up downloads.

⸻

🧪 Reproducibility
	•	Use the Random seed control to make runs repeatable.
	•	Scenario presets tweak obstacle counts/sizes/speeds for representative environments.
❓ Troubleshooting
	•	Slow rendering: enable Fast Mode, hide beams, reduce LiDAR rays and Steps.
	•	No movement / stuck drones: raise max_speed/max_accel or lower goal_radius / obstacle density.
	•	Coverage always low: increase Range/Rays, decrease Cell size, or shrink World size.
	•	Latency tip always “very low”: intentionally increase Horizon or Rays to exercise the planner.
