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
