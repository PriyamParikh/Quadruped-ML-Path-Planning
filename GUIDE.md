# User Guide

This document explains how to operate the `Quadruped-ML-Path-Planning` simulator and reproduce the principal workflow.

## 1. Requirements

Recommended:

- Python 3.11
- Windows 10/11 or Linux

Install dependencies:

```bash
pip install numpy matplotlib pillow scikit-learn joblib
```

For Ubuntu or Debian:

```bash
sudo apt install python3-tk
```

## 2. Start the Application

```bash
python ROBOQUAD_X_STUDIO.py
```

Replace the filename if the main GUI file has a different name.

## 3. Confirm Robot Geometry

Reported settings:

- body: 0.40 × 0.25 × 0.05 m
- upper leg: 0.06 m
- lower leg: 0.05 m

Joint limits:

- \(q_1\): \(-45^\circ\) to \(45^\circ\)
- \(q_2\): \(-100^\circ\) to \(80^\circ\)
- \(q_3\): \(-150^\circ\) to \(10^\circ\)

## 4. Joint and Pose Control

The GUI supports:

- joint sliders
- numeric joint-angle entry
- pose memory
- pose recall
- multi-pose motion sequences

## 5. Directional Motion

Built-in motion modes include:

- Forward
- Backward
- Left turn
- Right turn

## 6. Workspace Analysis

Generate:

- 3D reachable workspace
- X-Y projection
- X-Z projection
- Y-Z projection

## 7. Gait Analysis

Reported gait configuration:

- trot gait
- duty factor: 0.55
- gait cycle: 0.65 s
- diagonal pairs: FL–RR and FR–RL

## 8. Train the ML Models

Available models:

- Random Forest
- Extra Trees
- KNN
- MLP

Reported training configuration:

- 100000 samples
- 20% test fraction
- random seed 42

Compare:

- RMSE
- mean \(R^2\)
- normalised RMSE
- overall rank

Extra Trees was selected in the reported study.

## 9. Enter Destinations

Enter the required navigation points.

For controlled comparison, use the same five destinations for every planner.

## 10. Select a Planner

Available planners:

- Dijkstra
- A*
- RRT
- RRT*
- QGA*

## 11. Static Obstacles

Use the mouse-based obstacle placement option where available.

The obstacles should appear in:

- path-planning view
- simulation
- GIF/animation export

Use the same obstacle arrangement for every planner when comparing algorithms.

## 12. Dynamic Obstacle Test

During active simulation:

1. insert one dynamic obstacle
2. current HEAD position becomes the new start
3. remaining destinations are retained
4. the planner replans
5. the ML controller continues execution

## 13. Long-Route Settings

Recommended reported settings:

- automatic command budget: enabled
- look-ahead distance: 0.24 m
- minimum progress ratio: 0.25
- budget safety factor: 1.65
- extra reserve: 80

Automatic command budgeting prevents long routes from terminating because of a fixed global command limit.

## 14. Turn-First Tracking

For large heading error, the robot should turn first and then move forward.

This prevents the robot from moving backwards towards the destination when the body orientation is incorrect.

## 15. HEAD-Based Navigation

Use the robot HEAD point for:

- destination error
- planner-following error
- dynamic replanning start position

This prevents the tail or body centre from being treated as the navigation target.

## 16. Important Outputs

During simulation, monitor:

- desired planner route
- executed HEAD trajectory
- tracking RMSE
- final HEAD error
- planner name
- ML controller name
- command count
- planner time
- ML controller time
- singularity

## 17. Joint Kinematic Graphs

For all 12 joints, save:

- angular displacement
- angular velocity
- angular acceleration

## 18. Error Interpretation

Do not confuse local look-ahead error with route-tracking RMSE.

Local target error measures distance to the active look-ahead point.

Route-tracking RMSE measures the executed trajectory relative to the planner route.

## 19. Recommended Experimental Protocol

### Phase I — No Obstacle

Run:

- Dijkstra + Extra Trees
- A* + Extra Trees
- RRT + Extra Trees
- RRT* + Extra Trees
- QGA* + Extra Trees

### Phase II — Static Obstacles

Repeat with the same static obstacles.

### Phase III — Dynamic Obstacle

Repeat with one dynamically inserted obstacle.

## 20. Save These Results

For each planner and condition, save:

- path planning
- desired versus executed trajectory
- error analysis
- singularity graph
- robot position image
- 12 joint kinematic plots

Additional outputs can include:

- workspace graphs
- gait graphs
- ML ranking
- planner comparison
- GIF/animation
- CSV/Excel results

## 21. Publication-Quality Export

For publication graphics, retain:

- axis labels
- tick values
- legends
- units
- algorithm names
- condition names

A 600 DPI export is recommended.

## 22. Suggested Result Folder Structure

```text
results/
├── no_obstacle/
│   ├── Dijkstra/
│   ├── A_star/
│   ├── RRT/
│   ├── RRT_star/
│   └── QGA_star/
├── static_obstacles/
│   ├── Dijkstra/
│   ├── A_star/
│   ├── RRT/
│   ├── RRT_star/
│   └── QGA_star/
└── dynamic_obstacle/
    ├── Dijkstra/
    ├── A_star/
    ├── RRT/
    ├── RRT_star/
    └── QGA_star/
```

## 23. Troubleshooting

### Robot stops before completing a long route

Check:

- automatic command budgeting is enabled
- look-ahead tracking is enabled
- translational step is not too small
- planner route is complete

### Robot moves backwards towards the target

Check:

- turn-first policy is enabled
- HEAD tracking is active
- automatic reverse is disabled unless intentionally required

### Tail reaches the destination

Confirm that goal error uses the HEAD point rather than the body centre or tail.

### RRT or RRT* fails

Consider increasing:

- maximum iterations
- goal bias
- RRT* rewire radius

Also verify obstacle geometry and map bounds.

### Dynamic obstacle is missing from animation

Confirm that the dynamic obstacle is added to:

- planner map
- simulation scene
- GIF/animation export layer

## 24. Reproducibility

Keep the following fixed when comparing planners:

- robot geometry
- joint limits
- gait timing
- destination sequence
- ML controller
- obstacle configuration
- map bounds
- tracking tolerance
- simulation speed

Only change the planner being evaluated.
