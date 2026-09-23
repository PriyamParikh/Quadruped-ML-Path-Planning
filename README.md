# Quadruped-ML-Path-Planning

Machine learning assisted path planning, gait control, safety analysis, and dynamic navigation for a 12 DOF quadruped robot.

**Developed by Dr. Priyam Parikh and Shaurya Shah**

## Overview

`Quadruped-ML-Path-Planning` is a research-oriented framework that combines forward kinematics, gait generation, machine learning based control, path planning, singularity monitoring, obstacle avoidance, and dynamic replanning.

Implemented planners:

- Dijkstra
- A*
- RRT
- RRT*
- QGA* — Quadruped Gait Aware A*

Machine learning controllers:

- Random Forest
- Extra Trees
- K Nearest Neighbours
- Multilayer Perceptron

The navigation architecture is:

\[
\text{Path Planner}
\rightarrow
\text{Desired HEAD Route}
\rightarrow
\text{ML Controller}
\rightarrow
\text{Gait Generator}
\rightarrow
\text{12 DOF Quadruped Motion}
\]

## Proposed QGA* Planner

Conventional A* uses

\[
f(n)=g(n)+h(n)
\]

QGA* extends the cost to

\[
f_Q(n)=
g(n)
+w_hh(n)
+w_{\psi}C_{\psi}(n)
+w_oC_o(n)
+w_qC_q(n)
+w_sC_s(n)
+w_dC_d(n)
\]

where:

- \(C_{\psi}\): heading-change cost
- \(C_o\): obstacle and body-clearance cost
- \(C_q\): joint/gait-motion cost
- \(C_s\): singularity cost
- \(C_d\): dynamic-obstacle proximity cost

Fixed QGA* weights used in the reported experiments:

\[
w_h=1.00,\quad
w_{\psi}=0.45,\quad
w_o=0.35,\quad
w_q=0.15,\quad
w_s=0.08,\quad
w_d=0.40
\]

QGA* therefore does not minimise path length alone. It also evaluates quadruped-specific safety and execution suitability.

## Quadruped Model

The robot has four identical legs and 12 revolute DOF.

Each leg contains:

- \(q_1\): hip yaw
- \(q_2\): hip pitch
- \(q_3\): knee pitch

Body dimensions:

\[
0.40\times0.25\times0.05\ \text{m}
\]

Leg lengths:

\[
L_1=0.06\ \text{m},\qquad L_2=0.05\ \text{m}
\]

Joint limits:

\[
q_1\in[-45^\circ,45^\circ]
\]

\[
q_2\in[-100^\circ,80^\circ]
\]

\[
q_3\in[-150^\circ,10^\circ]
\]

## Gait

The simulator uses a trot gait with:

- duty factor \(\beta=0.55\)
- gait cycle \(0.65\ \text{s}\)
- diagonal pairing FL–RR and FR–RL
- forward, backward, left-turn, and right-turn motion

## Machine Learning

Controller input:

\[
\mathbf{x}_k=
\begin{bmatrix}
|e_x| & e_y & d & e_{\psi} & s
\end{bmatrix}^{T}
\]

Controller output:

\[
\hat{\mathbf u}_k=
\mathcal M(\mathbf x_k)=
\begin{bmatrix}
\Delta s & \Delta\psi & A_{q_2} & A_{q_3}
\end{bmatrix}^{T}
\]

Extra Trees achieved the strongest overall ranking in the reported ML comparison:

\[
R^2_{\text{mean}}\approx0.99982
\]

\[
NRMSE\approx0.01182
\]

It was therefore used as the common execution controller in the final planner comparison.

## Experimental Conditions

Three controlled conditions were evaluated:

1. No obstacle
2. Static obstacles
3. One dynamically introduced obstacle

The same robot geometry, gait settings, five destinations, and Extra Trees controller were retained for every planner comparison.

## Key Safety Result

In the static-obstacle experiment, QGA* achieved:

- collision-free execution
- tracking RMSE of approximately \(0.00386\ \text{m}\)
- minimum planner clearance of approximately \(1.23\ \text{m}\)

RRT* was also collision-free, with approximately:

- tracking RMSE \(0.00557\ \text{m}\)
- minimum planner clearance \(0.23\ \text{m}\)

Dijkstra, A*, and RRT were not collision-free under the same static-obstacle condition.

QGA* therefore showed the strongest overall safety performance among the tested planners under the reported experimental conditions. This does not imply universal superiority because QGA* also required greater computation and sometimes longer paths.

## Installation

Python 3.11 is recommended.

```bash
pip install numpy matplotlib pillow scikit-learn joblib
```

For Ubuntu or Debian:

```bash
sudo apt install python3-tk
```

## Run

```bash
git clone https://github.com/PriyamParikh/Quadruped-ML-Path-Planning.git
cd Quadruped-ML-Path-Planning
python ROBOQUAD_X_STUDIO.py
```

Replace `ROBOQUAD_X_STUDIO.py` with the actual GUI filename if required.

## Documentation

- [METHODOLOGY.md](METHODOLOGY.md)
- [RESULTS.md](RESULTS.md)
- [GUIDE.md](GUIDE.md)

## Suggested Repository Structure

```text
Quadruped-ML-Path-Planning/
├── README.md
├── METHODOLOGY.md
├── RESULTS.md
├── GUIDE.md
├── LICENSE
├── requirements.txt
├── src/
├── figures/
├── results/
│   ├── no_obstacle/
│   ├── static_obstacles/
│   └── dynamic_obstacle/
├── animations/
├── models/
└── data/
```

## Authors

**Dr. Priyam Parikh**  
Product Design and Mechatronics  
Anant National University, Ahmedabad, India

**Shaurya Shah**

## Citation

```bibtex
@software{parikh_quadruped_ml_path_planning,
  author = {Priyam Parikh and Shaurya Shah},
  title = {Quadruped-ML-Path-Planning},
  year = {2026},
  url = {https://github.com/PriyamParikh/Quadruped-ML-Path-Planning},
  note = {Machine learning assisted path planning and QGA* navigation framework for quadruped robots}
}
```

## Disclaimer

This repository is a research simulation framework. Simulation results should not be interpreted as guaranteed hardware performance. Physical deployment requires further validation of sensing, actuation, latency, contact dynamics, terrain interaction, energy use, and platform-specific safety.
