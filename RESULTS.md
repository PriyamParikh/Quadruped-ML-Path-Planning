# Results

This document summarises the quantitative findings of the `Quadruped-ML-Path-Planning` study.

## 1. ML Controller Ranking

Four regressors were evaluated:

- Random Forest
- Extra Trees
- KNN
- MLP

Extra Trees achieved the strongest overall ranking and was selected for the final planner study.

Reported overall values:

\[
R^2_{\text{mean}}\approx0.99982
\]

\[
NRMSE\approx0.01182
\]

## 2. No-Obstacle Condition

QGA* achieved:

- tracking RMSE: \(0.0051\ \text{m}\)
- final HEAD error: \(0.0000\ \text{m}\)
- ML controller time: \(8.8244\ \text{s}\)
- commands: 329

Dijkstra, A*, RRT, and RRT* produced approximately \(0.0069\ \text{m}\) tracking RMSE and \(0.0075\ \text{m}\) final HEAD error.

## 3. Static-Obstacle Condition

QGA* achieved:

- collision-free execution: Yes
- tracking RMSE: approximately \(0.0039\ \text{m}\)
- minimum planner clearance: approximately \(1.23\ \text{m}\)

RRT* was also collision-free with approximately:

- tracking RMSE: \(0.0056\ \text{m}\)
- minimum planner clearance: \(0.23\ \text{m}\)

Dijkstra, A*, and RRT were not collision-free in the same static-obstacle condition.

## 4. Dynamic-Obstacle Condition

All five planners remained collision-free in the reported dynamic-obstacle experiment.

Tracking RMSE:

- Dijkstra: \(0.0054\ \text{m}\)
- A*: \(0.0000\ \text{m}\)
- RRT: \(0.0075\ \text{m}\)
- RRT*: \(0.0051\ \text{m}\)
- QGA*: \(0.0042\ \text{m}\)

QGA* also achieved zero final HEAD error.

A* produced the lowest tracking RMSE for this individual dynamic run, showing that QGA* does not dominate every isolated metric.

## 5. Singularity Behaviour

Representative QGA* runs maintained bounded singularity indicators.

The minimum singular value remained approximately

\[
\sigma_{\min}\approx2.29\times10^{-2}
\]

indicating that the robot did not approach a critical kinematic singularity in the evaluated runs.

## 6. Final Quantitative Comparison

| Condition | Planner | Tracking RMSE (m) | Final HEAD Error (m) | ML Time (s) | Commands |
|---|---|---:|---:|---:|---:|
| No obstacle | Dijkstra | 0.0069 | 0.0075 | 8.8964 | 336 |
| No obstacle | A* | 0.0069 | 0.0075 | 8.8675 | 336 |
| No obstacle | RRT | 0.0069 | 0.0075 | 9.4109 | 336 |
| No obstacle | RRT* | 0.0069 | 0.0075 | 9.5596 | 336 |
| No obstacle | QGA* | 0.0051 | 0.0000 | 8.8244 | 329 |
| Static obstacles | Dijkstra | 0.0071 | 0.0080 | 14.6805 | 336 |
| Static obstacles | A* | 0.0069 | 0.0162 | 17.9459 | 352 |
| Static obstacles | RRT | 0.0055 | 0.0132 | 17.3806 | 352 |
| Static obstacles | RRT* | 0.0056 | 0.0000 | 15.7890 | 338 |
| Static obstacles | QGA* | 0.0039 | 0.0179 | 18.3121 | 399 |
| Dynamic obstacle | Dijkstra | 0.0054 | 0.0000 | 2.8529 | 60 |
| Dynamic obstacle | A* | 0.0000 | 0.0032 | 2.4733 | 57 |
| Dynamic obstacle | RRT | 0.0075 | 0.0000 | 3.1476 | 70 |
| Dynamic obstacle | RRT* | 0.0051 | 0.0063 | 2.4166 | 62 |
| Dynamic obstacle | QGA* | 0.0042 | 0.0000 | 3.2426 | 69 |

## 7. Safety-Oriented Interpretation

The strongest evidence for QGA* comes from the static-obstacle case.

\[
RMSE_{\text{QGA*}}\approx0.00386\ \text{m}
\]

\[
d_{\min,\text{QGA*}}\approx1.23\ \text{m}
\]

Compared with RRT*:

\[
RMSE_{\text{RRT*}}\approx0.00557\ \text{m}
\]

\[
d_{\min,\text{RRT*}}\approx0.23\ \text{m}
\]

QGA* therefore provided lower tracking error and substantially larger clearance in the static-obstacle experiment.

## 8. Computational Trade-Off

For the static-obstacle case, QGA* required approximately:

- planning time: \(20.86\ \text{s}\)
- node expansions: 77,838
- route length: \(21.86\ \text{m}\)

Thus:

\[
\text{Improved safety awareness}
\Longleftrightarrow
\text{Higher computation and route cost}
\]

QGA* should therefore be described as the safest overall planner among the tested methods under the reported conditions, not as the fastest or universally best planner.
