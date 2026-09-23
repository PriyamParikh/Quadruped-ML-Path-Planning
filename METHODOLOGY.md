# Methodology

This document summarises the mathematical and experimental methodology used in `Quadruped-ML-Path-Planning`.

## 1. System Architecture

\[
\text{Planner}
\rightarrow
\text{Global Route}
\rightarrow
\text{Look-Ahead Target}
\rightarrow
\text{ML Controller}
\rightarrow
\text{Gait Command}
\rightarrow
\text{12 DOF Motion}
\]

The same robot geometry, gait parameters, controller settings, and five required destinations were retained for all comparative experiments.

## 2. Forward Kinematics

Each of the four legs is modelled as a 3 DOF serial chain.

The standard Denavit–Hartenberg transformation is

\[
{}^{i-1}\mathbf A_i=
\begin{bmatrix}
c_{\theta_i} & -s_{\theta_i}c_{\alpha_i} & s_{\theta_i}s_{\alpha_i} & a_i c_{\theta_i}\\
s_{\theta_i} & c_{\theta_i}c_{\alpha_i} & -c_{\theta_i}s_{\alpha_i} & a_i s_{\theta_i}\\
0 & s_{\alpha_i} & c_{\alpha_i} & d_i\\
0 & 0 & 0 & 1
\end{bmatrix}
\]

For one 3 DOF leg:

\[
{}^{0}\mathbf T_3=
{}^{0}\mathbf A_1
{}^{1}\mathbf A_2
{}^{2}\mathbf A_3
\]

The foot position is extracted as

\[
{}^{0}\mathbf p_f=
{}^{0}\mathbf T_3(1:3,4)
\]

and expressed in the global frame by

\[
{}^{G}\mathbf p_f=
{}^{G}\mathbf R_0\,{}^{0}\mathbf p_f+{}^{G}\mathbf p_0
\]

## 3. Workspace

\[
\mathcal W=
\left\{
\mathbf p_f(\mathbf q)
\mid
\mathbf q_{\min}\leq\mathbf q\leq\mathbf q_{\max}
\right\}
\]

Joint limits:

\[
q_1\in[-45^\circ,45^\circ],\quad
q_2\in[-100^\circ,80^\circ],\quad
q_3\in[-150^\circ,10^\circ]
\]

## 4. Singularity Monitoring

\[
\mathbf J(\mathbf q)=
\frac{\partial\mathbf p_f}{\partial\mathbf q}
\]

\[
\sigma_{\min}=
\min\left[\sigma(\mathbf J)\right]
\]

A value approaching zero indicates proximity to a kinematic singularity.

## 5. ML Gait Controller

Controller input:

\[
\mathbf{x}_{k}=
\begin{bmatrix}
|e_x| & e_y & d & e_{\psi} & s
\end{bmatrix}^{T}
\]

Controller output:

\[
\hat{\mathbf u}_{k}=
\mathcal M(\mathbf{x}_{k})=
\begin{bmatrix}
\Delta s & \Delta\psi & A_{q_2} & A_{q_3}
\end{bmatrix}^{T}
\]

A turn-first policy is used. Large heading error is corrected before forward translation.

## 6. Long-Route Look-Ahead Control

A local target is placed on the active planner segment:

\[
\mathbf p_L=
\Pi_{\mathcal P}(\mathbf p_H)+L_a\hat{\mathbf t}_{\mathcal P}
\]

with reported look-ahead distance

\[
L_a=0.24\ \text{m}
\]

Automatic command budgeting prevents long routes from stopping because of a fixed command ceiling.

## 7. Conventional A*

\[
f(n)=g(n)+h(n)
\]

with Euclidean heuristic

\[
h(n)=
\sqrt{(x_g-x_n)^2+(y_g-y_n)^2}
\]

## 8. Proposed QGA*

Search state:

\[
n=(x,y,\psi)
\]

Cost:

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

### Heading cost

\[
C_{\psi}(n)=
\frac{|\operatorname{wrap}(\psi_n-\psi_p)|}{\pi}
\]

### Obstacle-clearance cost

\[
d_o(n)=
\min_i\left[
\|\mathbf p_n-\mathbf o_i\|-
\left(r_i+c+\frac{W_b}{2}+m_b\right)
\right]
\]

\[
C_o(n)=
\frac{\Delta r}{d_o(n)+\Delta r+\varepsilon}
\]

### Joint/gait-motion cost

\[
C_q=
\frac{1}{12Nq_r}
\sum_{k=1}^{N}
\sum_{j=1}^{12}
|\Delta q_{j,k}|
\]

### Singularity cost

\[
C_s=
\frac{\sigma_r}{\sigma_{\min}(\mathbf J)+\varepsilon}
\]

### Dynamic-obstacle cost

\[
C_d=
\frac{\Delta r}{d_d+\Delta r+\varepsilon}
\]

### Fixed weights

\[
w_h=1.00,\quad
w_{\psi}=0.45,\quad
w_o=0.35,\quad
w_q=0.15,\quad
w_s=0.08,\quad
w_d=0.40
\]

## 9. Experimental Phases

### Phase I — No Obstacle

All five planners traversed the same five destinations without environmental obstacles.

### Phase II — Static Obstacles

The same destination sequence was retained while fixed obstacles were introduced.

### Phase III — Dynamic Obstacle

One obstacle was inserted during execution. Replanning started from the current HEAD position through the remaining destinations.

## 10. Dynamic Replanning

When a new obstacle appears:

\[
\mathbf s_0\leftarrow\mathbf s_H(t)
\]

\[
\mathcal V\leftarrow\mathcal V_{\text{remaining}}
\]

The route is recomputed from the current HEAD state rather than restarting the full mission.

## 11. Evaluation Metrics

Path length:

\[
L=\sum_{i=1}^{m}\|\mathbf P_i-\mathbf P_{i-1}\|
\]

Tracking RMSE:

\[
RMSE=
\sqrt{
\frac{1}{N}
\sum_{k=1}^{N}
d^2\left(\mathbf p_k^e,\mathcal P\right)
}
\]

Additional measures:

- minimum obstacle clearance
- collision status
- final HEAD error
- minimum singular value
- planning time
- node expansions
- ML controller time
- ML command count
- joint displacement
- angular velocity
- angular acceleration

## 12. Compact Mathematical QGA* Execution Logic

Given

\[
\mathcal G=(\mathcal N,\mathcal E),\quad
\mathbf s_0=(x_0,y_0,\psi_0),\quad
\mathcal V=\{\mathbf v_1,\ldots,\mathbf v_5\}
\]

select

\[
\mathbf n^*=\arg\min_{\mathbf n\in\mathcal Q_o}F_Q(\mathbf n)
\]

and evaluate every feasible neighbour by

\[
F_Q(\mathbf n_j)=
g(\mathbf n_j)+w_hh_j+w_{\psi}C_{\psi,j}+w_oC_{o,j}+w_qC_{q,j}+w_sC_{s,j}+w_dC_{d,j}
\]

The ML execution command is

\[
\mathbf u_t=
\mathcal M^*(\mathbf x_t)=
\begin{bmatrix}
\Delta s_t & \Delta\psi_t & A_{q_2,t} & A_{q_3,t}
\end{bmatrix}^{T}
\]

and the planar body state is updated by

\[
\begin{bmatrix}
x_{t+1}\\y_{t+1}
\end{bmatrix}
=
\begin{bmatrix}
x_t\\y_t
\end{bmatrix}
+
\Delta s_t
\begin{bmatrix}
\cos\psi_t\\\sin\psi_t
\end{bmatrix}
\]

\[
\psi_{t+1}=\operatorname{wrap}(\psi_t+\Delta\psi_t)
\]
