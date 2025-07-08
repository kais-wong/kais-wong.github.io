---
title: "1D Heat Equation"
date: 2025-06-10T15:34:30-04:00
categories:
  - projects
tags:
  - Diffusion Equation
  - Python
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: True
---

<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\(', '\\)']]
  }
};
</script>
<script type="text/javascript" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

# Background
## The Scenario
Suppose we have a 1D bar of metal. The left end of the bar is held at 0°C and the right end of the bar is held at 10°C. The sides of the bar are insulated such that heat is not lost or transferred out of the system. How can we build a computational model of this system?
## Heat Equation
The heat equation is a derivative of the diffusion equation that describes the distribution of heat in a given space over time. It is a partial differential equation.

$$
\frac{\partial T}{\partial t} = \kappa \nabla^2 T
$$

However, since we wil only consider the heat equation in 1D we have the following:

$$\frac{\delta T}{\delta t} = \kappa \dfrac{\delta^2 T}{\delta x^2}$$

Where $\kappa$ is the diffusivity constant.

# Solution

## Discretizing in space

To solve this partial differential equation, we will discritize it in space using the central finite differences approach. By doing so we can rewrite our PDE as a system of coupled ODEs.

$$
\frac{dT_{i}}{dt}=\kappa(\frac{T_{i-1}(t) - 2T_{i}(t) + T_{i+1}(t)}{(\Delta x)^2}) + \mathcal{O}(\Delta x^2)
$$

It follows that

$$
\begin{align*}
\frac{dT_1}{dt} &= \kappa \frac{T_0(t) - 2T_1(t) + T_2(t)}{(\Delta x)^2} \\
\frac{dT_2}{dt} &= \kappa \frac{T_1(t) - 2T_2(t) + T_3(t)}{(\Delta x)^2} \\
&\ \vdots \\
\frac{dT_{N-1}}{dt} &= \kappa \frac{T_{N-2}(t) - 2T_{N-1}(t) + T_N(t)}{(\Delta x)^2}
\end{align*}
$$

## Matrix form

We can use matrix notation to represent this more cleanly.　By inspection, it is apparent that we can combine this system into a matrix containing the coeffiencts of $T_i(t)$ multiplied by a vector containing $T(t)$.

$$
A =
\begin{bmatrix}
-2 & 1 & 0 & \cdots & 0 \\
1 & -2 & 1 & \ddots & \vdots \\
0 & 1 & -2 & \ddots & 0 \\
\vdots & \ddots & \ddots & \ddots & 1 \\
0 & \cdots & 0 & 1 & -2
\end{bmatrix}
\qquad
\mathbf{T}(t) =
\begin{bmatrix}
T_1(t) \\
T_2(t) \\
\vdots \\
T_{N-1}(t)
\end{bmatrix}
$$

The matrix A is actually the discrete representation of the $\frac{d^2}{dx^2}$. Addionally, due to the nature of this problem, we must add the Dirichlet boundary conditions using a term we will call $\vec{b}$.

$$
\vec{b} = [T_0,\ 0,\ 0,\ \ldots,\ 0,\ T_N]^T
$$

We have now unified this system of ODEs into a single equation in matrix form.

$$
\frac{d\mathbf{T}(t)}{dt} = \frac{\kappa}{(\Delta x)^2} A\, \mathbf{T}(t) + \vec{b}
$$

Now, by utilizing linear algebra, we have both represented our ODEs in a more legible manner, and one that allows us to concurrently solve the differential equation at each spacial node together in one step. We can now apply standard numerical methods to approximate the solution to this system of ODEs. 

I will apply three different numerical methods to solve the heat equation. Each method has its advantages and drawbacks, but method 3 is the most conventional.

## Method 1: Forward Euler (Explicit)

Apply the forward euler method where we can approximate the next point in time $f^{n+1}$ using information about the current point $f^{n}$ and its derivative $\frac{df^{n}}{dt}$.

$$
\vec{T}^{\,n+1} = \vec{T}^{\,n} + \Delta t \left( A\, \vec{T}^n + \vec{b}^{\,n} \right)
$$

This simplifies to

$$
\vec{T}^{\,n+1} = \left( I + \Delta t\, A \right) \vec{T}^{\,n} + \Delta t\, \vec{b}^{\,n}
$$


<!-- $$
\mathbf{T}^{n+1} = \mathbf{T}^{n} + \Delta t \left( A\, \mathbf{T}^n + \mathbf{b}^n \right)
$$ -->

We have now discretized the system of ODEs in time, allowing us to approximate a solution numerically.

To simulate the temperature evolution, we iterate this update from $t_{\text{initial}}$ to $t_{\text{final}}$.

Below is my implementation of the Forward Euler method for this system in Python:

```python
def forward_Euler(T, dt, time, k, dx, T_0, T_N, save_interval):
    """
    Description: Use forward Euler's method to solve the heat equation
    Input: T (array of temperatures of inside points), dt (time step), time, k, dx, T_0, T_N, save_interval
    Output: Return a list containing the temperatures for all nodes at the specified save interval
    """
    T_history = [T.copy()]
    alpha = k / dx**2
    n = len(T)
    M = sparse.eye(n) + dt * alpha * DiffusionMatrix(n)
    b = np.zeros(n)
    b[0] = T_0
    b[-1] = T_N
    b *= alpha
    steps = int((time + dt) /dt)
    for i in range(steps):
        T = M @ T + dt * b
        if i != 0 and (i * dt) % save_interval == 0:
            T_history.append(T.copy())

    return T_history
```

Forward Euler is an explicit scheme meaning that in order to solve for the next point in time, only the infomration at the current point is used. While this is efficient, explicit schemes are susceptible to blowing up for a large $\Delta t$. Specifically, ____

## Method 2: Backward Euler (Implicit)

```python
def backward_Euler(T, dt, time, k, dx, T_0, T_N, save_interval):
    """
    Description: Use backward Euler's method to solve the heat equation
    Input: T (array of temperatures of inside points), dt (time step), time, k, dx, T_0, T_N, save_interval
    Output: Return a list containing the temperatures for all nodes at the specified save interval
    """
    # basically we want to solve Mx = c, where M = (I - dtA) and c = T + dtb,
    # where A and b are the perviously defined terms

    alpha = k/dx**2        # the coefficient to A and b
    n = len(T)             # number of interior points
    A = alpha * DiffusionMatrix(n)
    b = np.zeros(n)
    b[0] = T_0
    b[-1] = T_N
    b *= alpha

    M = sparse.eye(n) - dt * A
    T_history = [T.copy()]
    steps = int((time + dt) /dt)

    for i in range(steps):
        b_eff = T + dt * b
        T = sparse.linalg.spsolve(M, b_eff)
        if i != 0 and (i * dt) % save_interval == 0:
            T_history.append(T.copy())

    return T_history
```

## Method 3: Crank-Nicolson (Implicit & Standard)

```python
def Crank_Nicolson(T, dt, time, k, dx, T_0, T_N, save_interval):
    """
    Description: Use Crank-Nicolson method to solve the heat equation
    Input: T (array of all temperatures), dt (time step), time, k, dx, T_0, T_N, save_interval
    Output: Return a list containing the temperatures for all nodes at the specified save interval
    """
    T_history = [T.copy()]
    alpha = k/dx**2
    n = len(T)
    M = sparse.eye(n) - dt/2 * alpha * DiffusionMatrix(n)
    M2 = sparse.eye(n) + dt/2 * alpha * DiffusionMatrix(n)
    A_eff = M.copy()
    b = np.zeros(n)
    b[0] = T_0
    b[-1] = T_N
    b *= alpha

    steps = int((time + dt) /dt)
    for i in range(steps):
        b_eff = M2 @ T + dt/2 * (b + b)
        T = sparse.linalg.spsolve(A_eff, b_eff)
        if i != 0 and (i * dt) % save_interval == 0:
            T_history.append(T.copy())

    return T_history
```
 
Using the code above, we can model the temeprature of every interior point on our grid over time.

There are various ways to represent this system, but I represented this model using a simple animation.

<!-- {% include figure popup=true image_path="/assets/images/2D_heat_diffusion2.gif" alt="2d_heat" caption="Simple model of the diffusion of heat on a 2D surface" %} -->
