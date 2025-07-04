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
---

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

$$\frac{\delta T}{\delta t} = \kappa \dfrac{\delta T^2}{\delta x^2}$$

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

The matrix A is actually the discrete representation of the $\frac{d^2}{dx^2}$. Addionally, due to the nature of this problem, we must add the Dirichlet boundary conditions using a term we will call $\vec{b}$

$$
\vec{b} = [T_0,\ 0,\ 0,\ \ldots,\ 0,\ T_N]^T
$$

We have now unified this system of ODEs into a single equation in matrix form.

$$
\frac{d\mathbf{T}(t)}{dt} = \frac{\kappa}{(\Delta x)^2} A\, \mathbf{T}(t) + \vec{b}
$$

## Method 1: Forward Euler (Explicit)

Now, by utilizing linear algebra, we have both represented our ODEs in a more legible manner, and one that allows us to concurrently solve the differential equation at each spacial node together in one step. 

We can now apply standard numerical methods to approximate the solution to this system of ODEs. Apply the forward euler method where we can approximate the next point in time $f^{n+1}$ using information about the current point $f^{n}$ and its derivative $\frac{df^{n}}{dt}$.

$$T^{n+1}_{i,j}=T^{n}_{i,j} + \Delta t \kappa (\frac{T_{i-1,j}(t) - 2T_{i,j}(t) + T_{i+1,j}(t)}{(\Delta x)^2})$$

<!-- For simplicity, assume that $\Delta x = \Delta y$.

$$T^{n+1}_{i,j}=T^{n}_{i,j} + \frac{\Delta t \kappa}{(\Delta x)^2} (T_{i-1,j}(t) + T_{i+1,j}(t) + T_next = {i,j-1}(t) + T_{i,j+1}(t) - 4T_{i,j}(t))$$ -->

We have now discitized the PDE in space and time. In essence, this represents the temperature of the discritized grid at a particular point in time. Now must now represent this in code.

Python allows us to express this operation simply using splicing. Let the term $\Delta t \kappa$ be alpha. Also, in order to simulate the temperature over time, will iterate this operation from $t_{inital}$ to $t_{final}$

```python
def forward_Euler2D(T, dt, time, k, dx, dy, T_0, T_N, save_interval):
    alpha = dt * k
    steps = int((time + dt) /dt)
    T_next = T.copy()
    T_history = [T.copy()]
    for step in range(steps):
        T_next[1:-1, 1:-1] = T[1:-1, 1:-1] + alpha * (
            (T[:-2, 1:-1] - 2 * T[1:-1, 1:-1] + T[2:, 1:-1]) / dx**2 +
            (T[1:-1, :-2] - 2 * T[1:-1, 1:-1] + T[1:-1, 2:]) / dy**2
        )
        T, T_nexr = T_next, T
        if step != 0 and step % (save_interval / dt) == 0:
            T_history.append(T.copy())
    return T_history
```

## Method 2: Backward Euler (Implicit)

## Method 3: Crank-Nicolson (Implicit & Standard)
 
Using the code above, we can model the temeprature of every interior point on our grid over time.

There are various ways to represent this system, but I represented this model using a simple animation.

{% include figure popup=true image_path="/assets/images/2D_heat_diffusion2.gif" alt="2d_heat" caption="Simple model of the diffusion of heat on a 2D surface" %}
