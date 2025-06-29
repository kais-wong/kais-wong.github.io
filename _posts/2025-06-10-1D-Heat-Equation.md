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

<script
  src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"
  type="text/javascript">
</script>

# Background
## The Scenario
Suppose we have a 1D bar of metal. The left end of the bar is held at 0 degrees C and the right end of the bar is held at 10 degrees C. The sides of the bar are insulated such that heat is not lost or transferred out of the system. How can we build a computational model of this system?
## Heat Equation
The heat equation is a derivative of the diffusion equation that describes the distribution of heat in a given space over time. It is a partial differential equation.

$$
\frac{\partial T}{\partial t} = \kappa \nabla^2 T
$$

However, since we wil only consider the heat equation in 1D we have the following:

$$\dfrac{\delta T}{\delta t} = \kappa \dfrac{\delta T^2}{\delta x^2}$$

Where $\kappa$ is the diffusivity constant.

To solve this partial differential equation, we will discritize it in space using the central finite differences approach. By doing so we can rewrite our PDE as a system of coupled ODEs.

$$\frac{dT_{i,j}}{dt}=\kappa(\frac{T_{i-1,j}(t) - 2T_{i,j}(t) + T_{i+1,j}(t)}{(\Delta x)^2})$$

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
 
Using the code above, we can model the temeprature of every interior point on our grid over time.

There are various ways to represent this system, but I represented this model using a simple animation.

{% include figure popup=true image_path="/assets/images/2D_heat_diffusion2.gif" alt="2d_heat" caption="Simple model of the diffusion of heat on a 2D surface" %}
