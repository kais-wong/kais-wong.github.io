---
title: "2D Heat Equation"
date: 2025-06-10T15:34:30-04:00
categories:
  - projects
header:
  image: 
  teaser: /assets/images/heat/2D_heat_diffusion2.gif
tags:
  - Diffusion Equation
  - Python
layout: single
toc: true
toc_label: "Table of Contents"
toc_icon: "cog"
toc_sticky: true
---
Modeling the heat dispersion from the center of a metal plate in 2D.
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
Suppose we have a 2D rectangular sheet of metal. The center of the sheet is initalized at a positive temperature. All sides of this surface are held at a constant temperature, and the top and bottom of the sheet are insulated such that heat is not lost or transferred out of the system. How can we build a computational model of this system?
## Heat Equation
The heat equation is a derivative of the diffusion equation that describes the distribution of heat in a given space over time. It is a partial differential equation.

$$
\frac{\partial T}{\partial t} = \kappa \nabla^2 T
$$

However, since we wil only consider the heat equation in 2D we have the following:

$$\dfrac{\delta T}{\delta t} = \kappa (\dfrac{\delta^2 T}{\delta x^2} + \dfrac{\delta^2 T}{\delta y^2})$$

Where $\kappa$ is the diffusivity constant.

# Solution

## Discretizing in space

To solve this partial differential equation, we will discritize it in space using the central finite differences approach. By doing so we can rewrite our PDE as a system of coupled ODEs.

<!-- $$\frac{dT_{i,j}}{dt}=\kappa(\frac{T_{i-1,j}(t) - 2T_{i,j}(t) + T_{i+1,j}(t)}{(\Delta x)^2} + \frac{T_{i,j-1}(t) - 2T_{i,j}(t) + T_{i,j+1}(t)}{(\Delta y)^2})$$ -->

$$
\begin{align}
\frac{dT_{i,j}}{dt} =\ & \kappa \Bigg(
    \frac{T_{i-1,j}(t) - 2T_{i,j}(t) + T_{i+1,j}(t)}{(\Delta x)^2} \\
    &+ \frac{T_{i,j-1}(t) - 2T_{i,j}(t) + T_{i,j+1}(t)}{(\Delta y)^2}
\Bigg)
\end{align}
$$

## Forward Euler

We can now apply standard numerical methods to approximate the solution to this system of ODEs. Apply the forward euler method where we can approximate the next point in time $f^{n+1}$ using information about the current point $f^{n}$ and its derivative $\frac{df^{n}}{dt}$.

<!-- $$T^{n+1}_{i,j}=T^{n}_{i,j} + \Delta t \kappa (\frac{T_{i-1,j}(t) - 2T_{i,j}(t) + T_{i+1,j}(t)}{(\Delta x)^2} + \frac{T_{i,j-1}(t) - 2T_{i,j}(t) + T_{i,j+1}(t)}{(\Delta y)^2})$$ -->

$$
\begin{align*}
T^{n+1}_{i,j} =\ & T^{n}_{i,j} + \Delta t\, \kappa \Bigg(
    \frac{T_{i-1,j}(t) - 2T_{i,j}(t) + T_{i+1,j}(t)}{(\Delta x)^2} \\
    &+ \frac{T_{i,j-1}(t) - 2T_{i,j}(t) + T_{i,j+1}(t)}{(\Delta y)^2}
\Bigg)
\end{align*}
$$

<!-- For simplicity, assume that $\Delta x = \Delta y$.

$$T^{n+1}_{i,j}=T^{n}_{i,j} + \frac{\Delta t \kappa}{(\Delta x)^2} (T_{i-1,j}(t) + T_{i+1,j}(t) + T_next = {i,j-1}(t) + T_{i,j+1}(t) - 4T_{i,j}(t))$$ -->

We have now discitized the PDE in space and time. In essence, this represents the temperature of the discritized grid at a particular point in time. Now must now represent this in code.

## Code

Python allows us to express this operation simply using splicing. Let the term $\Delta t \kappa$ be alpha. Also, in order to simulate the temperature over time, we will iterate this operation from $t_{inital}$ to $t_{final}$

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

# Graphics

There are various ways to represent this system, but I represented this model using a simple animation.

{% include figure popup=true image_path="/assets/images/heat/2D_heat_diffusion2.gif" alt="2d_heat" caption="Simple model of the diffusion of heat on a 2D surface" %}


<!-- <figure style="margin-top: -2em; margin-bottom: 0.2em;">
  <img src="/assets/images/2D_heat_diffusion2.gif" alt="Shimizu Black Beach" width="300" />
  <figcaption style="margin-top: -1em;font-size:0.8em; color:#666;">清水の黒の海岸 | A black beach in Shimizu, Japan. </figcaption>
</figure> -->

<!-- You'll find this post in your `_posts` directory. Go ahead and edit it and re-build the site to see your changes. You can rebuild the site in many different ways, but the most common way is to run `jekyll serve`, which launches a web server and auto-regenerates your site when a file is updated.

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

Jekyll also offers powerful support for code snippets:

```ruby
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
```

Check out the [Jekyll docs][jekyll-docs] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll’s GitHub repo][jekyll-gh]. If you have questions, you can ask them on [Jekyll Talk][jekyll-talk].

[jekyll-docs]: https://jekyllrb.com/docs/home
[jekyll-gh]:   https://github.com/jekyll/jekyll
[jekyll-talk]: https://talk.jekyllrb.com/ -->
