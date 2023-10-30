# Solving Differential Equations

Solving differential equations is a long-standing problem in mathematics. While numerical
mathematics offers methods to compute solutions, these solutions are, for the most part, not given
by formulas but by numerical values on a grid. This also avoids the issue that for many
differential equations, an **exact** closed-form expression of their solutions does not even exist.
For practical purposes, though, solutions don't have to be exact. An approximate formula might
suffice and be superior to a purely numerical solution:

* compact
* more meaningful
* better suited for studying properties
* faster evaluated once obtained

An interesting approach for finding closed-form solutions was presented by Dario Izzo here:

http://darioizzo.github.io/dcgp/notebooks/solving_odes.html

The examples presented there are impressive, their runtime short enough to try for yourself. Here,
we basically take the same approach to tackle another well-known differential equation that is
slightly more difficult to solve. We look for exact or approximate closed-form solutions of the
**heat equation** on a ring in polar coordinates (r, phi) using weighted dCGP expressions.

Running `heat2d.py` produces screen outputs like

```
8021			21	[0.0795774715459477*exp(-0.25*r**2/t)/t]
    2.059833516372035e-35
```

with

* 8021 being the number of the experiment in which the solution was found
* 21 being the generation within this experiment in which the solution was found
* the term in square brackets being the solution (coefficient for thermal diffusivity set to 1)
* 2.059833516372035e-35 being the error of the solution.
