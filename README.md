# Riemannian Gradient Descent on the Sphere

Implementation of Riemannian Gradient Descent for optimization on the unit sphere S^{n-1}.

## What this does
This project implements RGD to minimize a function subject to ||x|| = 1 constraint. 
Key steps: project gradient to tangent space, take step, then retract back to sphere.

## How to run
```bash
python riemannian_gd.py
