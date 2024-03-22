This repo is a simplified version of the project that I did for the exam of the corse "Topics in high order accurate time integration methods" at SISSA. The codes in methods.py are based on the ones from the [official course repo](https://github.com/accdavlo/HighOrderODESolvers).
The project consisted in learning a score based model in pytorch, and then to solve the equivalent ODE with methods written by the student in JAX. Some notes:
- The conversion of the model from torch to jax/numpy brings a relative error of 1e-07. This causes a plateau in the convergence of the numerical methods.
- Jax is fast
- Explicit methods are faster then implicit methods.
- Implicit Dec is probably the best method.