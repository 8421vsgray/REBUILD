# DROT projection benchmark record

Date: 2026-10-20
GPU: NVIDIA GeForce RTX 3090
PyTorch: 2.6.0+cu124
Solver: `rho=4`, `solver_eps=1e-6`, `max_iterations=20000`
Repetitions: 30 after 3 warmups

This is the formal-size benchmark record. The training/evaluation batch-size
ceiling is 512; B=1024 is not included in formal training.

| Input | B | Adapter mean / median / P95 | Native DROT mean / median / P95 | Adapter feasibility |
| --- | ---: | ---: | ---: | ---: |
| random | 512 | 412.050 / 411.578 / 414.672 ms | 409.686 / 409.481 / 412.785 ms | 30/30 |
| GW-like | 512 | 22.565 / 22.560 / 22.623 ms | 20.101 / 20.089 / 20.157 ms | 30/30 |

Maximum observed relative row/column residuals were
`1.9884e-4 / 1.9747e-4` for random input and
`6.7949e-5 / 1.0848e-4` for GW-like input. Negative-plan violation was
zero in all records.

The benchmark is not an optimality certificate. It measures the complete
adapter and records feasibility; the independent Euclidean objective
validation remains documented in `SOLVER_VALIDATION.md`.
