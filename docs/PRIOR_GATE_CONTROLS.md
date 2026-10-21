# Prior-gate controls

The gate is evaluated per epoch. It is not a hard-coded 200-epoch lock.
When a lock is triggered, only the remaining batches of that epoch use
identity cross-view targets; the next epoch evaluates the gate again.

Use these fields in YAML:

~~~yaml
prior_gate:
  lock: true
  lockmass: 0.001
  raw_offdiag_tol: 0.001
  on_projection_failure: true
~~~

lock is the boolean master switch. When false, the gate precheck and epoch
locking are disabled; projection failures are raised instead of replaced.
lockmass is the final minimum off-diagonal mass that triggers a low-diffusion
lock. raw_offdiag_tol independently controls the pre-projection check and
inherits lockmass when omitted.

Legacy enabled and offdiag_tol fields remain supported. When both names are
present, lock overrides enabled and lockmass overrides offdiag_tol. Existing
YAML files without the new names therefore retain their previous behavior.

For free diffusion:

~~~yaml
prior_gate:
  lock: false
  lockmass: 0.001
~~~
