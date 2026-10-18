"""Float32 array constructors for MJX 3.11 when the policy enables JAX x64.

MJX assumes the process-wide JAX default is float32. Changing enable_x64
inside a nested JIT is unsafe in JAX 0.6 (invalid mixed-type StableHLO).
Instead adapt only MJX/CRAX physics modules' local NumPy namespaces. No
policy, QP, JAX global module, or installed source file is modified.
"""
import sys
import jax.numpy as jnp


class _PhysicsNumpy:
    _constructors = {"array", "asarray", "zeros", "ones", "full", "eye",
                     "empty", "zeros_like", "ones_like", "full_like", "empty_like",
                     "identity", "linspace", "logspace", "geomspace", "arange"}

    def __getattr__(self, name):
        fn = getattr(jnp, name)
        if name not in self._constructors:
            return fn

        def create(*args, **kwargs):
            result = fn(*args, **kwargs)
            if hasattr(result, "dtype") and result.dtype == jnp.float64:
                return result.astype(jnp.float32)
            return result
        return create


def configure_mjx_float32():
    proxy = _PhysicsNumpy()
    for name, module in tuple(sys.modules.items()):
        if name.startswith(("mujoco.mjx.", "crax.mjx.")) or name in {"crax.base", "crax.math"}:
            for alias in ("jp", "jnp"):
                if getattr(module, alias, None) is jnp:
                    setattr(module, alias, proxy)
