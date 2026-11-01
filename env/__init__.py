"""Common reset/step/observe environment interface."""


def make_env(config):
    backend = config.env.get("backend", "jax")
    if backend == "jax":
        from env.jax_env.jax_social_nav import JaxSocialNav
        return JaxSocialNav()
    if backend == "crax_mjx":
        from env.crax_env.crax_social_nav import CraxSocialNav
        return CraxSocialNav(config)
    raise ValueError(f"Unsupported environment backend {backend!r}; use jax or crax_mjx")


def finish_qp_failure(state, proposed, reward, done, metrics, failed, collision_penalty):
    """Discard failed rows' proposed transition; no placeholder action is committed."""
    import jax
    import jax.numpy as jnp

    metrics = dict(metrics)
    failed = (jnp.zeros_like(done, dtype=bool) if failed is None
              else jax.lax.stop_gradient(jnp.asarray(failed, dtype=bool)))
    if failed.shape != done.shape:
        raise ValueError("qp_failure must have one flag per environment")

    def select(old, new):
        if new.ndim == 0:
            return new
        mask = failed.reshape(failed.shape + (1,)*(new.ndim-failed.ndim))
        return jnp.where(mask, old, new)

    next_state = jax.tree_util.tree_map(select, state, proposed)
    terminal_reward = jnp.asarray(collision_penalty / 10., reward.dtype)
    metrics["episode_return"] = jnp.where(
        failed, metrics["episode_return"] - reward + terminal_reward,
        metrics["episode_return"])
    for key in ("success", "collision", "timeout", "truncated",
                "physical_contact", "boundary_violation", "robot_omega_delta", "robot_omega_delta_penalty"):
        if key in metrics:
            metrics[key] = jnp.where(failed, jnp.zeros_like(metrics[key]), metrics[key])
    if "terminated" in metrics:
        metrics["terminated"] = jnp.where(failed, jnp.ones_like(metrics["terminated"]), metrics["terminated"])
    metrics["qp_failure"] = failed.astype(jnp.float32)
    return next_state, jnp.where(failed, terminal_reward, reward), jnp.maximum(done, failed), metrics
