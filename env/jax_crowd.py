"""Shared functional crowd motion, observations and rewards for physical robots."""
import jax
import jax.numpy as jnp
from env.jax_obstacles import advance_obstacles
from env.jax_social_nav import compute_distances, fixed_order_policy_obs_from_state, reset_env, reward_done

def make_crowd_kernels(cfg):
    """Reuse source obstacle motion, reward arithmetic and observation signs."""
    def advance(state, rng):
        return advance_obstacles(cfg, state, rng)

    def finish(state, command, clearance):
        distance, _ = compute_distances(cfg, state)
        reward, done, success, collision, timeout, step, boundary = reward_done(
            cfg, state, distance, clearance, command, jnp.zeros((1,), jnp.float32))
        total = state.episode_returns + reward
        length = state.episode_lengths + 1
        state = state._replace(step_count=step, prev_dist_to_goal=distance,
                              episode_returns=total, episode_lengths=length)
        return state, reward, success, collision, timeout, boundary

    return {
        "advance": jax.jit(advance), "finish": jax.jit(finish),
        "observe": jax.jit(lambda state: fixed_order_policy_obs_from_state(cfg, state)),
        "reset": jax.jit(lambda rng: reset_env(cfg, rng, 1)),
    }
