"""Shared JAX network layers and squashed-Gaussian policy functions."""

from __future__ import annotations

import math
import numpy as np

import jax
import jax.nn as jnn
import jax.numpy as jnp


def orthogonal_linear(key, in_dim, out_dim, gain):
    out_dim = int(out_dim)
    in_dim = int(in_dim)
    flat_shape = (out_dim, in_dim)
    matrix_shape = (in_dim, out_dim) if out_dim < in_dim else flat_shape
    matrix = np.asarray(jax.random.normal(key, matrix_shape, dtype=jnp.float32))
    q, r = np.linalg.qr(matrix)
    q *= np.sign(np.diag(r))
    if out_dim < in_dim:
        q = q.T
    weight = jnp.asarray(float(gain) * q[:out_dim, :in_dim], dtype=jnp.float32)
    bias = jnp.zeros((out_dim,), dtype=jnp.float32)
    return {"weight": weight, "bias": bias}


def torch_default_linear(key, in_dim, out_dim):
    weight_key, bias_key = jax.random.split(key)
    bound = 1.0 / math.sqrt(float(in_dim))
    return {
        "weight": jax.random.uniform(
            weight_key,
            (int(out_dim), int(in_dim)),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
        "bias": jax.random.uniform(
            bias_key,
            (int(out_dim),),
            minval=-bound,
            maxval=bound,
            dtype=jnp.float32,
        ),
    }


def init_linear(key, in_dim, out_dim, gain, use_init_weights):
    if bool(use_init_weights):
        return orthogonal_linear(key, in_dim, out_dim, gain)
    return torch_default_linear(key, in_dim, out_dim)


def dense(layer, x):
    """Apply a small linear layer without lowering it as a GEMM."""
    return jnp.sum(x[..., None, :] * layer["weight"], axis=-1) + layer["bias"]


def mlp(layers, x, act_name):
    for layer in layers[:-1]:
        x = activation(dense(layer, x), act_name)
    return dense(layers[-1], x)


def init_head(keys, in_dim, hidden_dim, out_dim, use_init_weights):
    sqrt2 = math.sqrt(2.0)
    k1, k2 = keys
    return [
        init_linear(k1, in_dim, hidden_dim, sqrt2, use_init_weights),
        init_linear(k2, hidden_dim, out_dim, sqrt2 * 0.01, use_init_weights),
    ]


def cfg_get(cfg, name, default=None):
    if isinstance(cfg, dict):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def require_config_value(cfg, name, expected, path):
    value = cfg_get(cfg, name, expected)
    if value != expected:
        raise ValueError(f"{path}.{name} is fixed to {expected!r}; got {value!r}")


def activation(x, act_name):
    act_name = str(act_name)
    if act_name == "relu":
        return jnn.relu(x)
    if act_name == "tanh":
        return jnp.tanh(x)
    if act_name == "silu":
        return jnn.silu(x)
    raise ValueError(f"unsupported activation: {act_name}")


def normalize_robot_type(robot_type):
    robot_type = str(robot_type).strip().lower()
    if robot_type not in {"single_integrator", "unicycle"}:
        raise ValueError(
            f"unsupported robot.type={robot_type!r}; expected 'single_integrator' or 'unicycle'"
        )
    return robot_type


def actor_std(params):
    logstd = params["actor"]["logstd"]
    logstd_min = math.log(1e-6)
    logstd_max = math.log(10.0)
    in_bounds = jnp.all((logstd >= logstd_min) & (logstd <= logstd_max))
    return jax.lax.cond(
        in_bounds,
        lambda value: jnp.clip(jnp.exp(value), 1e-6, 10.0),
        lambda value: jnp.exp(jnp.clip(value, logstd_min, logstd_max)),
        logstd,
    )


def action_scale_bias(params):
    action_low = jnp.asarray(params["action_low"], dtype=jnp.float32)
    action_high = jnp.asarray(params["action_high"], dtype=jnp.float32)
    return 0.5 * (action_high - action_low), 0.5 * (action_high + action_low)


def base_logprob_entropy_from_policy_action(params, mean, policy_action):
    policy_action = jnp.asarray(policy_action, dtype=mean.dtype)
    std = actor_std(params).astype(mean.dtype)
    log_std = jnp.log(std)
    log_prob = -0.5 * (((policy_action - mean) / std) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi))
    entropy = 0.5 + 0.5 * math.log(2.0 * math.pi) + log_std
    return jnp.sum(log_prob, axis=-1), jnp.broadcast_to(jnp.sum(entropy), (mean.shape[0],))


def policy_action_to_env_action(params, policy_action, return_squashed=False):
    scale, bias = action_scale_bias(params)
    policy_action = jnp.asarray(policy_action, dtype=jnp.float32)
    squashed = jnp.tanh(policy_action)
    action = bias + scale * squashed
    if return_squashed:
        return action, squashed
    return action


def env_action_to_policy_action(params, action):
    scale, bias = action_scale_bias(params)
    action = jnp.asarray(action, dtype=jnp.float32)
    squashed = (action - bias) / jnp.maximum(scale, 1e-6)
    squashed = jnp.clip(squashed, -1.0 + 1e-6, 1.0 - 1e-6)
    policy_action = 0.5 * (jnp.log1p(squashed) - jnp.log1p(-squashed))
    return policy_action, squashed


def squash_log_prob(params, base_log_prob, squashed):
    scale, _bias = action_scale_bias(params)
    tanh_logdet = jnp.log(1.0 - jnp.square(squashed) + 1e-6).sum(axis=-1)
    scale_logdet = jnp.log(jnp.maximum(scale, 1e-6)).sum()
    return base_log_prob - tanh_logdet - scale_logdet


def action_logprob_entropy(
    params,
    mean,
    action=None,
    noise=None,
    policy_action=None,
):
    """Evaluate or sample the shared squashed-Gaussian policy distribution."""
    sampled_policy_action = policy_action
    if action is None:
        if policy_action is not None:
            action, squashed = policy_action_to_env_action(params, policy_action, return_squashed=True)
            base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, policy_action)
            logprob = squash_log_prob(params, base_logprob, squashed)
        elif noise is not None:
            noise = jnp.asarray(noise, dtype=mean.dtype)
            sampled_policy_action = mean + actor_std(params).astype(mean.dtype) * noise
            action, squashed = policy_action_to_env_action(params, sampled_policy_action, return_squashed=True)
            base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, sampled_policy_action)
            logprob = squash_log_prob(params, base_logprob, squashed)
        else:
            raise ValueError("noise or policy_action must be provided when action is None")
    elif policy_action is not None:
        _action, squashed = policy_action_to_env_action(params, policy_action, return_squashed=True)
        base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, policy_action)
        logprob = squash_log_prob(params, base_logprob, squashed)
    else:
        inferred_policy_action, squashed = env_action_to_policy_action(params, action)
        base_logprob, entropy = base_logprob_entropy_from_policy_action(params, mean, inferred_policy_action)
        logprob = squash_log_prob(params, base_logprob, squashed)
    return action, logprob, entropy, sampled_policy_action
