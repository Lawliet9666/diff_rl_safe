"""External, fixed-parameter CBF-QP filter for JAX PPO rollouts."""
from dataclasses import dataclass
import math

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp_linalg

from env.robots.base import Obstacles
from solver.jax_qpth.qp import solve_qp


@dataclass(frozen=True)
class SafetyFilter:
    robot: object
    robot_type: str
    dt: float
    num_humans: int
    top_k: int
    slack_weight: float
    speed_facets: int
    max_iter: int
    eps: float

    def select_obstacles(self, state):
        distance = jnp.linalg.norm(state.human_positions-state.robot_state[:, None, :2], axis=-1)
        _, ids = jax.lax.top_k(-jnp.where(state.human_mask > .5, distance, jnp.inf), self.top_k)
        batch = jnp.arange(state.robot_state.shape[0])[:, None]
        return Obstacles(state.human_positions[batch, ids], state.human_vels[batch, ids],
                         state.human_radii[batch, ids], state.human_mask[batch, ids]), ids

    def _qp_data(self, state, nominal):
        """Assemble the physical projection; whiten variables for mixed input units."""
        obstacles, ids = self.select_obstacles(state)
        x = state.robot_state.astype(jnp.float64)
        obstacles = jax.tree_util.tree_map(lambda value: value.astype(jnp.float64), obstacles)
        constraints = jax.vmap(lambda x, obs: self.robot.cbf_constraints(x, obs, self.dt))(x, obstacles)
        batch, rows, dim = constraints.A.shape
        slack_dim = self.top_k if self.slack_weight > 0 else 0
        total = dim+slack_dim
        matrices = jax.vmap(self.robot.control_output_matrix)(x)
        weights = jnp.asarray(self.robot.weights, jnp.float64)
        tracking = jnp.swapaxes(matrices, 1, 2) @ (weights[None, :, None]*matrices)
        q = jnp.zeros((batch, total, total), jnp.float64).at[:, :dim, :dim].set(tracking)
        p = jnp.zeros((batch, total), jnp.float64).at[:, :dim].set(
            -(tracking @ nominal.astype(jnp.float64)[..., None]).squeeze(-1))
        slack_eye = jnp.eye(slack_dim, dtype=jnp.float64)
        padding = jnp.zeros((batch, rows, slack_dim), jnp.float64)
        if slack_dim:
            q = q.at[:, dim:, dim:].set(self.slack_weight*slack_eye)
            padding = padding.at[:, :slack_dim, :].set(-slack_eye)
        g = jnp.concatenate([-constraints.A, padding], axis=-1)
        h = constraints.b
        if slack_dim:
            nonnegative = jnp.concatenate([jnp.zeros((slack_dim, dim)), -slack_eye], axis=-1)
            g = jnp.concatenate([g, jnp.broadcast_to(nonnegative, (batch, slack_dim, total))], axis=1)
            h = jnp.concatenate([h, jnp.zeros((batch, slack_dim))], axis=1)
        low, high = self.robot.input_bounds()
        bounds = jnp.concatenate([jnp.eye(dim), -jnp.eye(dim)], axis=0)
        bounds = jnp.pad(bounds, ((0, 0), (0, slack_dim)))
        g = jnp.concatenate([g, jnp.broadcast_to(bounds, (batch, 2*dim, total))], axis=1)
        h = jnp.concatenate([h, jnp.broadcast_to(jnp.concatenate([high, -low]), (batch, 2*dim))], axis=1)
        if self.robot_type == "single_integrator":
            # Inscribed speed polygon keeps the existing step() norm limiter inactive.
            angles = jnp.arange(self.speed_facets)*2*jnp.pi/self.speed_facets
            normals = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
            polygon = jnp.pad(normals, ((0, 0), (0, slack_dim)))
            bound = self.robot.v_max*math.cos(math.pi/self.speed_facets)*(1.-1e-7)
            g = jnp.concatenate([g, jnp.broadcast_to(polygon, (batch, self.speed_facets, total))], axis=1)
            h = jnp.concatenate([h, jnp.full((batch, self.speed_facets), bound)], axis=1)
        transform = jsp_linalg.solve_triangular(
            jnp.swapaxes(jnp.linalg.cholesky(q), 1, 2),
            jnp.broadcast_to(jnp.eye(total), q.shape), lower=False)
        scaled_q = jnp.swapaxes(transform, 1, 2) @ q @ transform
        scaled_p = (jnp.swapaxes(transform, 1, 2) @ p[..., None]).squeeze(-1)
        scaled_g = g @ transform
        scale = jnp.maximum(jnp.linalg.norm(scaled_g, axis=-1), 1.)
        return dict(q=scaled_q, p=scaled_p, g=scaled_g/scale[..., None], h=h/scale,
                    transform=transform, raw_g=g, raw_h=h, constraints=constraints,
                    ids=ids, mask=obstacles.mask)

    def __call__(self, state, nominal_action, *, active=None):
        if nominal_action.shape != (state.robot_state.shape[0], self.robot.control_dim):
            raise ValueError("safety filter nominal action must have shape [batch, Robot.control_dim]")
        state, nominal_action = jax.tree_util.tree_map(jax.lax.stop_gradient, (state, nominal_action))
        if active is not None:
            # Completed evaluation episodes must not submit invalid terminal states to a QP.
            initial = self.robot.initial_state(jnp.zeros(3, state.robot_state.dtype))
            state = state._replace(
                robot_state=jnp.where(active[:, None], state.robot_state, initial),
                human_positions=jnp.where(active[:, None, None], state.human_positions, 0.),
                human_vels=jnp.where(active[:, None, None], state.human_vels, 0.),
                human_radii=jnp.where(active[:, None], state.human_radii, 0.),
                human_mask=jnp.where(active[:, None], state.human_mask, 0.))
            nominal_action = jnp.where(active[:, None], nominal_action, 0.)
        data = self._qp_data(state, nominal_action)
        empty = jnp.empty((0,), jnp.float64)
        # Match BarrierNet's first-order versus high-order solver settings.
        high_order = self.robot_type in {"double_integrator", "humanoid", "quadrotor"}
        solution, _, dual, _ = solve_qp(
            data['q'], data['p'], data['g'], data['h'], empty, empty,
            eps=self.eps, verbose=-1, notImprovedLim=self.max_iter if high_order else 3,
            maxIter=self.max_iter, check_Q_spd=False, safe_step=high_order)
        physical = (data['transform'] @ solution[..., None]).squeeze(-1)
        dim = self.robot.control_dim
        action = physical[:, :dim].astype(nominal_action.dtype)
        slack = jnp.maximum(physical[:, dim:], 0.)
        if self.slack_weight == 0:
            slack = jnp.zeros_like(data['mask'])
        applied_solution = jnp.concatenate([action.astype(jnp.float64), physical[:, dim:]], axis=1)
        primal = jnp.maximum(jnp.max(
            (data['raw_g'] @ applied_solution[..., None]).squeeze(-1)-data['raw_h'], axis=1), 0.)
        dual_residual = jnp.max(jnp.abs(
            (data['q'] @ solution[..., None]).squeeze(-1)+data['p']+
            (jnp.swapaxes(data['g'], 1, 2) @ dual[..., None]).squeeze(-1)), axis=1)
        distance = data['h']-(data['g'] @ solution[..., None]).squeeze(-1)
        complementarity = jnp.max(jnp.abs(distance*dual), axis=1)
        dual_feasibility = jnp.maximum(jnp.max(-dual, axis=1), 0.)
        errors = jnp.stack([primal, dual_residual, complementarity, dual_feasibility], axis=1)
        residual = (data['constraints'].A @ action.astype(jnp.float64)[..., None]).squeeze(-1)+data['constraints'].b
        count = jnp.maximum(data['mask'].sum(axis=1), 1.)
        correction = jnp.linalg.norm(action-nominal_action, axis=1)
        diagnostics = {
            'qp_ids': data['ids'], 'slack': slack, 'cbf_residual': residual[:, :self.top_k],
            'filter/slack': (slack*data['mask']).sum(axis=1)/count,
            'filter/slack_max': jnp.max(slack*data['mask'], axis=1),
            'filter/correction': correction,
            # Tolerances below affect diagnostic labels only.
            'filter/intervention': (correction > 1e-5).astype(jnp.float32),
            'filter/cbf_violation': jnp.maximum(jnp.max(-residual[:, :self.top_k]*data['mask'], axis=1), 0.),
            'filter/primal_residual': primal,
            'filter/kkt_residual': jnp.max(errors[:, 1:], axis=1),
            'filter/initial_safe': jnp.all(data['constraints'].domain >= -1e-5, axis=1).astype(jnp.float32),
        }
        return jax.tree_util.tree_map(jax.lax.stop_gradient, (action, diagnostics))


def make_safety_filter(config, env_cfg):
    """Build only an explicitly enabled, ordinary-PPO JAX execution filter."""
    section = config.trainer.safety_filter
    if not isinstance(section.enabled, bool):
        raise ValueError("trainer.safety_filter.enabled must be true or false")
    if not section.enabled:
        return None
    if config.model.type not in {"ppo_mlp", "ppo_gnn"}:
        raise ValueError("external safety filter requires PPO-MLP or PPO-GNN; policy QP layers remain separate")
    from env.jax_env.jax_social_nav import JaxSocialNavConfig
    if not isinstance(env_cfg, JaxSocialNavConfig):
        raise ValueError("external safety filter currently requires the JAX environment")
    if not jax.config.x64_enabled:
        raise ValueError("external safety filter requires JAX_ENABLE_X64=true")
    if config.robot.type == 'humanoid' and env_cfg.robot.barrier != 'distance':
        raise ValueError("SF-PPO humanoid requires the distance HOCBF")
    top_k = section.qp_obs_top_k
    if top_k is not None and (not isinstance(top_k, int) or isinstance(top_k, bool) or top_k <= 0):
        raise ValueError("trainer.safety_filter.qp_obs_top_k must be null or a positive integer")
    for name, minimum in (("max_iter", 1), ("speed_facets", 4)):
        value = section[name]
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"trainer.safety_filter.{name} must be an integer >= {minimum}")
    for name in ('eps', 'slack_weight'):
        value = section[name]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) < 0 or (name != 'slack_weight' and value == 0):
            raise ValueError(f"invalid trainer.safety_filter.{name}")
    return SafetyFilter(env_cfg.robot, str(config.robot.type), env_cfg.dt, env_cfg.num_humans,
                        min(top_k, env_cfg.num_humans) if top_k is not None else env_cfg.num_humans,
                        float(section.slack_weight), section.speed_facets, section.max_iter,
                        float(section.eps))
