"""CRAX physics inside the shared compiled rollout and PPO update."""
import jax
import jax.numpy as jnp

from env.crax_point_crowd import CraxPointCrowd, CONTROL_METRICS
from trainer.jax_ppo_base_trainer import JaxPPOBaseTrainer
from trainer.jax_ppo_rollout import make_rollout_collector


class CraxPPOTrainer(JaxPPOBaseTrainer):
    def setup_env(self):
        super().setup_env()
        self.crax_env = CraxPointCrowd(self.config)
        self.rollout_collector = make_rollout_collector(self.model_module,
            use_qp=hasattr(self.model_module, "build_qp_context"),
            step_fn=self.crax_env.batch_step, reset_fn=self.crax_env.batch_reset,
            observe_fn=self.crax_env.batch_observe, timeout_aware=self.timeout_bootstrap)

    def _reset_training_env(self):
        state = jax.jit(lambda key: self.crax_env.batch_reset(None, key, self.num_envs))(self._next_rng_key())
        frame = jax.vmap(self.crax_env.observe)(state)
        history = jnp.repeat(frame[:, None], self.history_len, axis=1)
        return state, history, jnp.zeros(self.num_envs, jnp.float32)

    def eval(self, episodes=None):
        from trainer.crax_eval import evaluate_crax_point
        return evaluate_crax_point(self.config, self.train_state.params,
            episodes=int(self.config.trainer.eval_episodes if episodes is None else episodes),
            env=self.crax_env)

    def _extra_log_metrics(self, eval_metrics):
        if eval_metrics is None:
            return {}
        return {"eval/" + name: eval_metrics[name] for name in CONTROL_METRICS if name in eval_metrics}
