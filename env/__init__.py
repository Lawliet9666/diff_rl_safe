"""Navigation environments with a shared reset/step/observe interface."""


def make_env(config):
    backend = config.env.get("backend", "jax")
    if backend == "jax":
        if config.robot.get("simulator_model"):
            raise ValueError("robot=point requires env=crax_social_nav")
        from env.jax_env.jax_social_nav import JaxSocialNav
        return JaxSocialNav()
    if backend == "crax_mjx":
        from env.crax_env.crax_social_nav import CraxSocialNav
        return CraxSocialNav(config)
    raise ValueError(f"Unsupported environment backend {backend!r}; use jax or crax_mjx")
