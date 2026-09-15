import numpy as np
import gymnasium

import pufferlib
from pufferlib.ocean.kinematic_bicycle_integral import binding


class KinematicBicycleIntegral(pufferlib.PufferEnv):
    def __init__(
        self,
        num_envs=1,
        dt=0.05,
        v=2.0,
        wheelbase=2.5,
        psi_max=0.785,
        y_max=2.0,
        max_steps=500,
        d_y=0.0,
        regime_switch_step=-1,
        d_y_after_switch=0.0,
        k_psi=1.0,
        action_delay_steps=0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        y0_min=-0.5,
        y0_max=0.5,
        theta0_min=-0.2,
        theta0_max=0.2,
        w_y=1.0,
        w_theta=0.5,
        w_psi=0.01,
        alive_bonus=0.1,
        failure_penalty=100.0,
        w_center4=0.0,
        z_clip=5.0,
        reference=0.0,
        lambda_value=0.0,
        report_interval=1,
        render_mode="human",
        buf=None,
        seed=0,
    ):
        self.num_agents = num_envs
        self.agents_per_batch = num_envs
        self.report_interval = report_interval
        self.render_mode = render_mode
        self.tick = 0

        self.single_observation_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(3,),
            dtype=np.float32,
        )
        self.single_action_space = gymnasium.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

        super().__init__(buf, multiobjective_reward=False)
        self.multiobjective_reward = False
        self.actions = np.zeros(num_envs, dtype=np.float32)

        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            dt=dt,
            v=v,
            wheelbase=wheelbase,
            psi_max=psi_max,
            y_max=y_max,
            max_steps=max_steps,
            d_y=d_y,
            regime_switch_step=regime_switch_step,
            d_y_after_switch=d_y_after_switch,
            k_psi=k_psi,
            action_delay_steps=action_delay_steps,
            sensor_noise_y_std=sensor_noise_y_std,
            sensor_noise_theta_std=sensor_noise_theta_std,
            y0_min=y0_min,
            y0_max=y0_max,
            theta0_min=theta0_min,
            theta0_max=theta0_max,
            w_y=w_y,
            w_theta=w_theta,
            w_psi=w_psi,
            alive_bonus=alive_bonus,
            failure_penalty=failure_penalty,
            w_center4=w_center4,
            z_clip=z_clip,
            reference=reference,
            **{"lambda": lambda_value},
        )

    def reset(self, seed=None):
        self.tick = 0
        binding.vec_reset(self.c_envs, 0 if seed is None else seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = np.clip(actions, -1.0, 1.0).reshape(-1)
        self.tick += 1
        binding.vec_step(self.c_envs)

        info = []
        if self.tick % self.report_interval == 0:
            info.append(binding.vec_log(self.c_envs))

        return (
            self.observations,
            self.rewards,
            self.terminals,
            self.truncations,
            info,
        )

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def close(self):
        binding.vec_close(self.c_envs)
