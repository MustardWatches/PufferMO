'''High-perf many-agent snake with multi-objective rewards. Inspired by snake env from https://github.com/dnbt777'''

import numpy as np
import gymnasium

import pufferlib
from pufferlib import APIUsageError
from pufferlib.ocean.snake_mo import binding

class SnakeMO(pufferlib.PufferEnv):
    def __init__(self, num_envs=16, width=640, height=360,
            num_snakes=256, num_food=4096,
            vision=5, leave_corpse_on_death=True,
            reward_food=0.1, reward_corpse=0.1, reward_death=-1.0,
            report_interval=128, max_snake_length=1024,
            render_mode='human', buf=None, seed=0,
            freeze_on_done=False, max_ticks=0, max_ticks_offset_mod=1, gamma=1):

        self.num_envs = num_envs
        self.freeze_on_done = freeze_on_done

        if num_envs is not None:
            num_snakes = num_envs * [num_snakes]
            width = num_envs * [width]
            height = num_envs * [height]
            num_food = num_envs * [num_food]
            leave_corpse_on_death = num_envs * [leave_corpse_on_death]

        self.num_snakes = num_snakes

        if not (len(num_snakes) == len(width) == len(height) == len(num_food)):
            raise APIUsageError('num_snakes, width, height, num_food must be lists of equal length')

        for w, h in zip(width, height):
            if w < 2*vision+2 or h < 2*vision+2:
                raise APIUsageError('width and height must be at least 2*vision+2')

        max_area = max([w*h for h, w in zip(height, width)])
        self.max_snake_length = min(max_snake_length, max_area)
        self.report_interval = report_interval

        self.single_observation_space = gymnasium.spaces.Box(
            low=0, high=2, shape=(2*vision+1, 2*vision+1), dtype=np.int8)
        self.single_action_space = gymnasium.spaces.Discrete(4)
        
        self.reward_dim = 3  # food, corpse, death
        self.single_reward_space = gymnasium.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.reward_dim,),
            dtype=np.float32
        )
        
        self.num_agents = sum(num_snakes)
        self.render_mode = render_mode
        self.tick = 0

        self.cell_size = int(np.ceil(1280 / max(max(width), max(height))))

        # Enable vector rewards
        super().__init__(buf, multiobjective_reward=True)
        c_envs = []
        offset = 0
        for i in range(num_envs):
            ns = num_snakes[i]
            obs_slice = self.observations[offset:offset+ns]
            act_slice = self.actions[offset:offset+ns]
            rew_slice = self.rewards[offset:offset+ns,:]
            weights_slice = self.weights[offset:offset+ns,:]
            term_slice = self.terminals[offset:offset+ns]
            trunc_slice = self.truncations[offset:offset+ns]
            # Seed each env uniquely: i + seed * num_envs
            env_seed = i + seed * num_envs
            env_id = binding.env_init(
                obs_slice, 
                act_slice, 
                rew_slice,
                weights_slice,
                term_slice, 
                trunc_slice,
                env_seed,
                width=width[i], 
                height=height[i],
                num_snakes=ns, 
                num_food=num_food[i],
                vision=vision, 
                leave_corpse_on_death=leave_corpse_on_death[i],
                reward_food=reward_food, 
                reward_corpse=reward_corpse,
                reward_death=reward_death, 
                max_snake_length=self.max_snake_length,
                cell_size=self.cell_size,
                freeze_on_done=freeze_on_done,
                max_ticks=max_ticks,
                max_ticks_offset_mod=max_ticks_offset_mod,
                gamma=gamma
            )
            c_envs.append(env_id)
            offset += ns
        self.c_envs = binding.vectorize(*c_envs)
        if self.freeze_on_done:
            self.infos = [{} for _ in range(num_envs)]
            self.done_envs = [False for _ in range(num_envs)]

    def reset(self, seed=None):
        self.tick = 0
        if seed is None:
            binding.vec_reset(self.c_envs, 0)
        else:
            binding.vec_reset(self.c_envs, seed)
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions
        self.tick += 1
        binding.vec_step(self.c_envs)

        infos = []
        if self.freeze_on_done:
            infos = self.infos
            for i in range(self.num_envs):
                if self.done_envs[i]:
                    continue
                ns = self.num_snakes[i]
                done = self.terminals[i*ns:(i+1)*ns].all() or self.truncations[i*ns:(i+1)*ns].all()
                if done:
                    self.done_envs[i] = True
                    log = binding.vec_log_single(self.c_envs, i)
                    infos[i] = log or {}
        else:
            if self.tick % self.report_interval == 0:
                infos.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards, self.weights,
            self.terminals, self.truncations, infos)

    def set_weights(self, weights):
        binding.vec_put(self.c_envs, weights=weights)

    def render(self):
        binding.vec_render(self.c_envs, self.cell_size)

    def close(self):
        binding.vec_close(self.c_envs)

def test_performance(timeout=10, atn_cache=1024):
    env = SnakeMO()
    env.reset()
    tick = 0

    total_snakes = env.num_agents
    actions = np.random.randint(0, 4, (atn_cache, total_snakes))

    import time
    start = time.time()
    while time.time() - start < timeout:
        atns = actions[tick % atn_cache]
        env.step(atns)
        tick += 1

    print(f'SPS: %f', total_snakes * tick / (time.time() - start))

if __name__ == '__main__':
    test_performance()
