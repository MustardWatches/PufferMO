import gymnasium
import numpy as np
import pufferlib
from pufferlib.ocean.tetris_mo import binding

class TetrisMO(pufferlib.PufferEnv):
    def __init__(
        self, 
        num_envs=1, 
        n_cols=10, 
        n_rows=20,
        use_deck_obs=True,
        n_noise_obs=10,
        n_init_garbage=4,
        render_mode=None, 
        log_interval=32,
        buf=None, 
        seed=0,
        freeze_on_done=False,
        max_ticks=0,        
        gamma=0.0
    ):
        self.single_observation_space = gymnasium.spaces.Box(low=0, high=1,
            shape=(n_cols*n_rows + 6 + 7 * 4 + n_noise_obs,), dtype=np.float32)
        self.single_action_space = gymnasium.spaces.Discrete(7)

        self.reward_dim = 3  # combo, hard_drop, rotate
        self.single_reward_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.reward_dim,), dtype=np.float32)

        self.render_mode = render_mode
        self.log_interval = log_interval
        self.num_agents = num_envs
        self.freeze_on_done = freeze_on_done

        super().__init__(buf, multiobjective_reward=True)
        self.n_cols = n_cols
        self.n_rows = n_rows
        self.c_envs = binding.vec_init(
            self.observations,
            self.actions,
            self.rewards,
            self.weights,
            self.terminals,
            self.truncations,
            num_envs,
            seed,
            n_cols=n_cols,
            n_rows=n_rows,
            use_deck_obs=use_deck_obs,
            n_noise_obs=n_noise_obs,
            n_init_garbage=n_init_garbage,
            freeze_on_done=freeze_on_done,
            max_ticks=max_ticks,            
            gamma=gamma,
        )
        if self.freeze_on_done:
            self.infos = [{} for _ in range(num_envs)]
            self.done_envs = [False for _ in range(num_envs)]               
 
    def reset(self, seed=0):
        binding.vec_reset(self.c_envs, seed)
        self.tick = 0
        return self.observations, []

    def step(self, actions):
        self.actions[:] = actions

        self.tick += 1
        binding.vec_step(self.c_envs)

        infos = []
        if self.freeze_on_done:
            infos = self.infos
            for i in range(self.num_agents):
                if self.done_envs[i]:
                    continue
                done = self.terminals[i] or self.truncations[i]
                if done:
                    self.done_envs[i] = True
                    log = binding.vec_log_single(self.c_envs, i)
                    infos[i] = log or {}
        else:
            if self.tick % self.log_interval == 0:
                infos.append(binding.vec_log(self.c_envs))

        return (self.observations, self.rewards, self.weights,
            self.terminals, self.truncations, infos)

    def render(self):
        binding.vec_render(self.c_envs, 0)

    def set_weights(self, weights):
        binding.vec_put(self.c_envs, weights=weights)

    def close(self):
        binding.vec_close(self.c_envs)

if __name__ == '__main__':
    TIME = 10
    num_envs = 4096
    env = Tetris(num_envs=num_envs)
    actions = [
        [env.single_action_space.sample() for _ in range(num_envs) ]for _ in range(1000)
    ]
    obs, _ = env.reset(seed = np.random.randint(0,1000))

    import time
    start = time.time()
    tick = 0
    
    while time.time() - start < TIME:
        action = actions[tick%1000]
        env.render()
        print(np.array(obs[0][0:200]).reshape(20,10), obs[0][200:206], obs[0][206:])
        obs, _, _, _, _ = env.step(action)
        tick += 1
    print('SPS:', (tick*num_envs) / (time.time() - start))
    env.close()

