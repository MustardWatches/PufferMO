import math

import numpy as np

from pufferlib.ocean.kinematic_bicycle.kinematic_bicycle import (
    KinematicBicycle,
)
from pufferlib.ocean.kinematic_bicycle_integral.kinematic_bicycle_integral import (
    KinematicBicycleIntegral,
)


ZERO_ACTION = np.array([[0.0]], dtype=np.float32)


def test_base_deterministic_dynamics():
    """The native base environment must match one Euler dynamics step."""
    dt = 0.05
    velocity = 2.0
    initial_y = 0.2
    initial_theta = 0.1

    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        dt=dt,
        v=velocity,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=initial_theta,
        theta0_max=initial_theta,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        observations, _ = env.reset(seed=1)

        np.testing.assert_allclose(
            observations[0],
            [initial_y, initial_theta],
            rtol=0.0,
            atol=1e-7,
        )

        observations, rewards, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_y = (
            initial_y
            + velocity * math.sin(initial_theta) * dt
        )
        expected_theta = initial_theta

        np.testing.assert_allclose(
            observations[0],
            [expected_y, expected_theta],
            rtol=1e-6,
            atol=1e-7,
        )

        assert rewards.shape == (1,)
        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_integral_deterministic_update():
    """The integral observation must use the previous observed lateral error."""
    dt = 0.05
    velocity = 2.0
    initial_y = 0.2
    initial_theta = 0.1
    reference = 0.0

    env = KinematicBicycleIntegral(
        num_envs=1,
        seed=1,
        dt=dt,
        v=velocity,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=initial_theta,
        theta0_max=initial_theta,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        reference=reference,
        z_clip=5.0,
        lambda_value=0.0,
    )

    try:
        observations, _ = env.reset(seed=1)

        np.testing.assert_allclose(
            observations[0],
            [initial_y, initial_theta, 0.0],
            rtol=0.0,
            atol=1e-7,
        )

        observations, rewards, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_y = (
            initial_y
            + velocity * math.sin(initial_theta) * dt
        )
        expected_theta = initial_theta
        expected_z = dt * (reference - initial_y)

        np.testing.assert_allclose(
            observations[0],
            [expected_y, expected_theta, expected_z],
            rtol=1e-6,
            atol=1e-7,
        )

        assert rewards.shape == (1,)
        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_positive_and_negative_steering():
    """Opposite normalized actions must produce opposite heading changes."""
    dt = 0.05
    velocity = 2.0
    wheelbase = 2.5
    psi_max = 0.4

    env = KinematicBicycle(
        num_envs=2,
        seed=1,
        dt=dt,
        v=velocity,
        wheelbase=wheelbase,
        psi_max=psi_max,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        env.reset(seed=1)

        actions = np.array(
            [[1.0], [-1.0]],
            dtype=np.float32,
        )
        observations, _, terminals, truncations, _ = env.step(actions)

        expected_heading_change = (
            velocity
            / wheelbase
            * math.tan(psi_max)
            * dt
        )

        # y is updated using the heading at the beginning of the Euler step.
        np.testing.assert_allclose(
            observations[:, 0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            observations[:, 1],
            [expected_heading_change, -expected_heading_change],
            rtol=1e-6,
            atol=1e-7,
        )

        assert not np.any(terminals)
        assert not np.any(truncations)
    finally:
        env.close()


def test_actions_are_clamped_to_normalized_bounds():
    """Actions outside [-1, 1] must match the corresponding boundary actions."""
    env = KinematicBicycle(
        num_envs=2,
        seed=1,
        dt=0.05,
        v=2.0,
        wheelbase=2.5,
        psi_max=0.4,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        env.reset(seed=1)
        bounded_observations, bounded_rewards, _, _, _ = env.step(
            np.array([[1.0], [-1.0]], dtype=np.float32)
        )

        # Copy because PufferMO returns views of its reusable native buffers.
        bounded_observations = bounded_observations.copy()
        bounded_rewards = bounded_rewards.copy()

        env.reset(seed=1)
        excessive_observations, excessive_rewards, terminals, truncations, _ = (
            env.step(
                np.array([[2.0], [-2.0]], dtype=np.float32)
            )
        )

        np.testing.assert_allclose(
            excessive_observations,
            bounded_observations,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            excessive_rewards,
            bounded_rewards,
            rtol=0.0,
            atol=1e-7,
        )

        assert not np.any(terminals)
        assert not np.any(truncations)
    finally:
        env.close()


def test_steering_effectiveness_scales_applied_command():
    """k_psi must scale the steering angle before the bicycle dynamics."""
    dt = 0.05
    velocity = 2.0
    wheelbase = 2.5
    psi_max = 0.4
    k_psi = 0.5

    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        dt=dt,
        v=velocity,
        wheelbase=wheelbase,
        psi_max=psi_max,
        k_psi=k_psi,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        env.reset(seed=1)

        observations, _, terminals, truncations, _ = env.step(
            np.array([[1.0]], dtype=np.float32)
        )

        applied_psi = k_psi * psi_max
        expected_theta = (
            velocity
            / wheelbase
            * math.tan(applied_psi)
            * dt
        )

        np.testing.assert_allclose(
            observations[0],
            [0.0, expected_theta],
            rtol=1e-6,
            atol=1e-7,
        )

        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_two_step_action_delay():
    """A command delayed by two steps must be applied on the third step."""
    dt = 0.05
    velocity = 2.0
    wheelbase = 2.5
    psi_max = 0.4

    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        dt=dt,
        v=velocity,
        wheelbase=wheelbase,
        psi_max=psi_max,
        action_delay_steps=2,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        env.reset(seed=1)

        observations, _, terminals, truncations, _ = env.step(
            np.array([[1.0]], dtype=np.float32)
        )
        np.testing.assert_allclose(
            observations[0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )
        np.testing.assert_allclose(
            observations[0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_theta = (
            velocity
            / wheelbase
            * math.tan(psi_max)
            * dt
        )

        np.testing.assert_allclose(
            observations[0],
            [0.0, expected_theta],
            rtol=1e-6,
            atol=1e-7,
        )

        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_constant_lateral_disturbance():
    """A constant disturbance must add d_y * dt to lateral position."""
    dt = 0.05
    disturbance = 0.3

    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        dt=dt,
        v=2.0,
        y_max=100.0,
        max_steps=100,
        d_y=disturbance,
        regime_switch_step=-1,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        env.reset(seed=1)

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_y = disturbance * dt

        np.testing.assert_allclose(
            observations[0],
            [expected_y, 0.0],
            rtol=0.0,
            atol=1e-7,
        )

        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_disturbance_regime_switch():
    """The disturbance must switch after the configured episode step."""
    dt = 0.05
    initial_disturbance = 0.2
    switched_disturbance = -0.4

    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        dt=dt,
        v=2.0,
        y_max=100.0,
        max_steps=100,
        d_y=initial_disturbance,
        regime_switch_step=1,
        d_y_after_switch=switched_disturbance,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        env.reset(seed=1)

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_y_after_first_step = initial_disturbance * dt

        np.testing.assert_allclose(
            observations[0],
            [expected_y_after_first_step, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_y_after_second_step = (
            expected_y_after_first_step
            + switched_disturbance * dt
        )

        np.testing.assert_allclose(
            observations[0],
            [expected_y_after_second_step, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_failure_penalty_logging_and_automatic_reset():
    """Leaving the lane must terminate, penalize, log, and reset the episode."""
    initial_y = 0.2
    alive_bonus = 0.1
    w_y = 1.0
    failure_penalty = 100.0

    env = KinematicBicycle(
        num_envs=1,
        seed=7,
        dt=0.05,
        v=2.0,
        y_max=0.1,
        max_steps=100,
        d_y=0.0,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        w_y=w_y,
        w_theta=0.5,
        w_psi=0.01,
        alive_bonus=alive_bonus,
        failure_penalty=failure_penalty,
        report_interval=1,
    )

    try:
        observations, _ = env.reset(seed=7)

        np.testing.assert_allclose(
            observations[0],
            [initial_y, 0.0],
            rtol=0.0,
            atol=1e-7,
        )

        observations, rewards, terminals, truncations, info = env.step(
            ZERO_ACTION
        )

        expected_reward = (
            alive_bonus
            - w_y * initial_y**2
            - failure_penalty
        )

        np.testing.assert_allclose(
            rewards[0],
            expected_reward,
            rtol=1e-6,
            atol=1e-6,
        )

        assert bool(terminals[0])
        assert not bool(truncations[0])

        # The C environment resets immediately after the terminal transition.
        np.testing.assert_allclose(
            observations[0],
            [initial_y, 0.0],
            rtol=0.0,
            atol=1e-7,
        )

        assert len(info) == 1
        log = info[0]

        np.testing.assert_allclose(
            log["episode_return"],
            expected_reward,
            rtol=1e-6,
            atol=1e-6,
        )
        assert log["episode_length"] == 1.0
        assert log["failures"] == 1.0
        assert log["perf"] == 0.0
        assert log["n"] == 1.0
    finally:
        env.close()


def test_max_steps_uses_puffermo_terminal_compatibility_path():
    """The older binding reports the time limit as terminal, not truncation."""
    env = KinematicBicycle(
        num_envs=1,
        seed=11,
        dt=0.05,
        v=2.0,
        y_max=100.0,
        max_steps=2,
        d_y=0.0,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        report_interval=1,
    )

    try:
        env.reset(seed=11)

        observations, rewards, terminals, truncations, info = env.step(
            ZERO_ACTION
        )

        np.testing.assert_allclose(
            observations[0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rewards[0],
            0.1,
            rtol=1e-6,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])
        assert info == [{}]

        observations, rewards, terminals, truncations, info = env.step(
            ZERO_ACTION
        )

        np.testing.assert_allclose(
            rewards[0],
            0.1,
            rtol=1e-6,
            atol=1e-7,
        )
        assert bool(terminals[0])
        assert not bool(truncations[0])

        # The terminal step returns the reset observation.
        np.testing.assert_allclose(
            observations[0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )

        assert len(info) == 1
        log = info[0]

        np.testing.assert_allclose(
            log["episode_return"],
            0.2,
            rtol=1e-6,
            atol=1e-7,
        )
        assert log["episode_length"] == 2.0
        assert log["failures"] == 0.0
        assert log["perf"] == 1.0
        assert log["n"] == 1.0

        # A subsequent call belongs to a fresh episode.
        observations, _, terminals, truncations, info = env.step(
            ZERO_ACTION
        )

        np.testing.assert_allclose(
            observations[0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])
        assert info == [{}]
    finally:
        env.close()


def test_fourth_power_centerline_penalty():
    """w_center4 must add the configured quartic lateral-position penalty."""
    initial_y = 0.5
    alive_bonus = 0.1
    w_y = 1.0
    w_center4 = 2.0

    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        dt=0.05,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        w_y=w_y,
        w_theta=0.0,
        w_psi=0.0,
        alive_bonus=alive_bonus,
        w_center4=w_center4,
    )

    try:
        env.reset(seed=1)

        observations, rewards, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_reward = (
            alive_bonus
            - w_y * initial_y**2
            - w_center4 * initial_y**4
        )

        np.testing.assert_allclose(
            observations[0],
            [initial_y, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rewards[0],
            expected_reward,
            rtol=1e-6,
            atol=1e-7,
        )

        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_integral_lambda_reward_penalty():
    """The integral variant must penalize the updated integral state."""
    dt = 0.05
    initial_y = 0.5
    reference = 0.0
    lambda_value = 2.0
    alive_bonus = 0.1

    env = KinematicBicycleIntegral(
        num_envs=1,
        seed=1,
        dt=dt,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        w_y=0.0,
        w_theta=0.0,
        w_psi=0.0,
        alive_bonus=alive_bonus,
        w_center4=0.0,
        z_clip=5.0,
        reference=reference,
        lambda_value=lambda_value,
    )

    try:
        env.reset(seed=1)

        observations, rewards, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_z = dt * (reference - initial_y)
        expected_reward = alive_bonus - lambda_value * expected_z**2

        np.testing.assert_allclose(
            observations[0],
            [initial_y, 0.0, expected_z],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            rewards[0],
            expected_reward,
            rtol=1e-6,
            atol=1e-7,
        )

        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_integral_state_clipping():
    """The integral state must remain inside the configured clipping bounds."""
    z_clip = 0.02

    env = KinematicBicycleIntegral(
        num_envs=1,
        seed=1,
        dt=0.05,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=1.0,
        y0_max=1.0,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        z_clip=z_clip,
        reference=0.0,
        lambda_value=0.0,
    )

    try:
        env.reset(seed=1)

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        np.testing.assert_allclose(
            observations[0, 2],
            -z_clip,
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        np.testing.assert_allclose(
            observations[0, 2],
            -z_clip,
            rtol=0.0,
            atol=1e-7,
        )
        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_integral_nonzero_reference():
    """The integral update must track reference minus lateral position."""
    dt = 0.05
    initial_y = 0.25
    reference = 0.75

    env = KinematicBicycleIntegral(
        num_envs=1,
        seed=1,
        dt=dt,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        z_clip=5.0,
        reference=reference,
        lambda_value=0.0,
    )

    try:
        env.reset(seed=1)

        observations, _, terminals, truncations, _ = env.step(
            ZERO_ACTION
        )

        expected_z = dt * (reference - initial_y)

        np.testing.assert_allclose(
            observations[0],
            [initial_y, 0.0, expected_z],
            rtol=1e-6,
            atol=1e-7,
        )

        assert not bool(terminals[0])
        assert not bool(truncations[0])
    finally:
        env.close()


def test_reset_samples_stay_inside_configured_ranges():
    """Random initial states must remain inside their configured bounds."""
    y_min = -0.4
    y_max = 0.6
    theta_min = -0.15
    theta_max = 0.25

    env = KinematicBicycle(
        num_envs=128,
        seed=23,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        y0_min=y_min,
        y0_max=y_max,
        theta0_min=theta_min,
        theta0_max=theta_max,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        observations, _ = env.reset(seed=23)

        assert np.all(np.isfinite(observations))

        assert np.all(observations[:, 0] >= y_min)
        assert np.all(observations[:, 0] <= y_max)
        assert np.all(observations[:, 1] >= theta_min)
        assert np.all(observations[:, 1] <= theta_max)

        # Confirm that reset randomization is active across the vector.
        assert np.ptp(observations[:, 0]) > 0.1
        assert np.ptp(observations[:, 1]) > 0.05
    finally:
        env.close()


def test_zero_sensor_noise_is_deterministic_for_fixed_state():
    """With fixed state bounds and zero noise, observations must be exact."""
    initial_y = 0.25
    initial_theta = -0.1

    env = KinematicBicycle(
        num_envs=16,
        seed=31,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=initial_theta,
        theta0_max=initial_theta,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
    )

    try:
        first_observations, _ = env.reset(seed=31)
        first_observations = first_observations.copy()

        second_observations, _ = env.reset(seed=31)

        expected = np.tile(
            np.array([initial_y, initial_theta], dtype=np.float32),
            (env.num_agents, 1),
        )

        np.testing.assert_allclose(
            first_observations,
            expected,
            rtol=0.0,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            second_observations,
            expected,
            rtol=0.0,
            atol=1e-7,
        )
    finally:
        env.close()


def test_nonzero_sensor_noise_has_reasonable_statistics():
    """Configured Gaussian sensor noise must vary around the true state."""
    initial_y = 0.25
    initial_theta = -0.1
    y_noise_std = 0.1
    theta_noise_std = 0.05

    env = KinematicBicycle(
        num_envs=512,
        seed=41,
        v=0.0,
        y_max=100.0,
        max_steps=100,
        y0_min=initial_y,
        y0_max=initial_y,
        theta0_min=initial_theta,
        theta0_max=initial_theta,
        sensor_noise_y_std=y_noise_std,
        sensor_noise_theta_std=theta_noise_std,
    )

    try:
        observations, _ = env.reset(seed=41)

        assert np.all(np.isfinite(observations))

        observed_y_mean = float(np.mean(observations[:, 0]))
        observed_theta_mean = float(np.mean(observations[:, 1]))
        observed_y_std = float(np.std(observations[:, 0]))
        observed_theta_std = float(np.std(observations[:, 1]))

        # The samples should vary and remain statistically near the true state.
        assert 0.03 < observed_y_std < 0.2
        assert 0.015 < observed_theta_std < 0.1

        assert abs(observed_y_mean - initial_y) < 0.03
        assert abs(observed_theta_mean - initial_theta) < 0.015
    finally:
        env.close()


def test_multi_environment_vector_execution():
    """A native vector must update all environments independently."""
    num_envs = 32

    env = KinematicBicycle(
        num_envs=num_envs,
        seed=51,
        dt=0.05,
        v=2.0,
        wheelbase=2.5,
        psi_max=0.4,
        y_max=100.0,
        max_steps=100,
        d_y=0.0,
        y0_min=-0.5,
        y0_max=0.5,
        theta0_min=0.0,
        theta0_max=0.0,
        sensor_noise_y_std=0.0,
        sensor_noise_theta_std=0.0,
        report_interval=10_000,
    )

    try:
        observations, _ = env.reset(seed=51)

        assert observations.shape == (num_envs, 2)
        assert np.all(np.isfinite(observations))
        assert np.ptp(observations[:, 0]) > 0.1

        actions = np.linspace(
            -1.0,
            1.0,
            num_envs,
            dtype=np.float32,
        ).reshape(num_envs, 1)

        observations, rewards, terminals, truncations, info = env.step(
            actions
        )

        assert observations.shape == (num_envs, 2)
        assert rewards.shape == (num_envs,)
        assert terminals.shape == (num_envs,)
        assert truncations.shape == (num_envs,)

        assert np.all(np.isfinite(observations))
        assert np.all(np.isfinite(rewards))
        assert not np.any(terminals)
        assert not np.any(truncations)
        assert info == []

        # Initial heading is fixed at zero, so steering sign determines
        # the sign of the first heading update.
        assert np.all(observations[: num_envs // 2, 1] < 0.0)
        assert np.all(observations[num_envs // 2 :, 1] > 0.0)
    finally:
        env.close()


def test_repeated_create_reset_step_close_lifecycle():
    """Repeated delayed-buffer allocation and cleanup must remain stable."""
    environment_classes = (
        KinematicBicycle,
        KinematicBicycleIntegral,
    )

    for environment_class in environment_classes:
        for cycle in range(100):
            env = environment_class(
                num_envs=4,
                seed=cycle,
                dt=0.05,
                v=2.0,
                y_max=100.0,
                max_steps=20,
                action_delay_steps=3,
                y0_min=-0.2,
                y0_max=0.2,
                theta0_min=-0.1,
                theta0_max=0.1,
                sensor_noise_y_std=0.0,
                sensor_noise_theta_std=0.0,
                report_interval=10_000,
            )

            try:
                for reset_index in range(3):
                    observations, _ = env.reset(
                        seed=cycle + reset_index
                    )
                    assert np.all(np.isfinite(observations))

                    for _ in range(5):
                        observations, rewards, terminals, truncations, _ = (
                            env.step(
                                np.zeros(
                                    (4, 1),
                                    dtype=np.float32,
                                )
                            )
                        )

                        assert np.all(np.isfinite(observations))
                        assert np.all(np.isfinite(rewards))
                        assert np.all(np.isfinite(terminals))
                        assert np.all(np.isfinite(truncations))
            finally:
                env.close()


def test_long_vector_stress_execution():
    """Both native variants must survive a long randomized vector run."""
    rng = np.random.default_rng(61)

    environment_classes = (
        KinematicBicycle,
        KinematicBicycleIntegral,
    )

    for environment_class in environment_classes:
        env = environment_class(
            num_envs=128,
            seed=61,
            dt=0.05,
            v=2.0,
            wheelbase=2.5,
            psi_max=0.785,
            y_max=2.0,
            max_steps=500,
            d_y=0.15,
            regime_switch_step=100,
            d_y_after_switch=-0.2,
            k_psi=0.8,
            action_delay_steps=2,
            sensor_noise_y_std=0.02,
            sensor_noise_theta_std=0.01,
            y0_min=-0.5,
            y0_max=0.5,
            theta0_min=-0.2,
            theta0_max=0.2,
            report_interval=10_000,
        )

        try:
            observations, _ = env.reset(seed=61)
            assert np.all(np.isfinite(observations))

            for _ in range(5_000):
                actions = rng.uniform(
                    -1.5,
                    1.5,
                    size=(128, 1),
                ).astype(np.float32)

                observations, rewards, terminals, truncations, _ = env.step(
                    actions
                )

            assert np.all(np.isfinite(observations))
            assert np.all(np.isfinite(rewards))
            assert np.all(np.isfinite(terminals))
            assert np.all(np.isfinite(truncations))
        finally:
            env.close()


def test_base_accepts_shared_integral_config_options():
    """The base wrapper must accept integral-only shared config fields."""
    env = KinematicBicycle(
        num_envs=1,
        seed=1,
        v=0.0,
        y_max=100.0,
        y0_min=0.0,
        y0_max=0.0,
        theta0_min=0.0,
        theta0_max=0.0,
        z_clip=5.0,
        reference=0.0,
        lambda_value=0.0,
    )

    try:
        observations, _ = env.reset(seed=1)

        np.testing.assert_allclose(
            observations[0],
            [0.0, 0.0],
            rtol=0.0,
            atol=1e-7,
        )
    finally:
        env.close()


def test_wrappers_declare_scalar_reward_interface():
    """Both wrappers must explicitly identify as single-objective environments."""
    environment_classes = (
        KinematicBicycle,
        KinematicBicycleIntegral,
    )

    for environment_class in environment_classes:
        env = environment_class(
            num_envs=2,
            seed=1,
            v=0.0,
            y_max=100.0,
            y0_min=0.0,
            y0_max=0.0,
            theta0_min=0.0,
            theta0_max=0.0,
        )

        try:
            assert env.multiobjective_reward is False
            assert env.rewards.shape == (2,)
            assert env.rewards.dtype == np.float32
        finally:
            env.close()


def test_wrappers_expose_native_batch_metadata():
    """Native PufferEnv training expects the plural agents_per_batch field."""
    environment_classes = (
        KinematicBicycle,
        KinematicBicycleIntegral,
    )

    for environment_class in environment_classes:
        env = environment_class(
            num_envs=4,
            seed=1,
            v=0.0,
            y_max=100.0,
            y0_min=0.0,
            y0_max=0.0,
            theta0_min=0.0,
            theta0_max=0.0,
        )

        try:
            assert env.num_agents == 4
            assert env.agents_per_batch == 4
            assert env.agent_per_batch == 4
        finally:
            env.close()
