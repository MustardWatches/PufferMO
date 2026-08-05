#include "kinematic_bicycle_integral.h"

#define Env KinematicBicycle
#include "../env_binding.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    (void)args;

    env->num_agents = 1;
    env->rng = (unsigned int)unpack(kwargs, "seed");

    env->dt = unpack(kwargs, "dt");
    env->v = unpack(kwargs, "v");
    env->wheelbase = unpack(kwargs, "wheelbase");
    env->psi_max = unpack(kwargs, "psi_max");
    env->y_max = unpack(kwargs, "y_max");
    env->max_steps = (int)unpack(kwargs, "max_steps");

    env->d_y = unpack(kwargs, "d_y");
    env->regime_switch_step =
        (int)unpack(kwargs, "regime_switch_step");
    env->d_y_after_switch = unpack(kwargs, "d_y_after_switch");

    env->k_psi = unpack(kwargs, "k_psi");
    env->action_delay_steps =
        (int)unpack(kwargs, "action_delay_steps");

    env->sensor_noise_y_std =
        unpack(kwargs, "sensor_noise_y_std");
    env->sensor_noise_theta_std =
        unpack(kwargs, "sensor_noise_theta_std");

    env->y0_min = unpack(kwargs, "y0_min");
    env->y0_max = unpack(kwargs, "y0_max");
    env->theta0_min = unpack(kwargs, "theta0_min");
    env->theta0_max = unpack(kwargs, "theta0_max");

    env->w_y = unpack(kwargs, "w_y");
    env->w_theta = unpack(kwargs, "w_theta");
    env->w_psi = unpack(kwargs, "w_psi");
    env->alive_bonus = unpack(kwargs, "alive_bonus");
    env->failure_penalty = unpack(kwargs, "failure_penalty");
    env->w_center4 = unpack(kwargs, "w_center4");

    env->z_clip = unpack(kwargs, "z_clip");
    env->reference = unpack(kwargs, "reference");
    env->lambda_value = unpack(kwargs, "lambda");

    if (env->action_delay_steps < 0) {
        env->action_delay_steps = 0;
    }

    if (env->action_delay_steps > 0) {
        env->action_buffer = calloc(
            (size_t)env->action_delay_steps,
            sizeof(float)
        );
        if (env->action_buffer == NULL) {
            PyErr_SetString(
                PyExc_MemoryError,
                "Failed to allocate action delay buffer"
            );
            return -1;
        }
    }

    c_reset(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "failures", log->failures);
    assign_to_dict(dict, "n", log->n);
    return 0;
}
