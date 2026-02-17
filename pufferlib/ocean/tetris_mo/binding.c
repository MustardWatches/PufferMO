#include "tetris_mo.h"

#define Env TetrisMO
#define MY_PUT
#include "../env_binding_mo.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {
    env->n_rows = unpack(kwargs, "n_rows");
    env->n_cols = unpack(kwargs, "n_cols");
    env->use_deck_obs = unpack(kwargs, "use_deck_obs");
    env->n_noise_obs = unpack(kwargs, "n_noise_obs");
    env->n_init_garbage = unpack(kwargs, "n_init_garbage");
    env->max_ticks = unpack(kwargs, "max_ticks");
    env->freeze_on_done = unpack(kwargs, "freeze_on_done");
    env->gamma = unpack(kwargs, "gamma");
    init(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "episode_length", log->ep_length);
    assign_to_dict(dict, "episode_return", log->ep_return);
    assign_to_dict(dict, "scalarized_episode_return", log->scalarized_ep_return);
    assign_to_dict(dict, "episode_return_combo", log->ep_return_combo);
    assign_to_dict(dict, "episode_return_hard_drop", log->ep_return_hard_drop);
    assign_to_dict(dict, "episode_return_rotate", log->ep_return_rotate);
    assign_to_dict(dict, "discounted_episode_return", log->discounted_ep_return);
    assign_to_dict(dict, "discounted_scalarized_episode_return", log->discounted_scalarized_ep_return);
    assign_to_dict(dict, "discounted_episode_return_combo", log->discounted_ep_return_combo);
    assign_to_dict(dict, "discounted_episode_return_hard_drop", log->discounted_ep_return_hard_drop);
    assign_to_dict(dict, "discounted_episode_return_rotate", log->discounted_ep_return_rotate);
    assign_to_dict(dict, "avg_combo", log->avg_combo);
    assign_to_dict(dict, "lines_deleted", log->lines_deleted);
    assign_to_dict(dict, "game_level", log->game_level);
    assign_to_dict(dict, "ticks_per_line", log->ticks_per_line);
    assign_to_dict(dict, "weight_combo", log->weight_combo);
    assign_to_dict(dict, "weight_hard_drop", log->weight_hard_drop);
    assign_to_dict(dict, "weight_rotate", log->weight_rotate);

    // assign_to_dict(dict, "atn_frac_soft_drop", log->atn_frac_soft_drop);
    assign_to_dict(dict, "atn_frac_hard_drop", log->atn_frac_hard_drop);
    assign_to_dict(dict, "atn_frac_rotate", log->atn_frac_rotate);
    assign_to_dict(dict, "atn_frac_hold", log->atn_frac_hold);

    return 0;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    // Extract weights from kwargs
    PyObject* weights_obj = PyDict_GetItemString(kwargs, "weights");
    if (weights_obj == NULL) {
        PyErr_SetString(PyExc_KeyError, "Key 'weights' not found in kwargs");
        return 1;
    }

    // Validate it's a NumPy array
    if (!PyObject_TypeCheck(weights_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "weights must be a NumPy array");
        return 1;
    }

    PyArrayObject* weights_array = (PyArrayObject*)weights_obj;
    
    // Check contiguity
    if (!PyArray_ISCONTIGUOUS(weights_array)) {
        PyErr_SetString(PyExc_ValueError, "weights must be contiguous");
        return 1;
    }
    
    // Validate shape: must be 1D with REWARD_DIM elements
    npy_intp* dims = PyArray_DIMS(weights_array);
    if (PyArray_NDIM(weights_array) != 1 || dims[0] != REWARD_DIM) {
        PyErr_SetString(PyExc_ValueError, 
            "weights must be a 1D array with REWARD_DIM elements");
        return 1;
    }
    
    float* weights_data = (float*)PyArray_DATA(weights_array);
    for (int i = 0; i < REWARD_DIM; i++) {
        env->weights[i] = weights_data[i];
    }
    env->manual_weights = true;
    
    return 0;
}
