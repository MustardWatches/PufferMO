#include "snake_mo.h"

#define Env CSnakeMO
#define MY_PUT
#include "../env_binding_mo.h"

static int my_init(Env* env, PyObject* args, PyObject* kwargs) {   
    env->width = unpack(kwargs, "width");
    env->height = unpack(kwargs, "height");
    env->num_snakes = unpack(kwargs, "num_snakes");
    env->vision = unpack(kwargs, "vision");
    env->leave_corpse_on_death = unpack(kwargs, "leave_corpse_on_death");
    env->food = unpack(kwargs, "num_food");
    env->reward_food = unpack(kwargs, "reward_food");
    env->reward_corpse = unpack(kwargs, "reward_corpse");
    env->reward_death = unpack(kwargs, "reward_death");
    env->max_snake_length = unpack(kwargs, "max_snake_length");
    env->cell_size = unpack(kwargs, "cell_size");
    env->max_ticks = unpack(kwargs, "max_ticks");
    env->max_ticks_offset_mod = unpack(kwargs, "max_ticks_offset_mod");
    env->freeze_on_done = unpack(kwargs, "freeze_on_done");    
    env->gamma = unpack(kwargs, "gamma");
    init_csnake(env);
    return 0;
}

static int my_log(PyObject* dict, Log* log) {
    assign_to_dict(dict, "perf", log->perf);
    assign_to_dict(dict, "score", log->score);
    assign_to_dict(dict, "episode_return", log->episode_return);
    assign_to_dict(dict, "episode_return_food", log->episode_return_food);
    assign_to_dict(dict, "episode_return_corpse", log->episode_return_corpse);
    assign_to_dict(dict, "episode_return_death", log->episode_return_death);
    assign_to_dict(dict, "discounted_episode_return", log->discounted_episode_return);
    assign_to_dict(dict, "discounted_episode_return_food", log->discounted_episode_return_food);
    assign_to_dict(dict, "discounted_episode_return_corpse", log->discounted_episode_return_corpse);
    assign_to_dict(dict, "discounted_episode_return_death", log->discounted_episode_return_death);
    assign_to_dict(dict, "scalarized_episode_return", log->scalarized_episode_return);
    assign_to_dict(dict, "discounted_scalarized_episode_return", log->discounted_scalarized_episode_return);
    assign_to_dict(dict, "weight_food", log->weight_food);
    assign_to_dict(dict, "weight_corpse", log->weight_corpse);
    assign_to_dict(dict, "weight_death", log->weight_death);
    assign_to_dict(dict, "episode_length", log->episode_length);
    assign_to_dict(dict, "n", log->n);
    return 0;
}

static int my_put(Env* env, PyObject* args, PyObject* kwargs) {
    PyObject* weights_obj = PyDict_GetItemString(kwargs, "weights");
    if (weights_obj == NULL) {
        PyErr_SetString(PyExc_KeyError, "Key 'weights' not found in kwargs");
        return 1;
    }

    if (!PyObject_TypeCheck(weights_obj, &PyArray_Type)) {
        PyErr_SetString(PyExc_TypeError, "weights must be a NumPy array");
        return 1;
    }

    PyArrayObject* weights_array = (PyArrayObject*)weights_obj;
    if (!PyArray_ISCONTIGUOUS(weights_array)) {
        PyErr_SetString(PyExc_ValueError, "weights must be contiguous");
        return 1;
    }
    
    npy_intp* dims = PyArray_DIMS(weights_array);
    if (PyArray_NDIM(weights_array) != 1 || dims[0] != REWARD_DIM) {
        PyErr_SetString(PyExc_ValueError, "weights must be a 1D array with 3 elements");
        return 1;
    }
    
    float* weights_data = (float*)PyArray_DATA(weights_array);
    for (int j = 0; j < env->num_snakes; j++) {
        for (int i = 0; i < REWARD_DIM; i++) {
            env->weights[j * REWARD_DIM + i] = weights_data[i];
        }
    }
    env->manual_weights = true;
    
    return 0;
}
