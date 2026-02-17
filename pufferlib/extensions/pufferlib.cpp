#include <Python.h>
#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>
#include <vector>

extern "C" {
  /* Creates a dummy empty _C module that can be imported from Python.
     The import from Python will load the .so consisting of this file
     in this extension, so that the TORCH_LIBRARY static initializers
     below are run. */
  PyObject* PyInit__C(void)
  {
      static struct PyModuleDef module_def = {
          PyModuleDef_HEAD_INIT,
          "_C",   /* name of module */
          NULL,   /* module documentation, may be NULL */
          -1,     /* size of per-interpreter state of the module,
                     or -1 if the module keeps state in global variables. */
          NULL,   /* methods */
      };
      return PyModule_Create(&module_def);
  }
}

namespace pufferlib {

void puff_advantage_row(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon) {
    float lastpufferlam = 0;
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        float delta = rho_t*(rewards[t_next] + gamma*values[t_next]*nextnonterminal - values[t]);
        lastpufferlam = delta + gamma*lambda*c_t*lastpufferlam*nextnonterminal;
        advantages[t] = lastpufferlam;
    }
}

void puff_mo_advantage_row(float* values, float* rewards, float* dones,
        float* importance, float* advantages, float gamma, float lambda,
        float rho_clip, float c_clip, int horizon, int num_objectives) {
    float lastpufferlam[16] = {0};
    for (int t = horizon-2; t >= 0; t--) {
        int t_next = t + 1;
        float nextnonterminal = 1.0 - dones[t_next];
        float rho_t = fminf(importance[t], rho_clip);
        float c_t = fminf(importance[t], c_clip);
        for (int obj = 0; obj < num_objectives; obj++) {
            float delta = rho_t*(
                rewards[t_next * num_objectives + obj]
                + gamma*values[t_next * num_objectives + obj]*nextnonterminal
                - values[t * num_objectives + obj]
            );
            lastpufferlam[obj] = delta + gamma*lambda*c_t*lastpufferlam[obj]*nextnonterminal;
            advantages[t * num_objectives + obj] = lastpufferlam[obj];
        }
    }
}

void vtrace_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    for (const torch::Tensor& t : {values, rewards, dones, importance, advantages}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

void vtrace_mo_check(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        int num_steps, int horizon) {

    // Validate input tensors
    torch::Device device = values.device();
    int num_objectives = rewards.size(2);
    TORCH_CHECK(num_objectives <= 16, "Number of objectives must be <= 16, got ", num_objectives);
    for (const torch::Tensor& t : {dones, importance}) {
        TORCH_CHECK(t.dim() == 2, "Tensor must be 2D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
    for (const torch::Tensor& t : {values, rewards, advantages}) {
        TORCH_CHECK(t.dim() == 3, "Tensor must be 3D");
        TORCH_CHECK(t.device() == device, "All tensors must be on same device");
        TORCH_CHECK(t.size(0) == num_steps, "First dimension must match num_steps");
        TORCH_CHECK(t.size(1) == horizon, "Second dimension must match horizon");
        TORCH_CHECK(t.size(2) == num_objectives, "Third dimension must match num_objectives");
        TORCH_CHECK(t.dtype() == torch::kFloat32, "All tensors must be float32");
        if (!t.is_contiguous()) {
            t.contiguous();
        }
    }
}

// [num_steps, horizon]
void puff_advantage(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, const int horizon){
    for (int offset = 0; offset < num_steps*horizon; offset+=horizon) {
        puff_advantage_row(values + offset, rewards + offset,
            dones + offset, importance + offset, advantages + offset,
            gamma, lambda, rho_clip, c_clip, horizon
        );
    }
}

// [num_steps, horizon, num_objectives] for values, rewards, advantages
// [num_steps, horizon] for dones, importance
void puff_mo_advantage(float* values, float* rewards, float* dones, float* importance,
        float* advantages, float gamma, float lambda, float rho_clip, float c_clip,
        int num_steps, const int horizon, int num_objectives){
    int horizon_objectives = horizon * num_objectives;
    for (int i = 0; i < num_steps; i++) {
        int offset_2d = i * horizon;
        int offset_3d = i * horizon_objectives;
        puff_mo_advantage_row(values + offset_3d, rewards + offset_3d,
            dones + offset_2d, importance + offset_2d, advantages + offset_3d,
            gamma, lambda, rho_clip, c_clip, horizon, num_objectives
        );
    }
}

void compute_puff_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    vtrace_check(values, rewards, dones, importance, advantages, num_steps, horizon);
    puff_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, rho_clip, c_clip, num_steps, horizon
    );
}

void compute_puff_mo_advantage_cpu(torch::Tensor values, torch::Tensor rewards,
        torch::Tensor dones, torch::Tensor importance, torch::Tensor advantages,
        double gamma, double lambda, double rho_clip, double c_clip) {
    int num_steps = values.size(0);
    int horizon = values.size(1);
    int num_objectives = rewards.size(2);
    vtrace_mo_check(values, rewards, dones, importance, advantages, num_steps, horizon);
    puff_mo_advantage(values.data_ptr<float>(), rewards.data_ptr<float>(),
        dones.data_ptr<float>(), importance.data_ptr<float>(), advantages.data_ptr<float>(),
        gamma, lambda, rho_clip, c_clip, num_steps, horizon, num_objectives
    );
}

TORCH_LIBRARY(pufferlib, m) {
   m.def("compute_puff_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
   m.def("compute_puff_mo_advantage(Tensor(a!) values, Tensor(b!) rewards, Tensor(c!) dones, Tensor(d!) importance, Tensor(e!) advantages, float gamma, float lambda, float rho_clip, float c_clip) -> ()");
 }

TORCH_LIBRARY_IMPL(pufferlib, CPU, m) {
  m.impl("compute_puff_advantage", &compute_puff_advantage_cpu);
  m.impl("compute_puff_mo_advantage", &compute_puff_mo_advantage_cpu);
}

}
