#include "bindings.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Release the GIL during kernel dispatch — these only touch tensors,
    // no Python objects, so multi-threaded callers can overlap.
    using pybind11::call_guard;
    using pybind11::gil_scoped_release;
    m.def("bilagrid_sample_forward", &bilagrid_sample_forward_tensor,
          call_guard<gil_scoped_release>());
    m.def("bilagrid_sample_backward", &bilagrid_sample_backward_tensor,
          call_guard<gil_scoped_release>());
    m.def("bilagrid_uniform_sample_forward", &bilagrid_uniform_sample_forward_tensor,
          call_guard<gil_scoped_release>());
    m.def("bilagrid_uniform_sample_backward", &bilagrid_uniform_sample_backward_tensor,
          call_guard<gil_scoped_release>());
    m.def("bilagrid_patched_sample_forward", &bilagrid_patched_sample_forward_tensor,
          call_guard<gil_scoped_release>());
    m.def("bilagrid_patched_sample_backward", &bilagrid_patched_sample_backward_tensor,
          call_guard<gil_scoped_release>());
    m.def("tv_loss_forward", &tv_loss_forward_tensor,
          call_guard<gil_scoped_release>());
    m.def("tv_loss_backward", &tv_loss_backward_tensor,
          call_guard<gil_scoped_release>());
}
