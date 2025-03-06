//
// Created by Iason Andronis on 2025-02-26.
//

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include "ttc.h"

namespace py = pybind11;

PYBIND11_MODULE(xpcs_analysis_py, m){
    m.doc() = "xpcs analysis library";
    m.def("generateTTC",
          xpcs::generateTTC,
          py::arg("evts"), "evts: Event matrix",
          py::return_value_policy::reference_internal);
}