#ifndef XPCS_ANALYSIS_LIBRARY_H
#define XPCS_ANALYSIS_LIBRARY_H

#include <iostream>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace xpcs {
    Eigen::ArrayXX<float> generateTTC(const Eigen::Ref<Eigen::MatrixX<uint16_t>>&);
}

#endif //XPCS_ANALYSIS_LIBRARY_H
