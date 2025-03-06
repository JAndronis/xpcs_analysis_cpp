#ifndef XPCS_ANALYSIS_LIBRARY_H
#define XPCS_ANALYSIS_LIBRARY_H

#include <iostream>
#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace xpcs {
    Eigen::ArrayXX<double> generateTTC(const Eigen::Ref<Eigen::MatrixX<uint16_t>>& evts);
}

#endif //XPCS_ANALYSIS_LIBRARY_H
