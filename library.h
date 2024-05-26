#ifndef XPCS_ANALYSIS_LIBRARY_H
#define XPCS_ANALYSIS_LIBRARY_H

#include <iostream>
#include <cstddef>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace xpcs {
    Eigen::MatrixX<double> generateTTC(
        const Eigen::Ref<Eigen::ArrayXX<uint16_t>>& evts,
        const uint64_t &nframes,
        const bool &verbose);
}

#endif //XPCS_ANALYSIS_LIBRARY_H
