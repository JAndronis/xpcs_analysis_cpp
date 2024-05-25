#ifndef XPCS_ANALYSIS_LIBRARY_H
#define XPCS_ANALYSIS_LIBRARY_H

#include <iostream>
#include <Eigen/Core>
#include <Eigen/Sparse>

namespace xpcs {
    Eigen::MatrixX<double> generateTTC(
        const Eigen::SparseMatrix<uint16_t, Eigen::RowMajor> &evts,
        const Eigen::ArrayX<uint64_t> &sqnc,
        const bool &verbose,
        const int &numthreads);
}

#endif //XPCS_ANALYSIS_LIBRARY_H
