//
// Created by Iason Andronis on 2025-03-06.
//

#ifndef XPCS_ANALYSIS__G2_H
#define XPCS_ANALYSIS__G2_H

#include <omp.h>
#include <Eigen/Core>
#include <Eigen/Dense>

namespace xpcs {
    Eigen::ArrayX<float> generateG2(const Eigen::Ref<Eigen::MatrixX<uint16_t>>&);
}

#endif //XPCS_ANALYSIS__G2_H
