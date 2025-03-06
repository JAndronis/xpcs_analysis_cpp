//
// Created by Iason Andronis on 2025-03-06.
//

#include "g2.h"

Eigen::ArrayX<float> xpcs::generateG2(const Eigen::Ref<Eigen::MatrixX<uint16_t>>& evts) {
    auto nframes = evts.cols();
    auto npixels = (float) evts.rows();
    Eigen::ArrayX<float> g2 = Eigen::ArrayX<float>::Ones(nframes);

    Eigen::VectorXf dimg_t0  = evts.col(0).cast<float>();
    #pragma omp parallel for shared(g2, dimg_t0)
    for (long tau=0; tau<nframes; tau++) {
        Eigen::VectorXf dimg_tau = evts.col(tau).cast<float>();
        float norm_t0 = dimg_t0.mean();
        float norm_tau = dimg_tau.mean();
        float denom = (norm_t0 * norm_tau) / (float) nframes;
        float nom = dimg_t0.dot(dimg_tau) / npixels / (float) nframes;
        g2(tau) = nom / denom;
    }
    return g2;
}