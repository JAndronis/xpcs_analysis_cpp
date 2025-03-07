
#include "ttc.h"

Eigen::ArrayXX<float> xpcs::generateTTC(const Eigen::Ref<Eigen::MatrixX<uint16_t>>& evts) {
    auto nframes = evts.cols();
    auto npixels = (double) evts.rows();
    Eigen::ArrayXX<float> ttc = Eigen::ArrayXX<float>::Ones(nframes, nframes);
    
    omp_set_num_threads(omp_get_max_threads());
    Eigen::setNbThreads(1);
    #pragma omp parallel for shared(ttc, evts, nframes, npixels)
    for (long i = 0; i < nframes; i++) {
        Eigen::VectorXf dimg_t1 = evts.col(i).cast<float>();
        float norm_t1 = dimg_t1.mean();
        for (long j = i; j < nframes; j++) {
            Eigen::VectorXf dimg_t2 = evts.col(j).cast<float>();
            float dot = dimg_t1.dot(dimg_t2);
            float norm_t2 = dimg_t2.mean();
            float value = dot / (norm_t1 * norm_t2) / npixels;
            ttc(i, j) = value;
            ttc(j, i) = value;
        }
    }
    return ttc;
}
