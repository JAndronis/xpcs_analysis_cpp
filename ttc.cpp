
#include "ttc.h"

Eigen::ArrayXX<double> xpcs::generateTTC(const Eigen::Ref<Eigen::MatrixX<uint16_t>>& evts) {
    auto nframes = evts.cols();
    auto npixels = (double) evts.rows();
    std::cout<<"Npixels: "<<npixels<<std::endl;
    std::cout<<"Nframes: "<<nframes<<std::endl;
    Eigen::ArrayXX<double> ttc = Eigen::ArrayXX<double>::Ones(nframes, nframes);
    
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for shared(ttc, evts, nframes, npixels)
    for (long i = 0; i < nframes; i++) {
        for (long j = i; j < nframes; j++) {
            Eigen::VectorXd dimg_t1 = evts.col(i).cast<double>();
            Eigen::VectorXd dimg_t2 = evts.col(j).cast<double>();
            double dot = dimg_t1.dot(dimg_t2);
            double norm_t1 = dimg_t1.mean();
            double norm_t2 = dimg_t2.mean();
            ttc(i, j) = dot / (norm_t1*norm_t2) / npixels;
            ttc(j, i) = dot / (norm_t1*norm_t2) / npixels;
        }
    }
    return ttc;
}
