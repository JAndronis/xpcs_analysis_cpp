#include "library.h"

void printProgress(const double &progress) {
    constexpr int barWidth = 50;
    const int pos = static_cast<int>(barWidth * progress);
    std::stringstream stream;
    stream << "[";
    for (int k = 0; k < barWidth; ++k) {
        if (k < pos)
            stream << "=";
        else if (k == pos)
            stream << ">";
        else
            stream << " ";
    }
    stream << "] " << progress * 100 << "%\r";
    std::cout << stream.rdbuf() << std::endl;
}

Eigen::MatrixX<double> xpcs::generateTTC(
    const Eigen::Ref<Eigen::ArrayXX<uint16_t>>& evts,
    const uint64_t &nframes,
    const bool &verbose) {

    Eigen::MatrixX<double> ttc = Eigen::MatrixX<double>::Ones(nframes, nframes);
    double progress = 0.0;

    #pragma omp parallel for shared(ttc, evts, progress, verbose, nframes) default(none)
    for (uint64_t i = 0; i < nframes; i++) {
        for (uint64_t j = i; j < nframes; j++) {
            auto dimg_t1 = Eigen::VectorX<uint16_t>(evts.col(i));
            auto dimg_t2 = Eigen::VectorX<uint16_t>(evts.col(j));
            double ttc_val = dimg_t1.dot(dimg_t2);
            double norm_t1 = dimg_t1.sum();
            double norm_t2 = dimg_t2.sum();
            ttc_val /= (norm_t1 * norm_t2);
            ttc(i, j) = ttc_val;
            ttc(j, i) = ttc_val;
        }
        if (verbose) {
            progress += 1.0 / static_cast<double>(nframes);
            printProgress(progress);
        }
    }
    return ttc;
}
