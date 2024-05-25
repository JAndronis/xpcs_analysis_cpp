#include "library.h"

void printProgress(double progress) {
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
    const Eigen::SparseMatrix<uint16_t, Eigen::RowMajor> &evts,
    const Eigen::ArrayX<uint64_t> &sqnc,
    const bool &verbose,
    const int &numthreads) {

    long w = sqnc.maxCoeff();
    Eigen::MatrixX<double> ttc = Eigen::MatrixX<double>::Ones(w + 1, w + 1);
    double progress = 0.0;

#pragma omp parallel for num_threads(numthreads) shared(ttc, evts, sqnc, progress, verbose) default(none)
    for (uint64_t i = 0; i < sqnc.size(); i++) {
        for (uint64_t j = i; j < sqnc.size(); j++) {
            auto dimg_t1 = Eigen::VectorX<uint16_t>(evts.row(i));
            auto dimg_t2 = Eigen::VectorX<uint16_t>(evts.row(j));
            double ttc_val = dimg_t1.dot(dimg_t2);
            double norm_t1 = dimg_t1.sum();
            double norm_t2 = dimg_t2.sum();
            ttc_val /= dimg_t1.size();
            norm_t1 /= dimg_t1.size();
            norm_t2 /= dimg_t2.size();
            ttc_val /= (norm_t1 * norm_t2);
            ttc(sqnc(i), sqnc(j)) = ttc_val;
            ttc(sqnc(j), sqnc(i)) = ttc_val;
        }
        if (verbose) {
            progress += 1.0 / static_cast<double>(sqnc.size());
            printProgress(progress);
        }
    }
    return ttc;
}
