
#include "library.h"

#include <iostream>

double xpcs::c2(const arma::SpCol<uint16_t> &img_t1,
                const arma::SpCol<uint16_t> &img_t2) {
    double nominator = arma::dot(img_t1, img_t2);
    double sum_t1 = arma::accu(img_t1);
    double sum_t2 = arma::accu(img_t2);
    double denominator = sum_t1 * sum_t2;
    if (denominator == 0) {
        return 0.;
    }
    double result = nominator / denominator;
    return result;
}

arma::mat xpcs::generateTTC(const arma::SpMat<uint16_t> &imgs,
                            const arma::Col<uint64_t> &sqnc,
                            const int &num_threads,
                            const bool &verbose) {
    arma::mat ttc(sqnc.n_elem, sqnc.n_elem);
    if (verbose)
        std::cout << "Generating TTC: " << std::endl;
    auto progress = 0.0;
    int barWidth = 50;
    #pragma omp parallel for num_threads(num_threads) shared(ttc, sqnc, verbose, imgs, progress, barWidth)
    for (uint64_t i=0; i<sqnc.n_elem; i++) {
        std::stringstream stream;
        if (verbose)
            stream << "[";
        for (uint64_t j=i; j<sqnc.n_elem; j++) {
            uint64_t ttc_i = sqnc(i);
            uint64_t ttc_j = sqnc(j);
            double ttc_val = xpcs::c2(imgs.col(i), imgs.col(j));
            ttc(ttc_i, ttc_j) = ttc_val;
            ttc(ttc_j, ttc_i) = ttc_val;
        }
        if (verbose) {
            int      pos = static_cast<int>(barWidth*progress);
            for (int k   = 0 ; k < barWidth ; ++k) {
                if (k < pos)
                    stream << "=";
                else if (k==pos)
                    stream << ">";
                else
                    stream << " ";
            }
            stream << "] " << progress*100 << "%\r";
            std::cout << stream.rdbuf();
            progress += 1.0/static_cast<double>(sqnc.n_elem);
        }
    }
    if (verbose)
        std::cout << std::endl;
    return ttc;
}