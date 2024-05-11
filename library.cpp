
#include "library.h"

#include <iostream>

double xpcs::c2(const arma::SpCol<uint16_t> &img_t1,
                const arma::SpCol<uint16_t> &img_t2) {
    arma::SpCol<uint16_t> corr = img_t1 % img_t2;
    double sum_t1 = arma::accu(img_t1);
    double sum_t2 = arma::accu(img_t2);
    double denominator = sum_t1 * sum_t2;
    if (denominator == 0) {
        return 0.;
    }
    double nominator = arma::accu(corr);
    double result = nominator / denominator;
    return result;
}

arma::mat xpcs::generateTTC(const arma::SpMat<uint16_t> &imgs,
                            const arma::uvec &sqnc) {
    arma::mat ttc(sqnc.n_elem, sqnc.n_elem);
    std::cout << "Generating TTC: " << std::endl;
    auto progress = 0.0;
    int barWidth = 100;
    for (int i=0; i<sqnc.n_elem; i++) {
        std::cout << "[";
        for (int j=i; j<sqnc.n_elem; j++) {
            uint64_t ttc_i = sqnc(i);
            uint64_t ttc_j = sqnc(j);
            double ttc_val = xpcs::c2(imgs.col(i), imgs.col(j));
            ttc(ttc_i, ttc_j) = ttc_val;
            ttc(ttc_j, ttc_i) = ttc_val;
        }
        int pos = barWidth*progress;
        for (int k=0; k<barWidth ; ++k) {
            if (k < pos)
                std::cout << "=";
            else if (k==pos)
                std::cout << ">";
            else
                std::cout << " ";
        }
        std::cout << "] " << i << " / " << sqnc.n_elem << "\r";
        std::cout.flush();
        progress += 1.0/(float) sqnc.n_elem;
    }
    std::cout << std::endl;
    return ttc;
}