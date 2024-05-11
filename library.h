#ifndef XPCS_ANALYSIS_LIBRARY_H
#define XPCS_ANALYSIS_LIBRARY_H

#ifndef ARMA_USE_HDF5
#define ARMA_USE_HDF5
#endif

#include <iostream>
#include <armadillo>

const int EIGER_X = 1028;
const int EIGER_Y = 512;

namespace xpcs {

double c2(const arma::SpCol<uint16_t> &img_t1, const arma::SpCol<uint16_t> &img_t2);

arma::mat generateTTC(const arma::SpMat<uint16_t> &imgs, const arma::uvec &sqnc);

}

#endif //XPCS_ANALYSIS_LIBRARY_H
