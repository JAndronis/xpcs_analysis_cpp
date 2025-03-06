//
// Created by Iason Andronis on 2024-05-07.
//

#include "ttc.h"

#include <iostream>

constexpr int EIGER_X = 300;
constexpr int EIGER_Y = 300;

int main(int argc, const char *argv[]) {
    auto nframes = 912;
    Eigen::MatrixX<uint16_t> evts = Eigen::MatrixX<uint16_t>::Random(EIGER_X*EIGER_Y, nframes);
//    auto result = xpcs::generateTTC(evts);
//    std::cout << "min:" << result.minCoeff() << std::endl;
//    std::cout << "max:" << result.maxCoeff() << std::endl;
    return 0;
}
