//
// Created by Iason Andronis on 2024-05-07.
//

#include "ttc.h"
#include "g2.h"

#include <iostream>

constexpr int EIGER_X = 300;
constexpr int EIGER_Y = 300;

int main(int argc, const char *argv[]) {
    auto nframes = 912;
    Eigen::ArrayXX<uint16_t> evts = Eigen::ArrayXX<uint16_t>::Zero(EIGER_X*EIGER_Y, nframes);
    evts.setRandom();
    auto result = xpcs::generateG2(evts);
    Eigen::Index minRow, minCol;
    Eigen::Index maxRow, maxCol;
    double min = result.minCoeff(&minRow, &minCol);
    double max = result.maxCoeff(&maxRow, &maxCol);
    std::cout << "min: " << min << std::endl;
    std::cout << "max: " << max << std::endl;
    return 0;
}
