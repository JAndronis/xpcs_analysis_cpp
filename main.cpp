//
// Created by Iason Andronis on 2024-05-07.
//

#include "library.h"

#include <argparse/argparse.hpp>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>
#include <iostream>

typedef Eigen::Triplet<uint16_t> Triplet16;

const std::string FILE_NAME("/home/iason/CLionProjects/xpcs_analysis_cpp/data/scan_000181_eiger500k.hdf5");
const std::string DATA("entry/instrument/Eiger/data");
const std::string DT("entry/instrument/Eiger/count_time");
const std::string SEQUENCE_NUMBER("entry/instrument/Eiger/sequence_number");

constexpr int EIGER_X = 1028;
constexpr int EIGER_Y = 512;

int main(int argc, const char *argv[]) {
    argparse::ArgumentParser program("xpcs");
    program.add_argument("-f", "--file")
            .help("HDF5 file")
            .default_value(FILE_NAME);
    program.add_argument("-o", "--output")
            .help("Output file")
            .default_value("ttc.hdf5");
    program.add_argument("-v", "--verbose")
            .help("Verbose output")
            .default_value(false)
            .implicit_value(true);
    program.add_argument("-nt", "--num_threads")
            .help("Number of threads")
            .default_value(1)
            .action([](const std::string &value) { return std::stoi(value); });
    program.add_argument("--max-images")
            .help("Maximum number of images to load")
            .default_value(-1)
            .action([](const std::string &value) { return std::stoi(value); });

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto filename = program.get<std::string>("--file");
    auto output_file = program.get<std::string>("--output");
    auto verbose = program.get<bool>("--verbose");
    auto numthreads = program.get<int>("--num_threads");
    auto max_images = program.get<int>("--max-images");

    const HighFive::File file(filename, HighFive::File::ReadOnly);

    // get sequence number
    const auto sqnc_dataset = file.getDataSet(SEQUENCE_NUMBER);
    auto _sqnc = sqnc_dataset.read<std::vector<uint64_t> >();
    uint64_t max_imgs = _sqnc.size();

    // get dt
    const auto dt_dataset = file.getDataSet(DT);
    auto dt = dt_dataset.read<double>();

    // roi
    constexpr int ymin = 100;
    constexpr int ymax = 350;
    constexpr int xmin = 600;
    constexpr int xmax = 800;
    constexpr int masked_row_size = ymax - ymin;
    constexpr int masked_col_size = xmax - xmin;

    // get image dataset
    Eigen::ArrayX<uint64_t> sqnc;
    if (max_images > 0 && max_images < max_imgs) {
        sqnc.resize(max_images);
        for (int i = 0; i < max_images; i++) {
            sqnc(i) = _sqnc[i];
        }
        max_imgs = max_images;
    } else {
        sqnc.resize(max_imgs);
    }

    const auto img_dataset = file.getDataSet(DATA);
    const auto img_dataspace = img_dataset.getSpace();
    const auto img_dims = img_dataspace.getDimensions();
    Eigen::SparseMatrix<uint16_t> evnts(max_imgs, masked_row_size * masked_col_size);
    evnts.reserve(masked_row_size * masked_col_size * max_imgs);

    Eigen::ArrayXX<uint16_t> avg_img = Eigen::ArrayXX<uint16_t>::Zero(masked_row_size, masked_col_size);

    for (uint64_t k = 0; k < max_imgs; k++) {
        if (k % 1000 == 0 && verbose) {
            std::cout << "Loaded image " << k << std::endl;
        }
        if (k == max_imgs) {
            break;
        }
        std::vector<std::vector<std::vector<uint16_t> > > chunk;
        img_dataset.select({k, 0, 0}, {1, EIGER_Y, EIGER_X}).read(chunk);
        auto slice = chunk[0];
        for (int i = 0; i < masked_row_size; i++) {
            for (int j = 0; j < masked_col_size; j++) {
                auto slice_i = i + ymin;
                auto slice_j = j + xmin;
                if (slice[slice_i][slice_j] > 0) {
                    evnts.insert(k, i * masked_col_size + j) = slice[slice_i][slice_j];
                    avg_img(i, j) += slice[slice_i][slice_j];
                }
            }
        }
    }

    avg_img /= max_imgs;

    Eigen::MatrixX<double> ttc = xpcs::generateTTC(evnts, sqnc, verbose, numthreads);

    HighFive::File output(output_file, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    HighFive::DataSet dataset = output.createDataSet<double>("ttc", HighFive::DataSpace::From(ttc));
    dataset.write(ttc);

    HighFive::DataSet avg_img_dataset = output.createDataSet<uint16_t>("avg_img", HighFive::DataSpace::From(avg_img));
    avg_img_dataset.write(avg_img);

    HighFive::DataSet dt_dataset_out = output.createDataSet<double>("dt", HighFive::DataSpace::From(dt));
    dt_dataset_out.write(dt);

    return 0;
}
