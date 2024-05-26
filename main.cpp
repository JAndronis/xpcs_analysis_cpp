//
// Created by Iason Andronis on 2024-05-07.
//

#include "library.h"

#include <argparse/argparse.hpp>
#include <highfive/highfive.hpp>
#include <highfive/eigen.hpp>
#include <iostream>

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
    program.add_argument("--max-images")
            .help("Maximum number of images to load")
            .default_value(-1)
            .action([](const std::string &value) { return std::stoi(value); });
    program.add_argument("--data-entry")
            .help("Data dataset")
            .default_value("");
    program.add_argument("--dt-entry")
            .help("Count time dataset")
            .default_value(DT);
    program.add_argument("--sequence-number")
            .help("Sequence number dataset")
            .default_value("");
    program.add_argument("--roi")
            .help("Rectangular region of interest: Ymin Ymax Xmin Xmax")
            .default_value(""); // 100 350 600 800

    try {
        program.parse_args(argc, argv);
    } catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    const auto filename = program.get<std::string>("--file");
    const auto output_file = program.get<std::string>("--output");
    const auto verbose = program.get<bool>("--verbose");
    const auto max_images = program.get<int>("--max-images");
    const auto data_entry = program.get<std::string>("--data-entry");
    const auto dt_entry = program.get<std::string>("--dt-entry");
    const auto sqnc_entry = program.get<std::string>("--sequence-number");
    const auto roi = program.get<std::string>("--roi");

    int number;
    std::stringstream iss(roi);
    std::vector<int> roi_vec;
    while (iss >> number)
        roi_vec.push_back(number);

    // open file
    const HighFive::File file(filename, HighFive::File::ReadOnly);

    // get image dataset
    const auto img_dataset = file.getDataSet(data_entry);
    const auto img_dataspace = img_dataset.getSpace();
    const auto img_dims = img_dataspace.getDimensions();
    uint64_t nframes = img_dims[0];

    // get sequence number
    Eigen::ArrayX<uint64_t> sqnc;
    bool sqnc_arg_empty = sqnc_entry.empty();
    bool max_imgs_arg = max_images > 0;

    if (not sqnc_arg_empty) {
        const auto sqnc_dataset = file.getDataSet(sqnc_entry);
        auto _sqnc = sqnc_dataset.read<std::vector<uint64_t>>();
    }

    if (max_imgs_arg && max_images < nframes) {
        nframes = max_images;
        sqnc.resize(max_images);
        if (not sqnc_arg_empty) {
            const auto sqnc_dataset = file.getDataSet(sqnc_entry);
            auto _sqnc = sqnc_dataset.read<std::vector<uint64_t> >();
            for (uint64_t i = 0; i < max_images; i++) {
                sqnc(i) = _sqnc[i];
            }
        } else {
            for (uint64_t i = 0; i < max_images; i++) {
                sqnc(i) = i;
            }
        }
    } else {
        sqnc = Eigen::ArrayX<uint64_t>::LinSpaced(nframes, 0, nframes-1);
    }

    // get dt
    const auto dt_dataset = file.getDataSet(dt_entry);
    auto dt = dt_dataset.read<double>();

    // roi
    if (not roi_vec.empty() && roi_vec.size() != 4) {
        std::cerr << "ROI must have 4 values" << std::endl;
        return 1;
    }

    if (roi_vec.empty()) {
        roi_vec = {0, EIGER_Y, 0, EIGER_X};
    }

    const auto ymin = static_cast<uint64_t>(roi_vec[0]);
    const auto ymax = static_cast<uint64_t>(roi_vec[1]);
    const auto xmin = static_cast<uint64_t>(roi_vec[2]);
    const auto xmax = static_cast<uint64_t>(roi_vec[3]);
    const uint64_t masked_row_size = ymax - ymin;
    const uint64_t masked_col_size = xmax - xmin;
    const auto _nframes = nframes;
    Eigen::ArrayXX<uint16_t> evnts(masked_row_size * masked_col_size, _nframes);
    Eigen::ArrayXX<uint16_t> total_img = Eigen::ArrayXX<uint16_t>::Zero(masked_row_size, masked_col_size);

    for (uint64_t k = 0; k < nframes; k++) {
        if (k % 1000 == 0 && verbose) {
            std::cout << "Loaded image " << k << std::endl;
        }
        std::vector<std::vector<std::vector<uint16_t> > > chunk;
        img_dataset.select({k, 0, 0}, {1, EIGER_Y, EIGER_X}).read(chunk);
        auto slice = chunk[0];
        #pragma omp parallel for
        for (uint64_t i = 0; i < masked_row_size; i++) {
            for (uint64_t j = 0; j < masked_col_size; j++) {
                auto slice_i = i + ymin;
                auto slice_j = j + xmin;
                if (slice[slice_i][slice_j] > 0) {
                    uint64_t _time = sqnc(k);
                    evnts(i * masked_col_size + j, _time) = slice[slice_i][slice_j];
                    total_img(i, j) += slice[slice_i][slice_j];
                }
            }
        }
        #pragma omp barrier
    }

    Eigen::MatrixX<double> ttc = xpcs::generateTTC(evnts, nframes, verbose);

    HighFive::File output(output_file, HighFive::File::ReadWrite | HighFive::File::Create | HighFive::File::Truncate);

    HighFive::DataSet dataset = output.createDataSet<double>("ttc", HighFive::DataSpace::From(ttc));
    dataset.write(ttc);

    HighFive::DataSet total_img_dataset = output.createDataSet<uint16_t>("total_img", HighFive::DataSpace::From(total_img));
    total_img_dataset.write(total_img);

    HighFive::DataSet dt_dataset_out = output.createDataSet<double>("dt", HighFive::DataSpace::From(dt));
    dt_dataset_out.write(dt);

    return 0;
}
