//
// Created by Iason Andronis on 2024-05-07.
//

#include "library.h"

#include <tuple>

#include <argparse/argparse.hpp>
#include <H5Cpp.h>

const H5std_string FILE_NAME("../data/scan_000181_eiger500k.hdf5");
const H5std_string DATA("entry/instrument/Eiger/data");
const H5std_string DT("entry/instrument/Eiger/count_time");
const H5std_string SEQUENCE_NUMBER("entry/instrument/Eiger/sequence_number");

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

    try {
        program.parse_args(argc, argv);
    }
    catch (const std::exception &err) {
        std::cerr << err.what() << std::endl;
        std::cerr << program;
        return 1;
    }

    auto filename    = program.get<std::string>("--file");
    auto output_file = program.get<std::string>("--output");
    auto verbose     = program.get<bool>("--verbose");
    auto numthreads  = program.get<int>("--num_threads");

    std::vector<std::tuple<uint64_t, uint64_t>> blocks(numthreads);

    try {

        H5::H5File file(filename, H5F_ACC_RDONLY);

        // sequence dataset
        H5::DataSet   sqnc_dataset   = file.openDataSet(SEQUENCE_NUMBER);
        H5::DataSpace sqnc_filespace = sqnc_dataset.getSpace();
        int           sqnc_rank;  // should be 1
        hsize_t       sqnc_dims[1];
        sqnc_rank = sqnc_filespace.getSimpleExtentDims(sqnc_dims);

        H5::DataSpace         sqnc_memspace(sqnc_rank, sqnc_dims);
        const hsize_t         max_imgs       = sqnc_dims[0];
        constexpr hsize_t     sqnc_offset[1] = {0};
        const hsize_t         sqnc_count[1]  = {max_imgs};
        std::vector<uint64_t> sqnc(max_imgs);

        sqnc_filespace.selectHyperslab(H5S_SELECT_SET, sqnc_count, sqnc_offset);
        sqnc_dataset.read(sqnc.data(), H5::PredType::NATIVE_INT64, sqnc_memspace, sqnc_filespace);

        arma::Col<uint64_t> sqnc_arma(sqnc);
        sqnc_dataset.close();

        uint64_t counter = 0;
        for (auto &i: blocks) {
            i = std::tuple<uint64_t, uint64_t>(counter, counter + sqnc_arma.n_elem / numthreads);
            counter += sqnc_arma.n_elem / numthreads;
        }


        // get dt
        H5::DataSet dt_dataset = file.openDataSet(DT);
        float       dt;
        dt_dataset.read(&dt, H5::PredType::NATIVE_FLOAT, H5S_ALL, H5S_ALL);
        dt_dataset.close();
        auto time_axis = arma::regspace(dt, dt, dt*max_imgs);

        // image dataset
        H5::DataSet           img_dataset = file.openDataSet(DATA);
        H5::DSetCreatPropList cparms      = img_dataset.getCreatePlist();

        // chunk space
        hsize_t       chunk_dims[3];
        const int     rank_chunk = cparms.getChunk(3, chunk_dims);
        H5::DataSpace chunk_space(rank_chunk, chunk_dims);

        // file space
        H5::DataSpace img_filespace = img_dataset.getSpace();

        auto ymin            = 100;
        auto ymax            = 350;
        auto xmin            = 600;
        auto xmax            = 800;
        auto masked_row_size = ymax - ymin;
        auto masked_col_size = xmax - xmin;

        arma::SpMat<uint16_t> imgs(masked_row_size*masked_col_size, max_imgs);

        arma::Mat<uint16_t> avg_img(EIGER_X, EIGER_Y, arma::fill::zeros);
        arma::Mat<uint16_t> avg_roi(masked_row_size, masked_col_size, arma::fill::zeros);
        hsize_t             offset[3] = {0, 0, 0};

        for (hsize_t k = 0 ; k < max_imgs ; k++) {

            if (k%1000==0 && verbose) {
                std::cout << "Loaded image " << k << std::endl;
            }
            offset[0] = k;
            img_filespace.selectHyperslab(H5S_SELECT_SET, chunk_dims, offset);

            // allocate buffer
            uint16_t data[EIGER_Y][EIGER_X];
            img_dataset.read(data, H5::PredType::NATIVE_UINT16, chunk_space, img_filespace);
            arma::Mat<uint16_t> img(&data[0][0], EIGER_X, EIGER_Y, false, true);
//            for (int i = 0 ; i < EIGER_Y ; i++) {
//                for (int j = 0 ; j < EIGER_X ; j++) {
//                    img(i, j) = data[i][j];
//                }
//            }
            arma::Mat<uint16_t> sub_img = img.submat(xmin, ymin, xmax - 1, ymax - 1);
            arma::Col<uint16_t> sub_img_col = sub_img.as_col();
            arma::SpCol<uint16_t> sp_img(sub_img_col);
            arma::uvec locations = arma::find(sub_img>0);
            std::cout << locations << std::endl;
            imgs.col(k) = sp_img;
        }
        img_dataset.close();

        avg_img = arma::reshape(arma::Mat<uint16_t>(arma::mean(imgs, 1)), EIGER_X, EIGER_Y);
        arma::mat ttc = xpcs::generateTTC(imgs, sqnc_arma, numthreads, verbose);
        ttc.save(arma::hdf5_name(output_file, "/ttc/data"));
        time_axis.save(arma::hdf5_name(output_file, "/ttc/time_axis", arma::hdf5_opts::append));
        avg_img.save(arma::hdf5_name(output_file, "/waxs/data", arma::hdf5_opts::append));
//        avg_roi.save(arma::hdf5_name(output_file, "/waxs/roi_data", arma::hdf5_opts::append));
    } // end of try block
    // catch failure caused by the H5File operations
    catch (H5::FileIException &error) {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSet operations
    catch (H5::DataSetIException &error) {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSpace operations
    catch (H5::DataSpaceIException &error) {
        error.printErrorStack();
        return -1;
    }
    return 0;
}
