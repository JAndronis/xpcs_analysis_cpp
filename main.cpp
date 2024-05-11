//
// Created by Iason Andronis on 2024-05-07.
//

#include "library.h"

#include <H5Cpp.h>

const H5std_string FILE_NAME("/Users/iasonandronis/Nextcloud/SDAQS_personal/xpcs_analysis/data/scan_000181_eiger500k.hdf5");
const H5std_string DATA("entry/instrument/Eiger/data");
const H5std_string DT("entry/instrument/Eiger/count_time");
const H5std_string SEQUENCE_NUMBER("entry/instrument/Eiger/sequence_number");

int main(int argc, const char * argv[]) {
    try {
        H5::H5File file(FILE_NAME, H5F_ACC_RDONLY);

        // sequence dataset
        H5::DataSet sqnc_dataset = file.openDataSet(SEQUENCE_NUMBER);
        H5::DataSpace sqnc_filespace = sqnc_dataset.getSpace();
        int sqnc_rank;
        hsize_t sqnc_dims[1];
        sqnc_rank = sqnc_filespace.getSimpleExtentDims(sqnc_dims);
        H5::DataSpace sqnc_memspace(sqnc_rank, sqnc_dims);
        const hsize_t max_imgs = sqnc_dims[0];
        constexpr hsize_t sqnc_offset[1] = {0};
        const hsize_t sqnc_count[1] = {max_imgs};
        std::vector<uint64_t> sqnc(max_imgs);
        sqnc_filespace.selectHyperslab(H5S_SELECT_SET, sqnc_count, sqnc_offset);
        sqnc_dataset.read(sqnc.data(), H5::PredType::NATIVE_INT64, sqnc_memspace, sqnc_filespace);
        arma::uvec sqnc_arma(sqnc);
        sqnc_dataset.close();

        // get dt
        H5::DataSet dt_dataset = file.openDataSet(DT);
        float dt;
        dt_dataset.read(reinterpret_cast<std::string &>(dt), H5::PredType::NATIVE_FLOAT, H5S_ALL, H5S_ALL);
        dt_dataset.close();
        auto time_axis = arma::regspace(dt, dt, dt*max_imgs);

        // image dataset
        H5::DataSet img_dataset = file.openDataSet(DATA);
        H5::DSetCreatPropList cparms = img_dataset.getCreatePlist();

        // chunk space
        hsize_t chunk_dims[3];
        const int rank_chunk = cparms.getChunk(3, chunk_dims);
        H5::DataSpace chunk_space(rank_chunk, chunk_dims);

        // file space
        H5::DataSpace img_filespace = img_dataset.getSpace();

        arma::SpMat<uint16_t> imgs(EIGER_X*EIGER_Y, max_imgs);

        arma::Mat<uint16_t> avg_img(EIGER_Y, EIGER_X, arma::fill::zeros);

        hsize_t offset[3] = {0, 0, 0};
        for (hsize_t k=0; k<max_imgs; k++) {
            if (k % 1000 == 0) {
                std::cout << "Loaded image " << k << std::endl;
            }

            offset[0] = k;
            img_filespace.selectHyperslab(H5S_SELECT_SET, chunk_dims, offset);

            // allocate buffer
            uint16_t data_3d[1][EIGER_Y][EIGER_X];
            img_dataset.read(data_3d, H5::PredType::NATIVE_UINT16, chunk_space, img_filespace);
            arma::Cube<uint16_t> data(&data_3d[0][0][0], EIGER_Y, EIGER_X, 1);
            arma::Mat<uint16_t> img(data.slice(0));
//            arma::Mat<uint16_t> mask(EIGER_Y, EIGER_X, arma::fill::zeros);
//            mask.submat(100, 600, 350, 800).fill(1);
//            img %= mask;
            arma::SpCol<uint16_t> sp_img(img.as_col());
            imgs.col(k) = sp_img;
            avg_img += img;
        }
        img_dataset.close();

        avg_img /= max_imgs;
        avg_img.save(arma::csv_name("avg_img.csv"));

        arma::mat ttc = xpcs::generateTTC(imgs, sqnc_arma);
        ttc.save(arma::hdf5_name("ttc.hdf5", "/ttc/data"));
        time_axis.save(arma::hdf5_name("ttc.hdf5", "/ttc/time_axis", arma::hdf5_opts::append));

    } // end of try block
    // catch failure caused by the H5File operations
    catch( H5::FileIException &error ) {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSet operations
    catch( H5::DataSetIException &error ) {
        error.printErrorStack();
        return -1;
    }
    // catch failure caused by the DataSpace operations
    catch( H5::DataSpaceIException &error ) {
        error.printErrorStack();
        return -1;
    }
    return 0;
}