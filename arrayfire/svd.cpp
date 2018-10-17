/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/SVD>

int main(int argc, char* argv[])
{
    try
    {
        // Select a device and display arrayfire info
        int device = argc > 1 ? std::atoi(argv[1]) : 0;
        af::setDevice(device);
        af::info();

        Eigen::MatrixXf C = Eigen::MatrixXf::Random(1e4, 50); // host array
        af::array in(1e4, 50, C.data());                      // copy host data to device

        af::array u, s_vec, vt;
        svd(u, s_vec, vt, in);
    }
    catch (af::exception& e)
    {
        std::cout << e.what() << '\n';

        return 1;
    }

    return 0;
}
