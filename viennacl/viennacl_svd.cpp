#include <Eigen/Core>
#include <Eigen/SVD>
#include <iostream>

#define VIENNACL_WITH_OPENCL 1
#define VIENNACL_WITH_EIGEN 1

#include "viennacl/linalg/svd.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/vector.hpp"

using namespace Eigen;
using std::cout;

int main() {
  // initializers
  MatrixXf C;
  viennacl::matrix<float> vcl_C = viennacl::zero_matrix<float>(1e3, 50);
  viennacl::matrix<float> vcl_U = viennacl::zero_matrix<float>(1e3, 1e3);
  viennacl::matrix<float> vcl_V = viennacl::zero_matrix<float>(50, 50);

  // random data
  C.setRandom(1e5, 50);

  // copy eigen to viennacl
  viennacl::copy(C, vcl_C);

  // eigen SVD
  // JacobiSVD<MatrixXf> svd(C, ComputeThinU | ComputeThinV);
  //
  // std::cout << "diag" << std::endl;
  // std::cout << svd.singularValues() << std::endl;
  //
  // std::cout << "U" << std::endl;
  // std::cout << svd.matrixU() << std::endl;
  //
  // std::cout << "V" << std::endl;
  // std::cout << svd.matrixV() << std::endl;

  // viennacl svd

  viennacl::linalg::svd(vcl_C, vcl_U, vcl_V);

  viennacl::vector_base<float> D(vcl_C.handle(),
                                 std::min(vcl_C.size1(), vcl_C.size2()), 0,
                                 vcl_C.internal_size2() + 1);

  // std::cout << "diag" << std::endl;
  // std::cout << D << std::endl;
  //
  // std::cout << "U" << std::endl;
  // std::cout << vcl_U << std::endl;
  //
  // std::cout << "V" << std::endl;
  // std::cout << vcl_V << std::endl;

  std::cout << "V" << std::endl;
  return 0;
}
