#include <iostream>
#include <vector>
#include "qir-types-utils.hpp"

// Include the external QSharp function.
qukkos_include_qsharp(QUKKOS__Ansatz__body, double, ::Array *, int64_t);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qukkos -qrt ftqc vqe_ansatz.qs vqe_driver.cpp
// Run with:
// $ ./a.out
int main() {
  const std::vector<double> angles{1.0, 2.0};
  const double exp_val_xx = QUKKOS__Ansatz__body(qir::toArray(angles), 1024);
  std::cout << "<X0X1> = " << exp_val_xx << "\n";

  return 0;
}