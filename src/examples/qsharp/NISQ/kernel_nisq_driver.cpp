#include <iostream> 
#include <vector>
#include "import_kernel_utils.hpp"

// Util pre-processor to wrap Q# operation in a QUKKOS QuantumKernel.
qukkos_import_qsharp_kernel(QUKKOS__TestKernel, double);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qukkos -qpu aer:ibmqx2 -shots 1024 kernel_nisq.qs kernel_nisq_driver.cpp
// Run with:
// $ ./a.out
int main() {
  auto q = qalloc(3);
  qukkos::set_verbose(true);
  QUKKOS__TestKernel(q, 1.0);
  q.print();

  // Integrate w/ QUKKOS's kernel utility...
  // e.g. kernel print-out...
  // std::cout << "HELLO:\n";
  // QUKKOS__TestKernel::print_kernel(std::cout, q, M_PI/4);

  return 0;
}