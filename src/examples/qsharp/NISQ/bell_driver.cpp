#include <iostream> 
#include <vector>
#include "qir_nisq_kernel_utils.hpp"

// Util pre-processor to wrap Q# operation 
// in a QUKKOS QuantumKernel.
// Compile:
// Note: need to use alpha package since this kernel will take a qubit array.
// qukkos -qdk-version 0.17.2106148041-alpha bell.qs bell_driver.cpp -shots 1024
// Note: the first qreg argument is implicit.
qukkos_import_qsharp_kernel(QUKKOS__Bell);

int main() {
  // Allocate 2 qubits
  auto q = qalloc(2);
  // Print kernel
  QUKKOS__Bell::print_kernel(q);
  
  // Execute:
  QUKKOS__Bell(q);
  q.print();
  return 0;
}