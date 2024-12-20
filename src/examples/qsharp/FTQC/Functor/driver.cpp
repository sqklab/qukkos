#include <iostream>
#include <vector>

qukkos_include_qsharp(QUKKOS__Testing__TestFunctors__Interop, int64_t);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qukkos -qrt ftqc bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  const auto error_code = QUKKOS__Testing__TestFunctors__Interop();
  std::cout << "Error code: " << error_code << "\n";
  qukkos_expect(error_code == 0);
  return 0;
}