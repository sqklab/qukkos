#include <iostream>
#include <vector>

// For testing Q# IQFT circuit...
// Currently, Qubits are **not** allowed in EntryPoint => need to use a dummy entry point.
qukkos_include_qsharp(QUKKOS__Dummy, void);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// Use print-final-submission to see the instructions executed.
// $ qukkos -qrt ftqc -print-final-submission ...
// Run with:
// $ ./a.out
int main() {
  QUKKOS__Dummy();
  return 0;
}