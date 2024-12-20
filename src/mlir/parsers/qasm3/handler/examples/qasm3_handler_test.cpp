#include <stdio.h>

#include <cstring>
#define __qasm3__ [[clang::syntax(qasm3)]]

__qasm3__ int test(int i, qubit q, qubit r, qukkos::qreg& qq,
                   std::vector<double> xx) {

  // qreg pass by reference until we figure out ref counting on array!!!

  int mult = 10 * i;
  // now do some quantum
  print("test out vector :) ", xx[0]);
  h q;
  h qq[0];  // show using a qreg... this will result in all 00s
  cx q, r;
  bit c[2];
  c[0] = measure q;
  c[1] = measure r;
  reset q;
  reset r;
  return int[32](c);
}

// Run with...
// clang++ -std=c++17 -fplugin=/path/to/libqasm3-syntax-handler.so -I
// /path/to/qukkos/include/qukkos -I/path/to/xacc/include/xacc -c qasm3_test.cpp
// llc -filetype=obj test.bc (test comes from kernel name, so will need to do
// this for all kernels) clang++ -L /path/to/install/lib -lqir-qrt -lqukkos -lxacc
// -lqrt -lCppMicroServices test.o qasm3_test.o
// ./a.out

int main(int argc, char** argv) {
  // Can provide runtime parameters
  // via explicit initialize call.
  qukkos::initialize(argc, argv);
  // Otherwise it will be called automatically

  int x = 10, shots = 50;
  std::vector<double> xx{1.2};

  // Connect this to qalloc(...)
  auto qreg = qukkos::qalloc(2);

  // Can now extract the qubits individually
  auto q = qreg[0];
  auto r = qreg[1];

  // Run bell test...
  int ones = 0, zeros = 0;
  for (int i = 0; i < shots; i++) {
    // should be binary-as-int 00 = 0, or 11 = 3
    xx[0] = (double)i;
    if (test(32, q, r, qreg, xx)) {
      ones++;
    } else {
      zeros++;
    }
  }

  assert(zeros == shots);

  printf("Result: {'11':%d, '00':%d}\n", ones, zeros);

  // quantum memory will be freed
  // when it goes out of scope.
  return 0;
}