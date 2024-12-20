#include <iostream> 
#include <vector>

// Include the external QSharp function.
// With EntryPoint() annotation, there are 3 functions generated:
// QUKKOS__TestBell__body(): raw Q# callable (marked internal)
// QUKKOS__TestBell(): EntryPoint type (with result printing), no return
// QUKKOS__TestBell__Interop(): InteropFriendly function (type casting to C-type function)
qukkos_include_qsharp(QUKKOS__TestBell__Interop, int64_t, int64_t)

// Compile with:
// Include both the qsharp source and this driver file 
// in the command line.
// -qs-build-exe to activate entry point generation.
// $ qukkos -qrt ftqc -qs-build-exe bell.qs bell_driver.cpp
// Run with:
// $ ./a.out
int main() {
  auto one_count = QUKKOS__TestBell__Interop(1024);
  std::cout << "One count: " << one_count << "\n";
  // In the range of Bell experiment: 50-50
  qukkos_expect(one_count > 400 && one_count < 700);
  return 0;
}