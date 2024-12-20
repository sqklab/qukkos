#include <iostream> 
#include <vector>
#include "qir-types-utils.hpp"

// Include the external QSharp function.
qukkos_include_qsharp(QUKKOS__TestKernel__body, ::Array*);
qukkos_include_qsharp(QUKKOS__TestClean__body, void);

// Compile with:
// Include both the qsharp source and this driver file
// in the command line.
// $ qukkos -qrt ftqc kernel.qs driver.cpp
// Run with:
// $ ./a.out
int main() {
  // Kernel that clean-up all allocated objects.
  QUKKOS__TestClean__body();
  // No leak expected.
  assert(!qukkos::internal::AllocationTracker::get().checkLeak());
  // This kernel returns an Array,
  // (allocated in the kernel body)
  auto test = QUKKOS__TestKernel__body();
  const auto resultVec = qukkos::qir::fromArray<double>(test);
  // Should detect a leak.
  assert(qukkos::internal::AllocationTracker::get().checkLeak());

  for (const auto &el : resultVec) {
    std::cout << el << "\n";
  }

  // Release the ref-count of the returned array.
  // This should dealloc the Array
  test->release_ref();
  // No leak expected.
  assert(!qukkos::internal::AllocationTracker::get().checkLeak());

  return 0;
}