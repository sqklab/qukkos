#include <iostream> 
#include <vector>


qukkos_include_qsharp(QUKKOS__QuantumPhaseEstimation__Interop, int64_t)

int main() {
  auto result = QUKKOS__QuantumPhaseEstimation__Interop();
  std::cout << "Result decimal: " << result << "\n";
  qukkos_expect(result == 4);
  return 0;
}