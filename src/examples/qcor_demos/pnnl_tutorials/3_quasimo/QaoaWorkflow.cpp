#include "qukkos_qsim.hpp"

// Demonstrate qsim's QAOA workflow 

// Compile and run with:
/// $ qukkos -qpu qsim QaoaWorkflow.cpp
/// $ ./a.out

int main(int argc, char **argv) {
  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) + 0.21829 * Z(0) -
           6.125 * Z(1);
  auto problemModel = QuaSiMo::ModelFactory::createModel(H);
  auto optimizer = createOptimizer("nlopt");
  // Instantiate a QAOA workflow with the nlopt optimizer
  // "steps" = the (p) param in QAOA algorithm.
  auto workflow = QuaSiMo::getWorkflow("qaoa", {{"optimizer", optimizer}, {"steps", 8}});

  // Result should contain the ground-state energy along with the optimal
  // parameters.
  auto result = workflow->execute(problemModel);

  const auto energy = result.get<double>("energy");
  std::cout << "Ground-state energy = " << energy << "\n";
  return 0;
}