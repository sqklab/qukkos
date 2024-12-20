#include "qukkos_qsim.hpp"

// Validate the ground-state energy of the Deuteron Hamiltonian operator by the
// iterative QPE procedure.

// Compile and run with:
/// $ qukkos IterativeQpeVqe.cpp
/// $ ./a.out

/// Ansatz to bring the state into an eigenvector state of the Hamiltonian.
/// This optimized ansatz was found by VQE.
__qpu__ void eigen_state_prep(qreg q) {
  X(q[0]);
  // Theta angle found by VQE.
  double opt_theta = 0.297113;
  auto exponent_op = X(0) * Y(1) - Y(0) * X(1);
  exp_i_theta(q, opt_theta, exponent_op);
}

int main(int argc, char **argv) {
  // Create Hamiltonian:
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.143 * Y(0) * Y(1) + 0.21829 * Z(0) -
           6.125 * Z(1);
  auto problemModel =
      QuaSiMo::ModelFactory::createModel(eigen_state_prep, H, 2, 0);

  // Instantiate an IQPE workflow
  // NOTE: can turn off exp_i_theta compute-action-uncompute with {"cau-opt", false}
  auto workflow =
      QuaSiMo::getWorkflow("iqpe", {{"time-steps", 8}, {"iterations", 8}});

  auto result = workflow->execute(problemModel);
  const double phaseValue = result.get<double>("phase");
  const double energy = result.get<double>("energy");
  auto n_insts = result.get<std::vector<int>>("n-kernel-instructions");

  std::cout << "Final phase = " << phaseValue << "\n";
  // Expect: ~ -1.7 (due to limited bit precision)
  std::cout << "Energy = " << energy << "\n";
  return 0;
}