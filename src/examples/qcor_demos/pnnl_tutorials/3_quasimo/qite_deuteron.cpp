#include "qukkos_qsim.hpp"
// qukkos qite_deuteron.cpp -qpu qsim
__qpu__ void state_prep(qreg q) { X(q[0]); }

int main(int argc, char **argv) {
  using namespace QuaSiMo;

  // Create the Hamiltonian
  auto observable = -2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) +
                    .21829 * Z(0) - 6.125 * Z(1) + 5.907;

  // We'll run with 5 steps and .1 step size
  const int nbSteps = 5;
  const double stepSize = 0.1;

  // Create the model (2 qubits, 0 variational params in above state_prep)
  // and QITE workflow
  auto problemModel = ModelFactory::createModel(state_prep, &observable, 2, 0);
  auto workflow =
      getWorkflow("qite", {{"steps", nbSteps}, {"step-size", stepSize}});

  // Execute
  auto result = workflow->execute(problemModel);

  // Get the final energy and iteration values
  const auto energy = result.get<double>("energy");
  const auto energyAtStep = result.get<std::vector<double>>("exp-vals");
  std::cout << "QITE energy: [ ";
  for (const auto &val : energyAtStep) {
    std::cout << val << " ";
  }
  std::cout << "]\n";
  std::cout << "Ground state energy: " << energy << "\n";
}