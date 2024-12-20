// Compile with:
// qukkos gradients_optimization.cpp
// By default, kernel::autograd will use `central` gradient strategy.
// This can be changed by using the `-autograd` qukkos compiler flag:
// e.g.,
// qukkos -autograd forward gradients_optimization.cpp
// Options: forward, backward, central
// Note: users may need to adjust the optimizer step size to guarantee
// convergence.
__qpu__ void ansatz(qreg q, double theta) {
  X(q[0]);
  Ry(q[1], theta);
  CX(q[1], q[0]);
}

// Ansatz that takes a vector
__qpu__ void ansatz_vec(qreg q, std::vector<double> angles) {
  X(q[0]);
  Ry(q[1], angles[0]);
  CX(q[1], q[0]);
}

// Ansatz with an arbitrary signature
__qpu__ void ansatz_complex(qreg q, int idx, std::vector<double> angles) {
  X(q[0]);
  Ry(q[1], angles[idx]);
  CX(q[1], q[0]);
}

int main(int argc, char **argv) {
  // Allocate 2 qubits
  auto q = qalloc(2);

  // Programmer needs to set
  // the number of variational params
  auto n_variational_params = 1;

  // Create the Deuteron Hamiltonian
  auto H = 5.907 - 2.1433 * X(0) * X(1) - 2.1433 * Y(0) * Y(1) + .21829 * Z(0) -
           6.125 * Z(1);

  // Simple case 1: variational ansatz takes a single double
  {
    // Create the Optimizer (gradient-based)
    auto optimizer = createOptimizer("mlpack", {{"step-size", 1e-2}});
    ObjectiveFunction opt_function(
        [&](const std::vector<double> &x, std::vector<double> &dx) {
          auto q = qalloc(2);
          // Using kernel auto-gradient helper
          // Note: ansatz takes a single double argument => x[0]
          // Compile error if using the wrong signature.
          auto exp = ansatz::autograd(H, dx, q, x[0]);
          print("<E(", x[0], ") = ", exp);
          return exp;
        },
        n_variational_params);

    auto [energy, opt_params] = optimizer->optimize(opt_function);
    print(energy);
    qukkos_expect(std::abs(energy + 1.74886) < 0.1);
  }

  // Simple case 2: variational ansatz takes a vector<double>
  {
    // Create the Optimizer (gradient-based)
    auto optimizer = createOptimizer("mlpack", {{"step-size", 1e-2}});
    ObjectiveFunction opt_function(
        [&](const std::vector<double> &x, std::vector<double> &dx) {
          auto q = qalloc(2);
          // Using kernel auto-gradient helper
          // Note: ansatz_vec takes a vector of double; hence just forward the
          // whole vector.
          auto exp = ansatz_vec::autograd(H, dx, q, x);
          print("<E(", x[0], ") = ", exp);
          return exp;
        },
        n_variational_params);

    auto [energy, opt_params] = optimizer->optimize(opt_function);
    print(energy);
    qukkos_expect(std::abs(energy + 1.74886) < 0.1);
  }
  {
    // Create the Optimizer (gradient-based)
    auto optimizer = createOptimizer("mlpack", {{"step-size", 1e-2}});
    ObjectiveFunction opt_function(
        [&](const std::vector<double> &x, std::vector<double> &dx) {
          // Using kernel auto-gradient helper
          // Needs to provide args translator for this complex kernel.
          auto quantum_reg = qalloc(2);
          int index = 0;
          ArgsTranslator<qreg, int, std::vector<double>> args_translation(
              [&](const std::vector<double> x) {
                return std::tuple(quantum_reg, index, x);
              });
          auto exp = ansatz_complex::autograd(H, dx, x, args_translation);
          print("<E(", x[0], ") = ", exp);
          return exp;
        },
        n_variational_params);

    auto [energy, opt_params] = optimizer->optimize(opt_function);
    print(energy);
    qukkos_expect(std::abs(energy + 1.74886) < 0.1);
  }
}
