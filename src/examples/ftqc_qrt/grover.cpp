// Create a general grover search algorithm.
// Let's create that marks 2 states
// Show figures Init - [Oracle - Amplification for i in iters] - Measure
// https://www.nature.com/articles/s41467-017-01904-7

// Show off kernel composition, common patterns,
// functional programming (kernels taking other kernels)

using GroverPhaseOracle = KernelSignature<qreg>;

__qpu__ void amplification(qreg q) {
  // H q X q ctrl-ctrl-...-ctrl-Z H q Xq
  // compute - action - uncompute

  compute {
    H(q);
    X(q);
  }
  action {
    auto ctrl_bits = q.head(q.size() - 1);
    auto last_qubit = q.tail();
    Z::ctrl(ctrl_bits, last_qubit);
  }
}

__qpu__ void run_grover(qreg q, GroverPhaseOracle oracle, const int iterations,
                        int shots, std::vector<int> &results) {
  for (int k = 0; k < shots; k++) {
    H(q);

    for (int i = 0; i < iterations; i++) {
      oracle(q);
      amplification(q);
    }
    int bit_string = 0;
    for (int qid = 0; qid < q.size(); qid++) {
      if (Measure(q[qid])) {
        bit_string = bit_string + (1 << qid);
        X(q[qid]);
      }
    }
    // print("Result:", bit_string);
    results.emplace_back(bit_string);
  }
}

__qpu__ void oracle(qreg q) {
  // Mark 101 and 011
  CZ(q[0], q[2]);
  CZ(q[1], q[2]);
}

int main() {
  const int N = 3;

  // Allocate some qubits
  auto q = qalloc(N);
  // Call grover given the oracle and n iterations
  std::vector<int> results;
  run_grover(q, oracle, 1, 1024, results);
  qukkos_expect(results.size() == 1024);
  for (const auto &val : results) {
    // Only 2 possible values: 5 and 6:
    qukkos_expect(val == 5 || val == 6);
  }
}