#include "qukkos.hpp"

int main(int argc, char** argv) {
  set_shots(1024);

  auto x_lambda = qpu_lambda([](qubit q) { 
    X(q); });
  
  auto bell = qpu_lambda([](qreg q) {
    H(q[0]);
    // Call the captured lambda
    x_lambda.ctrl(q[0], q[1]);
    Measure(q);
  }, x_lambda);
  
  auto q = qalloc(2);
  bell(q);
  q.print();
  qukkos_expect(q.counts().size() == 2);
  qukkos_expect(q.counts()["00"] > 400);
  qukkos_expect(q.counts()["11"] > 400);
  // Entangled...
  qukkos_expect(q.counts()["00"] + q.counts()["11"] == 1024);
}
