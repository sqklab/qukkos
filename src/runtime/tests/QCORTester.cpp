/*******************************************************************************
 * Copyright (c) 2018-, UT-Battelle, LLC.
 * All rights reserved. This program and the accompanying materials
 * are made available under the terms of the MIT License 
 * which accompanies this distribution. 
 *
 * Contributors:
 *   Alexander J. McCaskey - initial API and implementation
 *   Thien Nguyen - implementation
 *******************************************************************************/
#include "qukkos.hpp"

#include <gtest/gtest.h>

using namespace xacc;
using namespace qukkos;

class rucc : public qukkos::QuantumKernel<class rucc, qreg, double> {
  friend class qukkos::QuantumKernel<class rucc, qreg, double>;

protected:
  void operator()(qreg q, double x) {
    if (!parent_kernel) {
      parent_kernel = qukkos::__internal__::create_composite(kernel_name);
      q.setNameAndStore("qreg_edRlhpjLiH");
    }
    ::quantum::set_current_program(parent_kernel);
    ::quantum::x(q[0]);
    ::quantum::ry(q[1], x);
    ::quantum::cnot(q[1], q[0]);
  }

public:
  inline static const std::string kernel_name = "rucc";
  rucc(qreg q, double x) : QuantumKernel<rucc, qreg, double>(q, x) {}
  rucc(std::shared_ptr<qukkos::CompositeInstruction> _parent, qreg q, double x)
      : QuantumKernel<rucc, qreg, double>(_parent, q, x) {}
  virtual ~rucc() {
    if (disable_destructor) {
      return;
    }
    ::quantum::set_backend("qpp");
    auto [q, x] = args_tuple;
    operator()(q, x);
    if (is_callable) {
      ::quantum::submit(q.results());
    }
  }
};
void rucc(std::shared_ptr<qukkos::CompositeInstruction> parent, qreg q,
          double x) {
  class rucc k(parent, q, x);
}
void __internal_call_function_rucc(qreg q, double x) { class rucc k(q, x); }

class rucc_vec
    : public qukkos::QuantumKernel<class rucc_vec, qreg, std::vector<double>> {
  friend class qukkos::QuantumKernel<class rucc_vec, qreg, std::vector<double>>;

protected:
  void operator()(qreg q, std::vector<double> x) {
    if (!parent_kernel) {
      parent_kernel = qukkos::__internal__::create_composite(kernel_name);
      q.setNameAndStore("qreg_sOsDByIVpM");
    }
    ::quantum::set_current_program(parent_kernel);
    ::quantum::x(q[0]);
    ::quantum::ry(q[1], x[0]);
    ::quantum::cnot(q[1], q[0]);
  }

public:
  inline static const std::string kernel_name = "rucc_vec";
  rucc_vec(qreg q, std::vector<double> x)
      : QuantumKernel<rucc_vec, qreg, std::vector<double>>(q, x) {}
  rucc_vec(std::shared_ptr<qukkos::CompositeInstruction> _parent, qreg q,
           std::vector<double> x)
      : QuantumKernel<rucc_vec, qreg, std::vector<double>>(_parent, q, x) {}
  virtual ~rucc_vec() {
    if (disable_destructor) {
      return;
    }
    ::quantum::set_backend("qpp");
    auto [q, x] = args_tuple;
    operator()(q, x);
    if (is_callable) {
      ::quantum::submit(q.results());
    }
  }
};
void rucc_vec(std::shared_ptr<qukkos::CompositeInstruction> parent, qreg q,
              std::vector<double> x) {
  class rucc_vec k(parent, q, x);
}
void __internal_call_function_rucc_vec(qreg q, std::vector<double> x) {
  class rucc_vec k(q, x);
}

// TEST(QUKKOSTester, checkTaskInitiate) {

//   ::quantum::initialize("qpp", "empty");

//   auto buffer = qalloc(4);

//   auto optimizer = qukkos::createOptimizer("nlopt");
//   std::shared_ptr<Observable> observable = qukkos::createObservable(
//       std::string("5.907 - 2.1433 X0X1 - 2.1433 Y0Y1 + .21829 Z0 - 6.125 Z1"));

//   // Create the ObjectiveFunction, here we want to run VQE
//   // need to provide ansatz and the Observable
//   auto objective = qukkos::createObjectiveFunction(rucc, observable, buffer, 1);

//   auto handle = qukkos::taskInitiate(objective, optimizer);
//   auto results = qukkos::sync(handle);
//   EXPECT_NEAR(-1.748865, results.opt_val, 1e-4);

//   auto objective_vec = qukkos::createObjectiveFunction(rucc_vec, observable, buffer,  1);
//   auto handle2 = qukkos::taskInitiate(objective_vec, optimizer);
//   auto results5 = qukkos::sync(handle2);
//   EXPECT_NEAR(-1.748865, results5.opt_val, 1e-4);
// }

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
