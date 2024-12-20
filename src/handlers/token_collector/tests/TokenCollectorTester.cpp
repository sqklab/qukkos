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
#include "test_utils.hpp"
#include "token_collector_util.hpp"
#include "xacc_service.hpp"
#include "clang/Sema/DeclSpec.h"
#include "gtest/gtest.h"
#include <xacc.hpp>
#include "xacc_config.hpp"
#include "qukkos_config.hpp"

TEST(TokenCollectorTester, checkSimple) {

  LexerHelper helper;

  auto [tokens, PP] =
      helper.Lex("H(q[0]);\nCX(q[0],q[1]);\nRy(q[3], theta);\nRx(q[0], "
                 "2.2);\nfor (int i = 0; i < "
                 "q.size(); i++) {\n  "
                 "Measure(q[i]);\n}\nunknown_func_let_compiler_find_it(q, 0, "
                 "q.size(), 0);\n");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }

  auto results = qukkos::run_token_collector(*PP, cached, {"q"});

  std::cout << results << "\n";
}

TEST(TokenCollectorTester, checkQPE) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(const auto nQubits = q.size();
  // Last qubit is the eigenstate of the unitary operator 
  // hence, prepare it in |1> state
  X(q[nQubits - 1]);

  // Apply Hadamard gates to the counting qubits:
  for (int qIdx = 0; qIdx < nQubits - 1; ++qIdx) {
    H(q[qIdx]);
  }

  // Apply Controlled-Oracle: in this example, Oracle is T gate;
  // i.e. Ctrl(T) = CPhase(pi/4)
  const auto bitPrecision = nQubits - 1;
  for (int32_t i = 0; i < bitPrecision; ++i) {
    const int nbCalls = 1 << i;
    for (int j = 0; j < nbCalls; ++j) {
      int ctlBit = i;
      // Controlled-Oracle
      Controlled::Apply(ctlBit, compositeOp, q);
    }
  }

  // Inverse QFT on the counting qubits:
  int startIdx = 0;
  int shouldSwap = 1;
  iqft(q, startIdx, bitPrecision, shouldSwap);

  // Measure counting qubits
  for (int qIdx = 0; qIdx < bitPrecision; ++qIdx) {
    Measure(q[qIdx]);
  })#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qukkos::run_token_collector(*PP, cached, {"q"});
  std::cout << results << "\n";
}

TEST(TokenCollectorTester, checkOpenQasm) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(using qukkos::openqasm;
  h r[0];
  cx r[0], r[1];
  creg c[2];
  measure r -> c;
  )#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qukkos::run_token_collector(*PP, cached, {"r"});
  std::cout << results << "\n";

  EXPECT_EQ(R"#(quantum::h(r[0]);
quantum::cnot(r[0], r[1]);
quantum::mz(r[0]);
quantum::mz(r[1]);
)#", results);
}


TEST(TokenCollectorTester, checkMixed) {
  LexerHelper helper;

  auto [tokens, PP] = helper.Lex(R"#(
  H(q[0]);
  CX(q[0], q[1]);

  using qukkos::openqasm;

  h r[0];
  cx r[0], r[1];

  using qukkos::xasm;
  
  for (int i = 0; i < q.size(); i++) {
    Measure(q[i]);
    Measure(r[i]);
  }
  )#");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qukkos::run_token_collector(*PP, cached, {"r"});
  std::cout << results << "\n";

  EXPECT_EQ(R"#(quantum::h(q[0]);
quantum::cnot(q[0], q[1]);
quantum::h(r[0]);
quantum::cnot(r[0], r[1]);
for ( int i = 0 ; i < q.size() ; i ++ ) { 
quantum::mz(q[i]);
quantum::mz(r[i]);
} 
)#", results);
}

TEST(TokenCollectorTester, checkPyXasm) {
  LexerHelper helper;

 auto [tokens, PP] = helper.Lex(R"(using qukkos::pyxasm;
    H(qb[0])
    CX(qb[0],qb[1])
    for i in range(qb.size()):
        X(qb[i])
        X(qb[i])
        Measure(qb[i])
)");

  clang::CachedTokens cached;
  for (auto &t : tokens) {
    cached.push_back(t);
  }
  auto results =
      qukkos::run_token_collector(*PP, cached, {"qb"});
  std::cout << results << "\n";
  EXPECT_EQ(R"#(quantum::h(qb[0]);
quantum::cnot(qb[0], qb[1]);
for (auto i : range(qb.size())) {
quantum::x(qb[i]);
quantum::x(qb[i]);
quantum::mz(qb[i]);
}
)#",
            results);
}
int main(int argc, char **argv) {
  std::string xacc_config_install_dir = std::string(XACC_INSTALL_DIR);
  std::string qukkos_root = std::string(QUKKOS_INSTALL_DIR);
  if (xacc_config_install_dir != qukkos_root) {
    xacc::addPluginSearchPath(std::string(QUKKOS_INSTALL_DIR) + "/plugins");
  }
  xacc::Initialize();
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  xacc::Finalize();
  return ret;
}
