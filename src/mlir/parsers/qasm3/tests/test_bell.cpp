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
#include "gtest/gtest.h"
#include "qukkos_mlir_api.hpp"

TEST(qasm3VisitorTester, checkPow) {
  const std::string check_pow = R"#(OPENQASM 3;
include "qelib1.inc";

qubit z[2];
int count = 0;
for i in [0:100] {
  h z[0];
  ctrl @ x z[0], z[1];
  bit g[2];
  measure z -> g;

  if (g[0] == 0 && g[1] == 0) {
    count += 1;
  }
  reset z;
}
print(count);
QUKKOS_EXPECT_TRUE(count > 30);
)#";
  auto mlir = qukkos::mlir_compile(check_pow, "check_pow",
                                 qukkos::OutputType::MLIR, false);
  std::cout << mlir << "\n";

 
  EXPECT_FALSE(qukkos::execute(check_pow, "check_pow"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}