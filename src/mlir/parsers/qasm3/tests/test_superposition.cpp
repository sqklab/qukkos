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

TEST(qasm3VisitorTester, checkSuperposition) {
  const std::string uint_index = R"#(OPENQASM 3;
include "qelib1.inc";
qubit q;
bit c;

const shots = 1024;
int[32] ones = 0;
int[32] zeros = 0;

for i in [0:shots] {
  h q;
  c = measure q;
  if (c == 1) {
   ones += 1;
  } else {
   zeros += 1;
  }
  reset q;
}

print("N |1> measured = ", ones);
print("N |0> measured = ", zeros);

// give the randomness a bit of wriggle room
QUKKOS_EXPECT_TRUE(ones > 450);
QUKKOS_EXPECT_TRUE(zeros > 450);
)#";
  auto mlir = qukkos::mlir_compile(uint_index, "uint_index",
                                 qukkos::OutputType::MLIR, false);
  std::cout << mlir << "\n";
  EXPECT_FALSE(qukkos::execute(uint_index, "uint_index"));
}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}