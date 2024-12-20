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

TEST(qasm3VisitorTester, checkDefaultTypes) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";

int i = 11;
QUKKOS_EXPECT_TRUE(i == 11);

int[32] ii = 2;
i += ii;
QUKKOS_EXPECT_TRUE(i == 13);

int64_t l = 1;
int[64] ll = 2;
QUKKOS_EXPECT_TRUE(l+ll == 3);

float f = 3.14;
float[32] ff = 1.0;

double d = 22.335;
print(d);

)#";
  auto mlir =
      qukkos::mlir_compile(src, "test", qukkos::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";

  EXPECT_FALSE(qukkos::execute(src, "test"));

}


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}