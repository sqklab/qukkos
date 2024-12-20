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

TEST(qasm3VisitorTester, checkGlobalConstInSubroutine) {
  const std::string global_const = R"#(OPENQASM 3;
include "qelib1.inc";
const shots = 1024;
print(shots);
int[32] i = shots + 1;
print(i);

const t =  22;
print(t);

const n = t / 2 ;

def test(float[32]:tt)-> int[64] {
    int[64] s = 10;
    print("s = ", s);
    print(tt);
    s = shots;
    for i in [0:n] {
      print(i);
    }
    return s;
}

// qubit q;
float[32] ttt;
int[64] r = test(ttt) ;
print(r);
print(ttt);
QUKKOS_EXPECT_TRUE(r == 1024);
)#";

  int opt_level = 0;

  auto mlir = qukkos::mlir_compile(global_const, "global_const",
                                 qukkos::OutputType::MLIR, true);
  std::cout << mlir << "\n";

  auto llvm = qukkos::mlir_compile(global_const, "global_const",
                                 qukkos::OutputType::LLVMIR, true);
  std::cout << llvm << "\n";

  // Hvae to set opt level 0 since llvm.lifetime.start.* can't be found by JIT
  EXPECT_FALSE(qukkos::execute(global_const, "global_const", 0));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
