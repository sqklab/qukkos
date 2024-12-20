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

TEST(qasm3VisitorTester, checkLoops) {
  const std::string for_stmt = R"#(OPENQASM 3;
include "qelib1.inc";

int[32] loop_count = 0;
for i in {11,22,33} {
    print(i);
    loop_count += 1;
}
QUKKOS_EXPECT_TRUE(loop_count == 3);

loop_count = 0;
for i in [0:10] {
    print(i);
    loop_count += 1;
    
}
QUKKOS_EXPECT_TRUE(loop_count == 10);
loop_count = 0;

for j in [0:2:4] {
    print("steps:", j);
    loop_count += 1;
}

QUKKOS_EXPECT_TRUE(loop_count == 2);
loop_count = 0;

for j in [0:4] {
    print("j in 0:4", j);
    loop_count += 1;
}

QUKKOS_EXPECT_TRUE(loop_count == 4);
loop_count = 0;

for i in [0:4] {
 for j in {1,2,3} {
     print(i,j);
     loop_count += 1;
 }
}
QUKKOS_EXPECT_TRUE(loop_count == 12);

loop_count = 0;
for i in [0:4] {
 for j in [0:3] {
     print(i,j);
     loop_count += 1;
 }
}
QUKKOS_EXPECT_TRUE(loop_count == 12);

)#";
  auto mlir = qukkos::mlir_compile(for_stmt, "for_stmt",
                                 qukkos::OutputType::MLIR, false);
  std::cout << "for_stmt MLIR:\n" << mlir << "\n";
  EXPECT_FALSE(qukkos::execute(for_stmt, "for_stmt"));

  const std::string while_stmt = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 0;
while (i < 10) {
  print(i);
  i += 1;
}
QUKKOS_EXPECT_TRUE(i == 10);
)#";
  auto mlir2 = qukkos::mlir_compile(while_stmt, "while_stmt",
                                 qukkos::OutputType::MLIR, false);
  std::cout << mlir2 << "\n";
  // We're using SCF while loop:
  EXPECT_TRUE(mlir2.find("scf.while") != std::string::npos);
  EXPECT_FALSE(qukkos::execute(while_stmt, "while_stmt"));

    const std::string decrement = R"#(OPENQASM 3;
include "qelib1.inc";
for j in [10:-1:0] {
  print(j);
}
)#";
  auto mlir3 = qukkos::mlir_compile(decrement, "decrement",
                                 qukkos::OutputType::MLIR, false);
  std::cout << mlir3 << "\n";
  EXPECT_FALSE(qukkos::execute(decrement, "decrement"));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}