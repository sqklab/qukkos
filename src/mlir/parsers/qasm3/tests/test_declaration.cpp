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

TEST(qasm3VisitorTester, checkDeclaration) {
  const std::string src = R"#(OPENQASM 3;
include "qelib1.inc";
int[10] x = 5;
int[10] y;
print(x);
QUKKOS_EXPECT_TRUE(x == 5);
QUKKOS_EXPECT_TRUE(y == 0);

int[5] xx=2, yy=1;
QUKKOS_EXPECT_TRUE(xx == 2);
QUKKOS_EXPECT_TRUE(yy == 1);

qubit q1[6], q2;
bit b1[4]="0100", b2 = "1";
QUKKOS_EXPECT_TRUE(b1[0] == 0);
QUKKOS_EXPECT_TRUE(b1[1] == 1);
QUKKOS_EXPECT_TRUE(b1[2] == 0);
QUKKOS_EXPECT_TRUE(b1[3] == 0);

bit k, kk[22];
QUKKOS_EXPECT_TRUE(k == 0);
QUKKOS_EXPECT_TRUE(kk[13] == 0);
bool bb = False;
bool m=True, n=bool(xx);
QUKKOS_EXPECT_TRUE(m == 1);
QUKKOS_EXPECT_TRUE(bb == 0);
QUKKOS_EXPECT_TRUE(n == 0);
const c = 5.5e3, d=5;
const e = 2.2;
QUKKOS_EXPECT_TRUE(c == 5500.0);
QUKKOS_EXPECT_TRUE(d == 5);
QUKKOS_EXPECT_TRUE(e == 2.2);
x q2;
k = measure q2;
QUKKOS_EXPECT_TRUE(k == 1);
for i in [0:22] {
    QUKKOS_EXPECT_TRUE(kk[i] == 0);
}

float[64] f = 3.14;
float[64] test = 3.14 - f;
QUKKOS_EXPECT_TRUE(test < .001);
)#";
  auto mlir =
      qukkos::mlir_compile(src, "test", qukkos::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir << "\n";
      auto llvmi =
      qukkos::mlir_compile(src, "test", qukkos::OutputType::LLVMMLIR, true);
  std::cout << "LLVM:\n" << llvmi << "\n";
    auto llvm =
      qukkos::mlir_compile(src, "test", qukkos::OutputType::LLVMIR, true);
  std::cout << "LLVM:\n" << llvm << "\n";
  EXPECT_FALSE(qukkos::execute(src, "test"));

  const std::string src2 = R"#(OPENQASM 3;
include "qelib1.inc";
bit kk[22];
for i in [0:22] {
    print("should only see 0 bc next statement will fail: ", i);
    QUKKOS_EXPECT_TRUE(kk[i] == 1);
}
)#";
  auto mlir2 =
      qukkos::mlir_compile(src2, "test", qukkos::OutputType::MLIR, true);
  std::cout << "MLIR:\n" << mlir2 << "\n";
  EXPECT_TRUE(qukkos::execute(src2, "test"));
}

TEST(qasm3VisitorTester, checkCregAddressingWithMemref) {
  const std::string broadcast = R"#(OPENQASM 3;
include "qelib1.inc";
int[32] i = 2;
const j = 1;
bit b[4] = "0101";

QUKKOS_EXPECT_TRUE(b[i] == 0);
QUKKOS_EXPECT_TRUE(b[j] == 1);
)#";
  auto mlir = qukkos::mlir_compile(broadcast, "creg_memref_addressing",
                                 qukkos::OutputType::MLIR, false);

  EXPECT_FALSE(qukkos::execute(broadcast, "creg_memref_addressing"));
}

TEST(qasm3VisitorTester, checkGate) {
  const std::string gate_def = R"#(OPENQASM 3;
gate cphase(x) a, b
{
  U(0, 0, x / 2) a;
  CX a, b;
  U(0, 0, -x / 2) b;
  CX a, b;
  U(0, 0, x / 2) b;
}
)#";
  auto mlir = qukkos::mlir_compile(gate_def, "gate_def",
                                 qukkos::OutputType::MLIR, false);

  std::cout << "gate_def MLIR:\n" << mlir << "\n";

  qukkos::execute(gate_def, "gate_def");

  std::cout << "LLVM:\n"
            << qukkos::mlir_compile(gate_def, "gate_def",
                                  qukkos::OutputType::LLVMIR, false)
            << "\n";
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}