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
#include "clang/CodeGen/CodeGenAction.h"
#include "gtest/gtest.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/ToolOutputFile.h"
#include "qukkos_clang_wrapper.hpp"
#include "qukkos_mlir_api.hpp"
#include "qukkos_syntax_handler.hpp"

TEST(qasm3VisitorTester, checkGlobalConstInSubroutine) {
  const std::string kernel_test = R"#(OPENQASM 3;
include "qelib1.inc";
int i = 10;
kernel test_this(int) -> int;
int j = test_this(i);
print(j);
QUKKOS_EXPECT_TRUE(j == 20);
)#";

  auto mlir = qukkos::mlir_compile(kernel_test, "kernel_test",
                                 qukkos::OutputType::MLIR, true);
  std::cout << mlir << "\n";

  auto llvm = qukkos::mlir_compile(kernel_test, "kernel_test",
                                 qukkos::OutputType::LLVMIR, true);
  std::cout << llvm << "\n";

  // -------------------------------------------//
  // Create an external llvm module containing the
  // actual kernel function code...
  qukkos::__internal__developer__flags__::add_predefines = false;  // Don't let QUKKOS SH AddPredefines run
  auto act = qukkos::emit_llvm_ir(R"#(extern "C" {
int test_this(int i) { return i + 10; }
}
)#");
  auto module = act->takeModule();

  // Add the module to a vector to pass to the JIT execute function
  std::vector<std::unique_ptr<llvm::Module>> extra_code_to_link;
  extra_code_to_link.push_back(std::move(module));
  // -------------------------------------------//

  EXPECT_FALSE(qukkos::execute(kernel_test, "kernel_test", extra_code_to_link, 0));
}

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  auto ret = RUN_ALL_TESTS();
  return ret;
}
