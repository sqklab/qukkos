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
#include "qukkos_mlir_api.hpp"

#include "Quantum/QuantumDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser.h"
#include "openqasm_mlir_generator.hpp"
#include "openqasmv3_mlir_generator.hpp"
#include "pass_manager.hpp"
#include "qukkos_config.hpp"
#include "qukkos_jit.hpp"
#include "quantum_to_llvm.hpp"
#include "tools/ast_printer.hpp"
#include "tools/qukkos-mlir-helper.hpp"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/TargetSelect.h"

namespace qukkos {

const std::string mlir_compile(const std::string &src,
                               const std::string &kernel_name,
                               const OutputType &output_type,
                               bool add_entry_point, int opt_level, std::map<std::string,std::string> extra_args) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  auto mlir_gen_result =
      qukkos::util::mlir_gen(src, kernel_name, add_entry_point, extra_args);
  mlir::OwningModuleRef &module = *(mlir_gen_result.module_ref);
  mlir::MLIRContext &context = *(mlir_gen_result.mlir_context);
  std::vector<std::string> &unique_function_names =
      mlir_gen_result.unique_function_names;

  // std::cout << "MLIR + Quantum Dialect:\n";
  if (output_type == OutputType::MLIR) {
    std::string s;
    llvm::raw_string_ostream os(s);
    module->print(os);
    os.flush();
    return s;
  }

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  if (opt_level > 0) {
    qukkos::configureOptimizationPasses(pm);
  }
  pm.addPass(std::make_unique<qukkos::ModifierRegionRewritePass>());
  pm.addPass(std::make_unique<qukkos::QuantumToLLVMLoweringPass>(
      true, unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return "";
  }

  if (output_type == OutputType::LLVMMLIR) {
    std::string s;
    llvm::raw_string_ostream os(s);
    module->print(os);
    os.flush();
    return s;
  }

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(opt_level, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return "";
  }

  if (output_type == OutputType::LLVMIR) {
    std::string s;
    llvm::raw_string_ostream os(s);
    llvmModule->print(os, nullptr, false, true);
    os.flush();
    return s;
  }

  exit(1);
  return "";
}

int execute(const std::string &src, const std::string &kernel_name,
            int opt_level, std::map<std::string, std::string> extra_args) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  auto mlir_gen_result = qukkos::util::mlir_gen(src, kernel_name, true, extra_args);
  mlir::OwningModuleRef &module = *(mlir_gen_result.module_ref);
  mlir::MLIRContext &context = *(mlir_gen_result.mlir_context);
  std::vector<std::string> &unique_function_names =
      mlir_gen_result.unique_function_names;

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  qukkos::configureOptimizationPasses(pm);
  pm.addPass(std::make_unique<qukkos::ModifierRegionRewritePass>());
  pm.addPass(std::make_unique<qukkos::QuantumToLLVMLoweringPass>(
      true, unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return 1;
  }

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(opt_level, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return 1;
  }

  // Instantiate our JIT engine
  QJIT jit;

  // Compile the LLVM module, this is basically
  // just building up the LLVM JIT engine and
  // loading all seen function pointers
  jit.jit_compile(
      std::move(llvmModule),
      std::vector<std::string>{
          std::string(QUKKOS_INSTALL_DIR) + std::string("/lib/libqir-qrt") +
              std::string(QUKKOS_LIB_SUFFIX),
          std::string(LLVM_ROOT) + std::string("/lib/libLLVMAnalysis") +
              std::string(QUKKOS_LIB_SUFFIX),
          std::string(LLVM_ROOT) + std::string("/lib/libLLVMInstrumentation") +
              std::string(QUKKOS_LIB_SUFFIX),
          std::string(LLVM_ROOT) + std::string("/lib/libLLVMX86CodeGen") +
              std::string(QUKKOS_LIB_SUFFIX)});

  std::vector<std::string> argv;
  std::vector<char *> cstrs;
  argv.insert(argv.begin(), "appExec");
  for (auto &s : argv) {
    cstrs.push_back(&s.front());
  }

  return jit.invoke_main(cstrs.size(), cstrs.data());
}

int execute(const std::string &src, const std::string &kernel_name,
            std::vector<std::unique_ptr<llvm::Module>> &extra, int opt_level) {
  mlir::registerAsmPrinterCLOptions();
  mlir::registerMLIRContextCLOptions();

  auto mlir_gen_result = qukkos::util::mlir_gen(src, kernel_name, true);
  mlir::OwningModuleRef &module = *(mlir_gen_result.module_ref);
  mlir::MLIRContext &context = *(mlir_gen_result.mlir_context);
  std::vector<std::string> &unique_function_names =
      mlir_gen_result.unique_function_names;

  // Create the PassManager for lowering to LLVM MLIR and run it
  mlir::PassManager pm(&context);
  qukkos::configureOptimizationPasses(pm);
  pm.addPass(std::make_unique<qukkos::ModifierRegionRewritePass>());
  pm.addPass(std::make_unique<qukkos::QuantumToLLVMLoweringPass>(
      true, unique_function_names));
  auto module_op = (*module).getOperation();
  if (mlir::failed(pm.run(module_op))) {
    std::cout << "Pass Manager Failed\n";
    return 1;
  }

  // Now lower MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);

  // Optimize the LLVM IR
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  auto optPipeline = mlir::makeOptimizingTransformer(opt_level, 0, nullptr);
  if (auto err = optPipeline(llvmModule.get())) {
    llvm::errs() << "Failed to optimize LLVM IR " << err << "\n";
    return 1;
  }

  // Use the Linker to add extra code to the module
  llvm::Linker linker(*llvmModule);
  for (auto &m : extra) {
    linker.linkInModule(std::move(m));
  }

  // Instantiate our JIT engine
  QJIT jit;

  // Compile the LLVM module, this is basically
  // just building up the LLVM JIT engine and
  // loading all seen function pointers
  jit.jit_compile(std::move(llvmModule),
                  std::vector<std::string>{std::string(QUKKOS_INSTALL_DIR) +
                                           std::string("/lib/libqir-qrt") +
                                           std::string(QUKKOS_LIB_SUFFIX)});

  std::vector<std::string> argv;
  std::vector<char *> cstrs;
  argv.insert(argv.begin(), "appExec");
  for (auto &s : argv) {
    cstrs.push_back(&s.front());
  }

  return jit.invoke_main(cstrs.size(), cstrs.data());
}

} // namespace qukkos