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
#pragma once
#include <cxxabi.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace llvm {
class Module;
}
namespace qukkos {
  class CompositeInstruction;
}
namespace xacc {
class HeterogeneousMap;
}  // namespace xacc

namespace qukkos {
class LLVMJIT;

class QJIT {
  template <typename... Args>
  using kernel_functor_t = void (*)(Args...);

 private:
  std::map<std::size_t, std::string> cached_kernel_codes;
  std::string demangle(const char *name) {
    int status = -1;
    std::unique_ptr<char, void (*)(void *)> res{
        abi::__cxa_demangle(name, NULL, NULL, &status), std::free};
    return (status == 0) ? res.get() : std::string(name);
  };

  std::string qjit_cache_path = "";

 protected:
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr;
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr_with_parent;
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr_hetmap;
  std::map<std::string, std::uint64_t> kernel_name_to_f_ptr_parent_hetmap;

  std::unique_ptr<LLVMJIT> jit;
  std::unique_ptr<llvm::Module> module;

 public:
  QJIT();
  ~QJIT();
  const std::pair<std::string, std::string> run_syntax_handler(
      const std::string &quantum_kernel_src,
      const bool add_het_map_kernel_ctor = false);
  void jit_compile(const std::string &quantum_kernel_src,
                   const bool add_het_map_kernel_ctor = false,
                   const std::vector<std::string> &kernel_dependency = {},
                   const std::string &extra_functions_src = "",
                   std::vector<std::string> extra_headers = {});

  void jit_compile(std::unique_ptr<llvm::Module> m,
                   std::vector<std::string> extra_shared_lib_paths = {});

  void write_cache();

  template <typename... Args>
  void invoke(const std::string &kernel_name, Args... args) {
    // Debug: print the Args... type
    // std::cout << "QJIT Invoke: " << __PRETTY_FUNCTION__ << "\n";
    auto f_ptr = kernel_name_to_f_ptr[kernel_name];
    void (*kernel_functor)(Args...) = (void (*)(Args...))f_ptr;
    kernel_functor(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void invoke_with_parent(const std::string &kernel_name,
                          std::shared_ptr<qukkos::CompositeInstruction> parent,
                          Args... args) {
    // Debug: print the Args... type
    // std::cout << "QJIT Invoke with Parent: " << __PRETTY_FUNCTION__ << "\n";
    auto f_ptr = kernel_name_to_f_ptr_with_parent[kernel_name];
    void (*kernel_functor)(std::shared_ptr<qukkos::CompositeInstruction>,
                           Args...) =
        (void (*)(std::shared_ptr<qukkos::CompositeInstruction>, Args...))f_ptr;
    kernel_functor(parent, std::forward<Args>(args)...);
  }

  // Invoke with type forwarding: Args &&
  template <typename... Args>
  void invoke_forwarding(const std::string &kernel_name, Args &&... args) {
    // std::cout << "QJIT Invoke: " << __PRETTY_FUNCTION__ << "\n";
    auto f_ptr = kernel_name_to_f_ptr[kernel_name];
    void (*kernel_functor)(Args...) = (void (*)(Args...))f_ptr;
    kernel_functor(std::forward<Args>(args)...);
  }

  // Invoke with type forwarding: Args &&
  template <typename... Args>
  void invoke_with_parent_forwarding(
      const std::string &kernel_name,
      std::shared_ptr<qukkos::CompositeInstruction> parent, Args &&... args) {
    // std::cout << "QJIT Invoke with Parent: " << __PRETTY_FUNCTION__ << "\n";
    auto f_ptr = kernel_name_to_f_ptr_with_parent[kernel_name];
    void (*kernel_functor)(std::shared_ptr<qukkos::CompositeInstruction>,
                           Args...) =
        (void (*)(std::shared_ptr<qukkos::CompositeInstruction>, Args...))f_ptr;
    kernel_functor(parent, std::forward<Args>(args)...);
  }

  int invoke_main(int argc, char **argv) {
    auto f_ptr = kernel_name_to_f_ptr["main"];
    int (*kernel_functor)(int, char **) = (int (*)(int, char **))f_ptr;
    return kernel_functor(argc, argv);
  }

  void invoke_with_hetmap(const std::string &kernel_name,
                          xacc::HeterogeneousMap &args);
  std::shared_ptr<qukkos::CompositeInstruction> extract_composite_with_hetmap(
      const std::string name, xacc::HeterogeneousMap &m);

  template <typename... Args>
  kernel_functor_t<Args...> get_kernel(const std::string &kernel_name) {
    auto f_ptr = kernel_name_to_f_ptr[kernel_name];
    void (*kernel_functor)(Args...) = (void (*)(Args...))f_ptr;
    return kernel_functor;
  }

  // The type of kernel functions:
  enum class KernelType { Regular, HetMapArg, HetMapArgWithParent };
  // Return kernel function pointer (as an integer)
  // Returns 0 if the kernel doesn't exist.
  std::uint64_t get_kernel_function_ptr(
      const std::string &kernelName,
      KernelType subType = KernelType::Regular) const;
};

}  // namespace qukkos