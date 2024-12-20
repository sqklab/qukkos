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

#include "clang/Parse/Parser.h"

using namespace clang;

namespace qukkos {
extern std::string qpu_name;
extern int shots;

// Add this for internal development, specifically JIT tests
// where I don't want AddPredefines to add qukkos.hpp. For example
// where I want to compile a simple c++ code with no dependencies, 
// I don't want to include qukkos.hpp bc it makes it much slower.
namespace __internal__developer__flags__ { extern bool add_predefines;}

class QUKKOSSyntaxHandler : public SyntaxHandler {
public:
  QUKKOSSyntaxHandler() : SyntaxHandler("qukkos") {}
  void GetReplacement(Preprocessor &PP, Declarator &D, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS) override;
  
  // For use with qukkos jit
  void GetReplacement(Preprocessor &PP, std::string &kernel_name,
                      std::vector<std::string> program_arg_types,
                      std::vector<std::string> program_parameters,
                      std::vector<std::string> bufferNames, CachedTokens &Toks,
                      llvm::raw_string_ostream &OS, bool add_het_map_ctor = false);

  void AddToPredefines(llvm::raw_string_ostream &OS) override;
};
} // namespace qukkos