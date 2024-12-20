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
#ifndef QUKKOS_HANDLERS_STAQTOKENCOLLECTOR_HPP_
#define QUKKOS_HANDLERS_STAQTOKENCOLLECTOR_HPP_

#include "token_collector.hpp"

namespace qukkos {
class StaqTokenCollector : public TokenCollector {
public:
  void collect(clang::Preprocessor &PP, clang::CachedTokens &Toks,
               std::vector<std::string> bufferNames,
               std::stringstream &ss, const std::string &kernel_name) override;
  const std::string name() const override { return "staq"; }
  const std::string description() const override { return ""; }
};

} // namespace qukkos

#endif