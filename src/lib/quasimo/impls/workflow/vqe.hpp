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
#include "qukkos_qsim.hpp"

namespace qukkos {
namespace QuaSiMo {
// VQE-type workflow which involves an optimization loop, i.e. an Optimizer.
class VqeWorkflow : public QuantumSimulationWorkflow {
public:
  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;

  virtual const std::string name() const override { return "vqe"; }
  virtual const std::string description() const override { return ""; }

private:
  std::shared_ptr<Optimizer> optimizer;
  HeterogeneousMap config_params;
};
} // namespace QuaSiMo
} // namespace qukkos