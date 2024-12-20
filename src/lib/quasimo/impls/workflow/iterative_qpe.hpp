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
// Iterative QPE workflow to estimate the energy of a Hamiltonian operator.
// For the first pass, we implement this as a workflow.
// This can be integrated as a CostFuncEvaluator if needed.
class IterativeQpeWorkflow : public QuantumSimulationWorkflow {
public:
  // Translate/stretch the Hamiltonian operator for QPE.
  struct HamOpConverter {
    double translation;
    double stretch;
    void fromObservable(Operator *obs);
    std::shared_ptr<Operator> stretchObservable(Operator *obs) const;
    double computeEnergy(double phaseVal) const;
  };

  virtual bool initialize(const HeterogeneousMap &params) override;
  virtual QuantumSimulationResult
  execute(const QuantumSimulationModel &model) override;
  virtual const std::string name() const override { return "iqpe"; }
  virtual const std::string description() const override { return ""; }

  static std::shared_ptr<CompositeInstruction> constructQpeTrotterCircuit(
      std::shared_ptr<Operator> obs, double trotter_step, size_t nbQubits,
      double compensatedAncRot = 0, int steps = 1, int k = 1, double omega = 0, bool cau_opt = true);

private:
  std::shared_ptr<CompositeInstruction>
  constructQpeCircuit(std::shared_ptr<Operator> obs, int k, double omega,
                      bool measure = true) const;

private:
  // Number of time slices (>=1)
  int num_steps;
  // Number of iterations (>=1)
  int num_iters;
  HamOpConverter ham_converter;

  bool cau_opt = true;
};
} // namespace QuaSiMo
} // namespace qukkos