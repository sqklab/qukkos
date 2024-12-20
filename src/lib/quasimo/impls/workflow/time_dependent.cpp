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
#include "time_dependent.hpp"
#include "qsim_utils.hpp"
#include "xacc_service.hpp"
#include "xacc.hpp"

namespace qukkos {
namespace QuaSiMo {
bool TimeDependentWorkflow::initialize(const HeterogeneousMap &params) {
  // Get workflow parameters (specific to TD workflow):
  t_0 = 0.0;
  dt = 0.1;
  if (params.keyExists<double>("dt")) {
    dt = params.get<double>("dt");
  }
  int nbSteps = 1;
  if (params.keyExists<int>("steps")) {
    nbSteps = params.get<int>("steps");
  }

  t_final = nbSteps * dt;
  config_params = params;
  return true;
}

QuantumSimulationResult
TimeDependentWorkflow::execute(const QuantumSimulationModel &model) {
  QuantumSimulationResult result;
  evaluator = getEvaluator(model.observable, config_params);
  auto ham_func = model.hamiltonian;
  // A TD workflow: stepping through Trotter steps,
  // compute expectations at each step.
  double currentTime = t_0;
  std::shared_ptr<CompositeInstruction> totalCirc;
  // Just support Trotter for now
  // TODO: support different methods:
  auto method = xacc::getService<AnsatzGenerator>("trotter");
  // List of all circuits to evaluate:
  std::vector<std::shared_ptr<xacc::CompositeInstruction>> allCircuits;
  for (;;) {
    // Evaluate the time-dependent Hamiltonian:
    auto ham_t = ham_func(currentTime);
    auto stepAnsatz = method->create_ansatz(&ham_t, {{"dt", dt}});
    // std::cout << "t = " << currentTime << "\n";
    // std::cout << stepAnsatz.circuit->toString() << "\n";
    // First step:
    if (!totalCirc) {
      // If there is a state-prep circuit (non-zero initial state)
      if (model.user_defined_ansatz) {
        totalCirc = model.user_defined_ansatz->evaluate_kernel({});
        totalCirc->addInstructions(stepAnsatz.circuit->getInstructions());
      } else {
        totalCirc = stepAnsatz.circuit;
      }
    } else {
      // Append Trotter steps
      totalCirc->addInstructions(stepAnsatz.circuit->getInstructions());
    }
    // std::cout << totalCirc->toString() << "\n";
    // Add the circuit for this time step to the list for later execution    
    allCircuits.emplace_back(xacc::ir::asComposite(totalCirc->as_xacc()->clone()));
    currentTime += dt;
    if (currentTime > t_final) {
      break;
    }
  }

  // Evaluate exp-val at all timesteps
  std::vector<std::shared_ptr<CompositeInstruction>> allCircuitsCasted;
  for (auto &circ : allCircuits) {
    allCircuitsCasted.emplace_back(
        std::make_shared<CompositeInstruction>(circ));
  }
  auto resultExpectationValues = evaluator->evaluate(allCircuitsCasted);
  result.insert("exp-vals", resultExpectationValues);
  return result;
}
} // namespace QuaSiMo
} // namespace qukkos