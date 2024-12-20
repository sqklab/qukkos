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
#include "trotter.hpp"

#include "xacc_service.hpp"

namespace qukkos {
namespace QuaSiMo {
Ansatz TrotterEvolution::create_ansatz(Operator *obs,
                                       const HeterogeneousMap &params) {
  Ansatz result;
  // This ansatz generator requires an observable.
  assert(obs != nullptr);
  double dt = 1.0;
  if (params.keyExists<double>("dt")) {
    dt = params.get<double>("dt");
  }

  // always default to true
  auto cau_opt = params.get_or_default("cau-opt", true);

  // Just use exp_i_theta for now
  // TODO: formalize a standard library kernel for this.
  auto expCirc = std::dynamic_pointer_cast<xacc::quantum::Circuit>(
      xacc::getService<xacc::Instruction>("exp_i_theta"));
  expCirc->expand({{"pauli", obs->toString()},
                   {"__internal_compute_action_uncompute_opt__", cau_opt}});
  result.circuit =
      std::make_shared<qukkos::CompositeInstruction>(expCirc->operator()({dt}));
  result.nb_qubits = expCirc->nRequiredBits();

  return result;
}
}  // namespace QuaSiMo
}  // namespace qukkos