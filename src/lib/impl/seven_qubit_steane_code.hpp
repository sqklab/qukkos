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

// Distance-3 Steane quantum error correction code with 7 qubits
std::vector<std::vector<qukkos::Operator>> seven_qubit_code_stabilizers() {
  static const std::vector<std::vector<qukkos::Operator>> STABILIZERS{
      // Steane code has two groups of syndromes to detect X and Z errors.
      // X syndromes
      {qukkos::X(0), qukkos::X(2), qukkos::X(4), qukkos::X(6)},
      {qukkos::X(1), qukkos::X(2), qukkos::X(5), qukkos::X(6)},
      {qukkos::X(3), qukkos::X(4), qukkos::X(5), qukkos::X(6)},
      // Z syndromes
      {qukkos::Z(0), qukkos::Z(2), qukkos::Z(4), qukkos::Z(6)},
      {qukkos::Z(1), qukkos::Z(2), qukkos::Z(5), qukkos::Z(6)},
      {qukkos::Z(3), qukkos::Z(4), qukkos::Z(5), qukkos::Z(6)}};
  return STABILIZERS;
}

__qpu__ void seven_qubit_code_encoder(qreg q, int dataQubitIdx,
                                      std::vector<int> scratchQubitIdx) {
  H(q[scratchQubitIdx[0]]);
  H(q[scratchQubitIdx[2]]);
  H(q[scratchQubitIdx[5]]);
  CX(q[dataQubitIdx], q[scratchQubitIdx[4]]);
  CX(q[scratchQubitIdx[5]], q[scratchQubitIdx[1]]);
  CX(q[scratchQubitIdx[5]], q[scratchQubitIdx[3]]);
  CX(q[scratchQubitIdx[1]], q[dataQubitIdx]);
  CX(q[scratchQubitIdx[2]], q[scratchQubitIdx[4]]);
  CX(q[scratchQubitIdx[0]], q[scratchQubitIdx[4]]);
  CX(q[scratchQubitIdx[4]], q[scratchQubitIdx[5]]);
  CX(q[scratchQubitIdx[2]], q[scratchQubitIdx[3]]);
  CX(q[scratchQubitIdx[0]], q[scratchQubitIdx[1]]);
}

__qpu__ void seven_qubit_code_recover(qreg q, std::vector<int> logicalReg,
                                      std::vector<int> syndromes) {
  auto xSyndromes = {syndromes[0], syndromes[1], syndromes[2]};
  auto zSyndromes = {syndromes[3], syndromes[4], syndromes[5]};
  auto xSyndromeIdx = syndrome_array_to_int(xSyndromes);
  auto zSyndromeIdx = syndrome_array_to_int(zSyndromes);
  if (xSyndromeIdx > 0) {
    Z(q[xSyndromeIdx - 1]);
  }
  if (zSyndromeIdx > 0) {
    X(q[zSyndromeIdx - 1]);
  }
}
