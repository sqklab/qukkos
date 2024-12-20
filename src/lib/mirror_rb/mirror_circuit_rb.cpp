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
#include "mirror_circuit_rb.hpp"
#include "AllGateVisitor.hpp"
#include "clifford_gate_utils.hpp"
#include "qukkos_ir.hpp"
#include "qukkos_pimpl_impl.hpp"
#include "xacc.hpp"
#include "xacc_plugin.hpp"
#include "xacc_service.hpp"
#include <cassert>
#include <random>
namespace {
std::vector<std::shared_ptr<xacc::Instruction>>
getLayer(std::shared_ptr<xacc::CompositeInstruction> circuit, int layerId) {
  std::vector<std::shared_ptr<xacc::Instruction>> result;
  assert(layerId < circuit->depth());
  auto graphView = circuit->toGraph();
  for (int i = 1; i < graphView->order() - 1; i++) {
    auto node = graphView->getVertexProperties(i);
    if (node.get<int>("layer") == layerId) {
      result.emplace_back(
          circuit->getInstruction(node.get<std::size_t>("id") - 1)->clone());
    }
  }
  assert(!result.empty());
  return result;
}
} // namespace
namespace xacc {
namespace quantum {
// Helper to convert a gate
class GateConverterVisitor : public AllGateVisitor {
public:
  GateConverterVisitor() {
    m_gateRegistry = xacc::getService<xacc::IRProvider>("quantum");
    m_program = m_gateRegistry->createComposite("temp_composite");
  }

  // Keep these 2 gates:
  void visit(CNOT &cnot) override { m_program->addInstruction(cnot.clone()); }
  void visit(U &u) override { m_program->addInstruction(u.clone()); }

  // Rotation gates:
  void visit(Ry &ry) override {
    const double theta = InstructionParameterToDouble(ry.getParameter(0));
    m_program->addInstruction(m_gateRegistry->createInstruction(
        "U", {ry.bits()[0]}, {theta, 0.0, 0.0}));
  }

  void visit(Rx &rx) override {
    const double theta = InstructionParameterToDouble(rx.getParameter(0));
    m_program->addInstruction(m_gateRegistry->createInstruction(
        "U", {rx.bits()[0]}, {theta, -1.0 * M_PI / 2.0, M_PI / 2.0}));
  }

  void visit(Rz &rz) override {
    const double theta = InstructionParameterToDouble(rz.getParameter(0));
    m_program->addInstruction(m_gateRegistry->createInstruction(
        "U", {rz.bits()[0]}, {0.0, theta, 0.0}));
  }

  void visit(X &x) override {
    Rx rx(x.bits()[0], M_PI);
    visit(rx);
  }
  void visit(Y &y) override {
    Ry ry(y.bits()[0], M_PI);
    visit(ry);
  }
  void visit(Z &z) override {
    Rz rz(z.bits()[0], M_PI);
    visit(rz);
  }
  void visit(S &s) override {
    Rz rz(s.bits()[0], M_PI / 2.0);
    visit(rz);
  }
  void visit(Sdg &sdg) override {
    Rz rz(sdg.bits()[0], -M_PI / 2.0);
    visit(rz);
  }
  void visit(T &t) override {
    Rz rz(t.bits()[0], M_PI / 4.0);
    visit(rz);
  }
  void visit(Tdg &tdg) override {
    Rz rz(tdg.bits()[0], -M_PI / 4.0);
    visit(rz);
  }

  void visit(Hadamard &h) override {
    m_program->addInstruction(m_gateRegistry->createInstruction(
        "U", {h.bits()[0]}, {M_PI / 2.0, 0.0, M_PI}));
  }

  void visit(Measure &measure) override {
    // Ignore measure
  }

  void visit(Identity &i) override {}

  void visit(CY &cy) override {
    // controlled-Y = Sdg(target) - CX - S(target)
    CNOT c1(cy.bits());
    Sdg sdg(cy.bits()[1]);
    S s(cy.bits()[1]);

    visit(sdg);
    visit(c1);
    visit(s);
  }

  void visit(CZ &cz) override {
    // CZ = H(target) - CX - H(target)
    CNOT c1(cz.bits());
    Hadamard h1(cz.bits()[1]);
    Hadamard h2(cz.bits()[1]);

    visit(h1);
    visit(c1);
    visit(h2);
  }

  void visit(CRZ &crz) override {
    const auto theta = InstructionParameterToDouble(crz.getParameter(0));
    // Decompose
    Rz rz1(crz.bits()[1], theta / 2);
    CNOT c1(crz.bits());
    Rz rz2(crz.bits()[1], -theta / 2);
    CNOT c2(crz.bits());

    // Revisit:
    visit(rz1);
    visit(c1);
    visit(rz2);
    visit(c2);
  }

  void visit(CH &ch) override {
    // controlled-H = Ry(pi/4, target) - CX - Ry(-pi/4, target)
    CNOT c1(ch.bits());
    Ry ry1(ch.bits()[1], M_PI_4);
    Ry ry2(ch.bits()[1], -M_PI_4);

    visit(ry1);
    visit(c1);
    visit(ry2);
  }

  std::shared_ptr<CompositeInstruction> getProgram() { return balanceLayer(); }

private:
  std::shared_ptr<CompositeInstruction> balanceLayer() {
    // Balance all layers
    // The mirroring protocol doesn't work well when there are layers that has
    // no gate on a qubit line, hence, just add an Identity there:
    auto program = m_gateRegistry->createComposite("temp_composite");
    const auto d = m_program->depth();
    const auto nbQubits = m_program->nPhysicalBits();
    for (int layer = 0; layer < d; ++layer) {
      std::set<int> qubits;
      auto current_layers = getLayer(m_program, layer);
      for (const auto &gate : current_layers) {
        program->addInstruction(gate);
        for (const auto &bit : gate->bits()) {
          qubits.emplace(bit);
        }
      }
      if (qubits.size() < nbQubits) {
        for (int i = 0; i < nbQubits; ++i) {
          if (!xacc::container::contains(qubits, i)) {
            program->addInstruction(
                m_gateRegistry->createInstruction("U", {(size_t)i}, {0.0, 0.0, 0.0}));
          }
        }
      }
    }
    return program;
  }
  std::shared_ptr<CompositeInstruction> m_program;
  std::shared_ptr<xacc::IRProvider> m_gateRegistry;
};
} // namespace quantum
} // namespace xacc

namespace qukkos {
std::pair<bool, xacc::HeterogeneousMap> MirrorCircuitValidator::validate(
    std::shared_ptr<xacc::Accelerator> qpu,
    std::shared_ptr<qukkos::CompositeInstruction> program,
    xacc::HeterogeneousMap options) {
  // Some default values: 4 Paulis for each qubit (double that to make sure we
  // cover more cases). Max is 1000 trials for many qubits.
  int n_trials = std::min(
      2 * static_cast<int>(std::pow(4, program->nPhysicalBits())), 1000);
  // 10% error allowed... (away from the expected bitstring)
  double eps = 0.1;
  int n_shots = 1024;

  if (options.keyExists<int>("trials")) {
    n_trials = options.get<int>("trials");
  }
  if (options.keyExists<double>("epsilon")) {
    eps = options.get<double>("epsilon");
  }

  qpu->updateConfiguration({{"shots", n_shots}});
  std::vector<double> trial_success_probs;
  auto provider = xacc::getIRProvider("quantum");
  for (int i = 0; i < n_trials; ++i) {
    auto mirror_data = qukkos::MirrorCircuitValidator::createMirrorCircuit(program);
    auto mirror_circuit = mirror_data.first;
    auto expected_result = mirror_data.second;
    const std::string expectedBitString = [&]() {
      std::string bitStr;
      if (qpu->getBitOrder() == xacc::Accelerator::BitOrder::MSB) {
        std::reverse(expected_result.begin(), expected_result.end());
      }
      for (const auto &bit : expected_result) {
        bitStr += std::to_string(bit);
      }
      return bitStr;
    }();

    for (size_t qId = 0; qId < mirror_circuit->nPhysicalBits(); ++qId) {
      mirror_circuit->addInstruction(
          provider->createInstruction("Measure", {qId}));
    }

    auto mc_buffer = xacc::qalloc(mirror_circuit->nPhysicalBits());
    qpu->execute(mc_buffer, mirror_circuit->as_xacc());
    {
      std::stringstream ss;
      ss << "Trial " << i << "\n";
      ss << "Circuit:\n" << mirror_circuit->toString() << "\n";
      ss << "Result:\n" << mc_buffer->toString() << "\n";
      ss << "Expected bitstring:" << expectedBitString << "\n";
      xacc::info(ss.str());
    }
    const auto bitStrProb =
        mc_buffer->computeMeasurementProbability(expectedBitString);
    trial_success_probs.emplace_back(bitStrProb);
  }
  xacc::HeterogeneousMap data;
  data.insert("trial-success-probabilities", trial_success_probs);
  const bool pass_fail_result =
      std::all_of(trial_success_probs.begin(), trial_success_probs.end(),
                  [&eps](double val) { return val > 1.0 - eps; });
  return std::make_pair(pass_fail_result, data);
}

std::pair<std::shared_ptr<CompositeInstruction>, std::vector<bool>>
MirrorCircuitValidator::createMirrorCircuit(
    std::shared_ptr<CompositeInstruction> in_circuit) {
  std::vector<std::shared_ptr<xacc::Instruction>> mirrorCircuit;
  auto gateProvider = xacc::getService<xacc::IRProvider>("quantum");

  // Gate conversion:
  xacc::quantum::GateConverterVisitor visitor;
  xacc::InstructionIterator it(in_circuit->as_xacc());
  while (it.hasNext()) {
    auto nextInst = it.next();
    if (nextInst->isEnabled() && !nextInst->isComposite()) {
      nextInst->accept(&visitor);
    }
  }

  auto program = visitor.getProgram();
  const int n = program->nPhysicalBits();
  // Tracking the Pauli layer as it is commuted through
  std::vector<qukkos::utils::PauliLabel> net_paulis(n,
                                                  qukkos::utils::PauliLabel::I);

  // Sympletic group
  const auto srep_dict = qukkos::utils::computeGateSymplecticRepresentations();

  const auto pauliListToLayer =
      [](const std::vector<qukkos::utils::PauliLabel> &in_paulis) {
        qukkos::utils::CliffordGateLayer_t result;
        for (int i = 0; i < in_paulis.size(); ++i) {
          const auto pauli = in_paulis[i];
          switch (pauli) {
          case qukkos::utils::PauliLabel::I:
            result.emplace_back(std::make_pair("I", std::vector<int>{i}));
            break;
          case qukkos::utils::PauliLabel::X:
            result.emplace_back(std::make_pair("X", std::vector<int>{i}));
            break;
          case qukkos::utils::PauliLabel::Y:
            result.emplace_back(std::make_pair("Y", std::vector<int>{i}));
            break;
          case qukkos::utils::PauliLabel::Z:
            result.emplace_back(std::make_pair("Z", std::vector<int>{i}));
            break;
          default:
            __builtin_unreachable();
          }
        }
        return result;
      };

  // program->as_xacc()->toGraph()->write(std::cout);
  const auto decomposeU3Angle = [](xacc::InstPtr u3_gate) {
    const double theta = InstructionParameterToDouble(u3_gate->getParameter(0));
    const double phi = InstructionParameterToDouble(u3_gate->getParameter(1));
    const double lam = InstructionParameterToDouble(u3_gate->getParameter(2));
    // Convert to 3 rz angles:
    const double theta1 = lam;
    const double theta2 = theta + M_PI;
    const double theta3 = phi + 3.0 * M_PI;
    return std::make_tuple(theta1, theta2, theta3);
  };

  const auto createU3GateFromAngle = [](size_t qubit, double theta1,
                                        double theta2, double theta3) {
    auto gateProvider = xacc::getService<xacc::IRProvider>("quantum");
    return gateProvider->createInstruction(
        "U", {qubit}, {theta2 - M_PI, theta3 - 3.0 * M_PI, theta1});
  };

  const auto d = program->depth();
  for (int layer = d - 1; layer >= 0; --layer) {
    auto current_layers = getLayer(program, layer);
    for (const auto &gate : current_layers) {
      if (gate->bits().size() == 1) {
        assert(gate->name() == "U");
        const auto u3_angles = decomposeU3Angle(gate);
        const auto [theta1_inv, theta2_inv, theta3_inv] =
            qukkos::utils::invU3Gate(u3_angles);
        const size_t qubit = gate->bits()[0];
        program->addInstruction(gateProvider->createInstruction(
            "U", {qubit},
            {theta2_inv - M_PI, theta1_inv - 3.0 * M_PI, theta3_inv}));
      } else {
        assert(gate->name() == "CNOT");
        program->addInstruction(gate->clone());
      }
    }
  }
  const int newDepth = program->depth();
  for (int layer = 0; layer < newDepth; ++layer) {
    auto current_layers = getLayer(program, layer);
    // New random Pauli layer
    const std::vector<qukkos::utils::PauliLabel> new_paulis = [](int nQubits) {
      static std::random_device rd;
      static std::mt19937 gen(rd());
      static std::uniform_int_distribution<size_t> dis(
          0, qukkos::utils::ALL_PAULI_OPS.size() - 1);
      std::vector<qukkos::utils::PauliLabel> random_paulis;
      for (int i = 0; i < nQubits; ++i) {
        random_paulis.emplace_back(qukkos::utils::ALL_PAULI_OPS[dis(gen)]);
      }
      // {
      //   std::stringstream ss;
      //   ss << "Random Pauli: ";
      //   for (const auto &p : random_paulis) {
      //     ss << p << " ";
      //   }
      //   xacc::info(ss.str());
      // }

      return random_paulis;
    }(n);

    const auto gateToLayerInfo = [](xacc::InstPtr gate, int nbQubits) {
      qukkos::utils::CliffordGateLayer_t result;
      std::vector<int> operands;
      for (const auto &bit : gate->bits()) {
        operands.emplace_back(bit);
      }

      for (int i = 0; i < nbQubits; ++i) {
        if (!xacc::container::contains(operands, i)) {
          result.emplace_back(std::make_pair("I", std::vector<int>{i}));
        }
      }

      result.emplace_back(std::make_pair(gate->name(), operands));
      return result;
    };

    const auto current_net_paulis_as_layer = pauliListToLayer(net_paulis);
    for (const auto &gate : current_layers) {
      if (gate->bits().size() == 1) {
        assert(gate->name() == "U");
        const auto new_paulis_as_layer = pauliListToLayer(new_paulis);
        const auto new_net_paulis_reps =
            qukkos::utils::computeCircuitSymplecticRepresentations(
                {new_paulis_as_layer, current_net_paulis_as_layer}, n,
                srep_dict);

        // Update the tracking net
        net_paulis = qukkos::utils::find_pauli_labels(new_net_paulis_reps.second);
        // {
        //   std::stringstream ss;
        //   ss << "Net Pauli: ";
        //   for (const auto &p : net_paulis) {
        //     ss << p << " ";
        //   }
        //   xacc::info(ss.str());
        // }

        const size_t qubit = gate->bits()[0];
        const auto [theta1, theta2, theta3] = decomposeU3Angle(gate);
        // Compute the pseudo_inverse gate:
        const auto [theta1_new, theta2_new, theta3_new] =
            qukkos::utils::computeRotationInPauliFrame(
                std::make_tuple(theta1, theta2, theta3), new_paulis[qubit],
                net_paulis[qubit]);
        mirrorCircuit.emplace_back(
            createU3GateFromAngle(qubit, theta1_new, theta2_new, theta3_new));
      } else {
        mirrorCircuit.emplace_back(gate->clone());
        // we need to account for how the net pauli changes when it gets passed
        // through the clifford layers
        const auto new_net_paulis_reps =
            qukkos::utils::computeCircuitSymplecticRepresentations(
                {gateToLayerInfo(gate, n), current_net_paulis_as_layer,
                 gateToLayerInfo(gate, n)},
                n, srep_dict);

        // Update the tracking net
        net_paulis = qukkos::utils::find_pauli_labels(new_net_paulis_reps.second);
        // {
        //   std::stringstream ss;
        //   ss << "Net Pauli: ";
        //   for (const auto &p : net_paulis) {
        //     ss << p << " ";
        //   }
        //   xacc::info(ss.str());
        // }
      }
    }
  }

  const auto [telp_s, telp_p] =
      qukkos::utils::computeLayerSymplecticRepresentations(
          pauliListToLayer(net_paulis), n, srep_dict);
  std::vector<bool> target_bitString;
  for (int i = n; i < telp_p.size(); ++i) {
    target_bitString.emplace_back(telp_p[i] == 2);
  }

  auto mirror_comp = gateProvider->createComposite(program->name() + "_MIRROR");
  mirror_comp->addInstructions(mirrorCircuit);
  return std::make_pair(std::make_shared<CompositeInstruction>(mirror_comp),
                        target_bitString);
}
} // namespace qukkos
REGISTER_PLUGIN(qukkos::MirrorCircuitValidator, qukkos::BackendValidator)
