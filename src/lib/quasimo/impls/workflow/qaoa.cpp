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
#include "qaoa.hpp"
#include "AlgorithmGradientStrategy.hpp"
#include "qsim_utils.hpp"
#include "xacc.hpp"
#include "xacc_service.hpp"
#include "Optimizer.hpp"

namespace qukkos {
namespace QuaSiMo {
bool QaoaWorkflow::initialize(const HeterogeneousMap &params) {
  const std::string DEFAULT_OPTIMIZER = "nlopt";
  optimizer.reset();
  if (params.pointerLikeExists<Optimizer>("optimizer")) {
    optimizer =
        xacc::as_shared_ptr(params.getPointerLike<Optimizer>("optimizer"));
  } else {
    optimizer = createOptimizer(DEFAULT_OPTIMIZER);
  }
  config_params = params;
  return (optimizer != nullptr);
}

QuantumSimulationResult
QaoaWorkflow::execute(const QuantumSimulationModel &model) {
  int nbSteps = 1;
  if (config_params.keyExists<int>("steps")) {
    nbSteps = config_params.get<int>("steps");
  }

  std::string parameterScheme = "Standard";
  if (config_params.stringExists("parameter-scheme")) {
    parameterScheme = config_params.getString("parameter-scheme");
  }

  auto qaoa_kernel = std::dynamic_pointer_cast<xacc::CompositeInstruction>(
      xacc::getService<xacc::Instruction>("qaoa"));
  qaoa_kernel->expand({{"nbQubits", model.observable->nBits()},
                       {"nbSteps", nbSteps},
                       {"cost-ham", std::dynamic_pointer_cast<xacc::Observable>(
                                        model.observable->get_as_opaque())},
                       {"parameter-scheme", parameterScheme}});
  evaluator = getEvaluator(model.observable, config_params);
  size_t nParams = qaoa_kernel->nVariables();
  assert(nParams > 1);
  const std::vector<double> init_params =
      qukkos::random_vector(-1.0, 1.0, nParams);

  (*optimizer)->xacc_opt->appendOption("initial-parameters", init_params);
  std::shared_ptr<xacc::AlgorithmGradientStrategy> gradient_strategy;
  if ((*optimizer)->xacc_opt->isGradientBased()) {
    if (config_params.stringExists("gradient-strategy")) {
      gradient_strategy = xacc::getService<xacc::AlgorithmGradientStrategy>(
          config_params.getString("gradient-strategy"));
    } else {
      // Default is to use autodiff
      gradient_strategy =
          xacc::getService<xacc::AlgorithmGradientStrategy>("autodiff");
    }
    gradient_strategy->initialize(
        {{"observable", std::dynamic_pointer_cast<xacc::Observable>(
                            model.observable->get_as_opaque())}});
  }

  auto result = optimizer->optimize(
      [&](const std::vector<double> &x, std::vector<double> &dx) {
        auto kernel = qaoa_kernel->operator()(x);
        auto energy =
            evaluator->evaluate(std::make_shared<CompositeInstruction>(kernel));
        if (gradient_strategy) {
          if (gradient_strategy->isNumerical()) {
            gradient_strategy->setFunctionValue(
                energy - (model.observable->hasIdentitySubTerm()
                              ? std::real(model.observable->getIdentitySubTerm().coefficient())
                              : 0.0));
          }

          auto grad_kernels =
              gradient_strategy->getGradientExecutions(qaoa_kernel, x);

          if (!grad_kernels.empty()) {
            auto tmp_grad = qalloc(model.observable->nBits());
            // Important note: these gradient kernels (not using the qsim
            // evaluator) need to be processed by the pass manager (e.g. perform
            // placement).
            std::vector<std::shared_ptr<CompositeInstruction>>
                grad_kernels_casted;
            for (auto &f : grad_kernels) {
              grad_kernels_casted.emplace_back(
                  std::make_shared<CompositeInstruction>(f));
            }
            executePassManager(grad_kernels_casted);
            xacc::internal_compiler::execute(tmp_grad.results(), grad_kernels);
            auto tmp_grad_children = tmp_grad.results()->getChildren();
            gradient_strategy->compute(dx, tmp_grad_children);
          }
          // This is an autodiff gradient calculation:
          else {
            gradient_strategy->compute(dx, {});
          }
        }
        // std::cout << "E(";
        // for (const auto &val : x) {
        //   std::cout << val << ",";
        // }
        // std::cout << ") = " << energy << "\n";
        return energy;
      },
      nParams);
  return {{"energy", (double)result.first}, {"opt-params", result.second}};
}
} // namespace QuaSiMo
} // namespace qukkos