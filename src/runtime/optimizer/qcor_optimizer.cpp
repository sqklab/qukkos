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
#include "qukkos_optimizer.hpp"

#include "Optimizer.hpp"
#include "objective_function.hpp"
#include "qukkos_pimpl_impl.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

namespace qukkos {

/// ------------- Optimizer Wrapper ---------------
Optimizer::Optimizer() = default;
Optimizer::Optimizer(const std::string &name)
    : m_internal(xacc::getOptimizer(name)) {}
Optimizer::Optimizer(const std::string &name, xacc::HeterogeneousMap &&options)
    : m_internal(xacc::getOptimizer(name, std::move(options))) {}
Optimizer::Optimizer(std::shared_ptr<xacc::Identifiable> generic)
    : m_internal(std::dynamic_pointer_cast<xacc::Optimizer>(generic)) {}
Optimizer::~Optimizer() = default;

// Define the internal implementation, wraps an XACC Optimizer
std::pair<double, std::vector<double>>
Optimizer::OptimizerImpl::optimize(xacc::OptFunction &opt) {
  return xacc_opt->optimize(opt);
}

std::string Optimizer::name() { return m_internal->xacc_opt->name(); }

std::pair<double, std::vector<double>> Optimizer::optimize(
    std::function<double(const std::vector<double> &)> opt, const int dim) {
  xacc::OptFunction opt_(opt, dim);
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    std::function<double(const std::vector<double> &, std::vector<double> &)>
        opt,
    const int dim) {
  xacc::OptFunction opt_(opt, dim);
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    std::shared_ptr<ObjectiveFunction> obj) {
  xacc::OptFunction opt_(
      [obj](const std::vector<double> &x, std::vector<double> &dx) {
        return (*obj)(x, dx);
      },
      obj->dimensions());
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    ObjectiveFunction *obj) {
  xacc::OptFunction opt_(
      [obj](const std::vector<double> &x, std::vector<double> &dx) {
        return (*obj)(x, dx);
      },
      obj->dimensions());
  return m_internal->optimize(opt_);
}

std::pair<double, std::vector<double>> Optimizer::optimize(
    ObjectiveFunction &obj) {
  xacc::OptFunction opt_([&](const std::vector<double> &x,
                             std::vector<double> &dx) { return obj(x, dx); },
                         obj.dimensions());
  return m_internal->optimize(opt_);
}
Optimizer::OptimizerImpl *Optimizer::operator->() {
  return m_internal.operator->();
}

std::shared_ptr<Optimizer> createOptimizer(const std::string &type,
                                           xacc::HeterogeneousMap &&options) {
  if (!xacc::isInitialized())
    xacc::internal_compiler::compiler_InitializeXACC();
  auto xacc_opt = xacc::getOptimizer(type, options);

  return std::make_shared<Optimizer>(xacc_opt);
}

}  // namespace qukkos