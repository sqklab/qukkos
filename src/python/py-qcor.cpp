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
#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "base/qukkos_qsim.hpp"
#include "py_costFunctionEvaluator.hpp"
#include "py_qsimWorkflow.hpp"
#include "qukkos_jit.hpp"
#ifdef QUKKOS_BUILD_MLIR_PYTHON_API
#include "qukkos_mlir_api.hpp"
#endif

#include "objective_function.hpp"
#include "qukkos_ir.hpp"
#include "qukkos_observable.hpp"
#include "qukkos_optimizer.hpp"
#include "qrt.hpp"
#include "xacc.hpp"
#include "xacc_internal_compiler.hpp"
#include "xacc_service.hpp"

namespace py = pybind11;
using namespace xacc;

namespace pybind11 {
namespace detail {
template <typename... Ts>
struct type_caster<Variant<Ts...>> : variant_caster<Variant<Ts...>> {};

template <>
struct visit_helper<Variant> {
  template <typename... Args>
  static auto call(Args &&...args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};

template <typename... Ts>
struct type_caster<mpark::variant<Ts...>>
    : variant_caster<mpark::variant<Ts...>> {};

template <>
struct visit_helper<mpark::variant> {
  template <typename... Args>
  static auto call(Args &&...args) -> decltype(mpark::visit(args...)) {
    return mpark::visit(args...);
  }
};
}  // namespace detail
}  // namespace pybind11

namespace {

// We only allow certain argument types for quantum kernel functors in python
// Here we enumerate them as a Variant
using AllowedKernelArgTypes =
    xacc::Variant<bool, int, double, std::string, xacc::internal_compiler::qreg,
                  xacc::internal_compiler::qubit, std::vector<double>,
                  std::vector<int>, qukkos::Operator, qukkos::PairList<int>,
                  std::vector<qukkos::Operator>, std::vector<std::string>>;

// We will take as input a mapping of arg variable names to the argument itself.
using KernelArgDict = std::map<std::string, AllowedKernelArgTypes>;

// Utility for mapping KernelArgDict to a HeterogeneousMap
class KernelArgDictToHeterogeneousMap {
 protected:
  xacc::HeterogeneousMap &m;
  const std::string &key;

 public:
  KernelArgDictToHeterogeneousMap(xacc::HeterogeneousMap &map,
                                  const std::string &k)
      : m(map), key(k) {}
  template <typename T>
  void operator()(const T &t) {
    m.insert(key, t);
  }
};

// Add type name to this list to support receiving from Python.
using PyHeterogeneousMapTypes =
    xacc::Variant<bool, int, double, std::string, std::vector<int>,
                  std::vector<std::pair<int, int>>,
                  std::shared_ptr<qukkos::IRTransformation>,
                  std::shared_ptr<qukkos::Optimizer>, std::vector<double>,
                  std::vector<std::vector<double>>>;
using PyHeterogeneousMap = std::map<std::string, PyHeterogeneousMapTypes>;

// Helper to convert a Python *dict* (as a map of variants) into a native
// HetMap.
xacc::HeterogeneousMap heterogeneousMapConvert(
    const PyHeterogeneousMap &in_pyMap) {
  xacc::HeterogeneousMap result;
  for (auto &item : in_pyMap) {
    auto visitor = [&](const auto &value) { result.insert(item.first, value); };
    mpark::visit(visitor, item.second);
  }

  return result;
}

qukkos::Operator convertToQUKKOSOperator(py::object op, bool keep_fermion = false) {
  if (py::hasattr(op, "terms")) {
    // this is from openfermion
    if (py::hasattr(op, "is_two_body_number_conserving")) {
      // This is a fermion Operator
      auto terms = op.attr("terms");
      // terms is a list of tuples
      std::stringstream ss;
      int i = 0;
      for (auto term : terms) {
        auto term_tuple = term.cast<py::tuple>();
        if (!term_tuple.empty()) {
          ss << terms[term].cast<std::complex<double>>() << " ";
          for (auto element : term_tuple) {
            auto element_pair = element.cast<std::pair<int, int>>();
            ss << element_pair.first << (element_pair.second ? "^" : "") << " ";
          }
        } else {
          // this was identity
          try {
            auto coeff = terms[term].cast<double>();
            ss << coeff;
          } catch (std::exception &e) {
            try {
              auto coeff = terms[term].cast<std::complex<double>>();
              ss << coeff;
            } catch (std::exception &e) {
              qukkos::error(
                  "Could not cast identity coefficient to double or complex.");
            }
          }
        }
        i++;
        if (i != py::len(terms)) {
          ss << " + ";
        }
      }
      auto obs_tmp = qukkos::createOperator("fermion", ss.str());
      if (keep_fermion) {
        return obs_tmp;
      } else {
        return qukkos::operatorTransform("jw", obs_tmp);
      }

    } else {
      if (keep_fermion) {
        xacc::error(
            "Error - you asked for a qukkos::FermionOperator, but this is an "
            "OpenFermion QubitOperator.");
      }
      // this is a qubit  operator
      auto terms = op.attr("terms");
      // terms is a list of tuples
      std::stringstream ss;
      int i = 0;
      for (auto term : terms) {
        auto term_tuple = term.cast<py::tuple>();
        if (!term_tuple.empty()) {
          ss << terms[term].cast<std::complex<double>>() << " ";
          for (auto element : term_tuple) {
            auto element_pair = element.cast<std::pair<int, std::string>>();
            ss << element_pair.second << element_pair.first << " ";
          }
        } else {
          // this was identity

          try {
            auto coeff = terms[term].cast<double>();
            ss << coeff;
          } catch (std::exception &e) {
            try {
              auto coeff = terms[term].cast<std::complex<double>>();
              ss << coeff;
            } catch (std::exception &e) {
              qukkos::error(
                  "Could not cast identity coefficient to double or complex.");
            }
          }
        }
        i++;
        if (i != py::len(terms)) {
          ss << " + ";
        }
      }
      return qukkos::createOperator(ss.str());
    }
  } else if (py::hasattr(op, "toString") && py::hasattr(op, "observe")) {
    auto string_rep = op.attr("toString");
    auto op_str = string_rep().cast<std::string>();
    if (op_str.find("^") != std::string::npos) {
      return qukkos::createOperator("fermion", op_str);
    } else {
      return qukkos::createOperator(op_str);
    }
  } else {
    // throw an error
    qukkos::error(
        "Invalid python object passed as a QUKKOS Operator/Observable. "
        "Currently, we only accept OpenFermion datastructures.");
    return qukkos::Operator();
  }
}

}  // namespace

namespace qukkos {

// PyObjectiveFunction implements ObjectiveFunction to
// enable the utility of pythonic quantum kernels with the
// existing qukkos ObjectiveFunction infrastructure. This class
// keeps track of the quantum kernel as a py::object, which it uses
// in tandem with the QUKKOS QJIT engine to create an executable
// functor representation of the quantum code at runtime. It exposes
// the ObjectiveFunction operator()() overloads to map vector<double>
// x to the correct pythonic argument structure. It delegates to the
// usual helper ObjectiveFunction (like vqe) for execution of the
// actual pre-, execution, and post-processing.
class PyObjectiveFunction : public qukkos::ObjectiveFunction {
 protected:
  py::object py_kernel;
  std::shared_ptr<ObjectiveFunction> helper;
  xacc::internal_compiler::qreg qreg;
  QJIT qjit;

 public:
  const std::string name() const override { return "py-objective-impl"; }
  const std::string description() const override { return ""; }
  PyObjectiveFunction(py::object q, qukkos::Operator qq, const int n_dim,
                      const std::string &helper_name,
                      xacc::HeterogeneousMap opts = {})
      : py_kernel(q) {
    // Set the OptFunction dimensions
    _dim = n_dim;

    qreg = ::qalloc(qq.nBits());

    // Set the helper objective
    helper = xacc::getService<qukkos::ObjectiveFunction>(helper_name);

    // Store the observable pointer and give it to the helper
    observable = qq;
    options = opts;
    options.insert("observable", observable);
    helper->set_options(options);
    helper->update_observable(observable);

    // Extract the QJIT source code
    auto src = py_kernel.attr("get_internal_src")().cast<std::string>();
    auto extra_cpp_src =
        py_kernel.attr("get_extra_cpp_code")().cast<std::string>();
    auto sorted_kernel_deps = py_kernel.attr("get_sorted_kernels_deps")()
                                  .cast<std::vector<std::string>>();

    // QJIT compile
    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true, sorted_kernel_deps, extra_cpp_src);
    qjit.write_cache();
  }

  // PyObjectiveFunction(py::object q, std::shared_ptr<qukkos::Observable> &qq,
  //                     const int n_dim, const std::string &helper_name,
  //                     xacc::HeterogeneousMap opts = {})
  //     : py_kernel(q) {
  //   // Set the OptFunction dimensions
  //   _dim = n_dim;
  //   qreg = ::qalloc(qq->nBits());

  //   // Set the helper objective
  //   helper = xacc::getService<qukkos::ObjectiveFunction>(helper_name);

  //   // Store the observable pointer and give it to the helper
  //   observable = qq;
  //   options = opts;
  //   options.insert("observable", observable);
  //   helper->set_options(options);
  //   helper->update_observable(observable);

  //   // Extract the QJIT source code
  //   auto src = py_kernel.attr("get_internal_src")().cast<std::string>();
  //   auto extra_cpp_src =
  //       py_kernel.attr("get_extra_cpp_code")().cast<std::string>();
  //   auto sorted_kernel_deps = py_kernel.attr("get_sorted_kernels_deps")()
  //                                 .cast<std::vector<std::string>>();

  //   // QJIT compile
  //   // this will be fast if already done, and we just do it once
  //   qjit.jit_compile(src, true, sorted_kernel_deps, extra_cpp_src);
  //   qjit.write_cache();
  // }
  // Evaluate this ObjectiveFunction at the dictionary of kernel args,
  // return the scalar value
  double operator()(const KernelArgDict args, std::vector<double> &dx) {
    std::function<std::shared_ptr<qukkos::CompositeInstruction>(
        std::vector<double>)>
        kernel_evaluator = [&](std::vector<double> x) {
          // qreg = ::qalloc(observable->nBits());
          // std::cout << "Allocating " << qreg.name() << "\n";
          auto _args =
              py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
          // Map the kernel args to a hetmap
          xacc::HeterogeneousMap m;
          for (auto &item : _args) {
            KernelArgDictToHeterogeneousMap vis(m, item.first);
            mpark::visit(vis, item.second);
          }

          // Get the kernel as a CompositeInstruction
          auto kernel_name =
              py_kernel.attr("kernel_name")().cast<std::string>();
          return qjit.extract_composite_with_hetmap(kernel_name, m);
        };

    kernel = kernel_evaluator(current_iterate_parameters);
    helper->update_kernel(kernel);
    helper->update_options("kernel-evaluator", kernel_evaluator);

    return (*helper)(qreg, dx);
  }

  // Evaluate this ObjectiveFunction at the parameters x
  double operator()(const std::vector<double> &x,
                    std::vector<double> &dx) override {
    current_iterate_parameters = x;
    helper->update_current_iterate_parameters(x);

    // Translate x into kernel args
    // qreg = ::qalloc(observable->nBits());
    auto args = py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
    // args will be a dictionary, arg_name to arg
    return operator()(args, dx);
  }

  virtual double operator()(xacc::internal_compiler::qreg &qreg,
                            std::vector<double> &dx) {
    throw std::bad_function_call();
    return 0.0;
  }

  xacc::internal_compiler::qreg get_qreg() override { return qreg; }
};

// PyKernelFunctor is a subtype of KernelFunctor from the qsim library
// that returns a CompositeInstruction representation of a pythonic
// quantum kernel given a vector of parameters x. This will
// leverage the QJIT infrastructure to create executable functor
// representation of the python kernel.
class PyKernelFunctor : public qukkos::KernelFunctor {
 protected:
  py::object py_kernel;
  QJIT qjit;
  std::size_t n_qubits;

 public:
  PyKernelFunctor(py::object q, const std::size_t nq, const std::size_t np)
      : py_kernel(q), n_qubits(nq) {
    nbParams = np;
    auto src = py_kernel.attr("get_internal_src")().cast<std::string>();
    auto extra_cpp_src =
        py_kernel.attr("get_extra_cpp_code")().cast<std::string>();
    auto sorted_kernel_deps = py_kernel.attr("get_sorted_kernels_deps")()
                                  .cast<std::vector<std::string>>();

    // this will be fast if already done, and we just do it once
    qjit.jit_compile(src, true, sorted_kernel_deps, extra_cpp_src);
    qjit.write_cache();
  }

  // Delegate to QJIT to create a CompositeInstruction representation
  // of the pythonic quantum kernel.
  std::shared_ptr<qukkos::CompositeInstruction> evaluate_kernel(
      const std::vector<double> &x) override {
    // Translate x into kernel args
    auto qreg = ::qalloc(n_qubits);
    auto args = py_kernel.attr("translate")(qreg, x).cast<KernelArgDict>();
    xacc::HeterogeneousMap m;
    for (auto &item : args) {
      KernelArgDictToHeterogeneousMap vis(m, item.first);
      mpark::visit(vis, item.second);
    }
    auto kernel_name = py_kernel.attr("kernel_name")().cast<std::string>();
    return qjit.extract_composite_with_hetmap(kernel_name, m);
  }
};
}  // namespace qukkos

PYBIND11_MODULE(_pyqukkos, m) {
  m.doc() = "Python bindings for QUKKOS.";

  py::class_<AllowedKernelArgTypes>(
      m, "AllowedKernelArgTypes",
      "The AllowedKernelArgTypes provides a variant structure "
      "to provide parameters to qukkos quantum kernels HeterogeneousMaps.")
      .def(py::init<int>(), "Construct as an int.")
      .def(py::init<bool>(), "Construct as a bool")
      .def(py::init<double>(), "Construct as a double.")
      .def(py::init<std::string>(), "Construct as a string.")
      .def(py::init<xacc::internal_compiler::qreg>(), "Construct as qreg")
      .def(py::init<std::vector<double>>(), "Construct as a List[double].");

  // Expose QUKKOS API functions
  // Handle QUKKOS CLI arguments:
  // when using via Python, we use this to set those runtime parameters.
  m.def(
      "Initialize",
      [](py::kwargs kwargs) {
        if (kwargs) {
          // QRT (if provided) must be set before quantum::initialize
          if (kwargs.contains("qrt")) {
            const auto value = std::string(py::str(kwargs["qrt"]));
            // QRT (if provided) should be set before quantum::initialize
            ::quantum::set_qrt(value);
          }

          for (auto arg : kwargs) {
            const auto key = std::string(py::str(arg.first));
            // Handle "qpu" key
            if (key == "qpu") {
              const auto value = std::string(py::str(arg.second));
              ::quantum::initialize(value, "empty");
            } else if (key == "shots") {
              const auto value = arg.second.cast<int>();
              ::quantum::set_shots(value);
            } else if (key == "opt") {
              const auto value = arg.second.cast<int>();
              xacc::internal_compiler::__opt_level = value;
            } else if (key == "print-opt-stats") {
              const auto value = arg.second.cast<bool>();
              xacc::internal_compiler::__print_opt_stats = value;
            } else if (key == "placement") {
              const auto value = std::string(py::str(arg.second));
              xacc::internal_compiler::__placement_name = value;
            } else if (key == "opt-pass") {
              const auto value = std::string(py::str(arg.second));
              xacc::internal_compiler::__user_opt_passes = value;
            } else if (key == "qubit-map") {
              const auto value = std::string(py::str(arg.second));
              xacc::internal_compiler::__qubit_map =
                  xacc::internal_compiler::parse_qubit_map(value.c_str());
            }
            /// TODO: handle other CLI parameters.
          }
        }
      },
      "Initialize QUKKOS runtime environment.");

  py::class_<qukkos::Optimizer, std::shared_ptr<qukkos::Optimizer>>(
      m, "Optimizer",
      "The Optimizer interface provides optimization routine implementations "
      "for use in algorithms.")
      .def(py::init<>(), "")
      .def("name", &qukkos::Optimizer::name, "")
      .def(
          "optimize",
          [&](qukkos::Optimizer &o, py::function &f, const int ndim) {
            qukkos::ObjectiveFunction opt(
                [&](const std::vector<double> &x, std::vector<double> &grad) {
                  auto ret = f(x);
                  if (py::isinstance<py::tuple>(ret)) {
                    auto result =
                        ret.cast<std::pair<double, std::vector<double>>>();
                    for (int i = 0; i < grad.size(); i++) {
                      grad[i] = result.second[i];
                    }
                    return result.first;
                  } else {
                    return ret.cast<double>();
                  }
                },
                ndim);
            return o.optimize(opt);
          },
          "")
      .def(
          "optimize",
          [&](qukkos::Optimizer &o, py::object &f) {
            if (!py::hasattr(f, "__call__")) {
              xacc::error(
                  "Invalid object passed to optimizer optimize, must have "
                  "__call__ implemented.");
            }
            if (!py::hasattr(f, "dimensions")) {
              xacc::error(
                  "Invalid object passed to optimizer optimize, must have "
                  "dimensions() implemented.");
            }
            qukkos::ObjectiveFunction opt(
                [&](const std::vector<double> &x, std::vector<double> &grad) {
                  if (grad.empty()) {
                    return f.attr("__call__")(x).cast<double>();
                  } else {
                    auto result =
                        f.attr("__call__")(x, grad)
                            .cast<std::pair<double, std::vector<double>>>();
                    grad = result.second;
                    return result.first;
                  }
                },
                f.attr("dimensions")().cast<int>());
            return o.optimize(opt);
          },
          "");

  py::class_<qukkos::Operator>(m, "Operator")
      .def(py::init<const std::string &, const std::string &>())
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self *= py::self)
      .def(py::self *= double())
      .def(py::self * double())
      .def(double() * py::self)
      .def("__add__", [](const qukkos::Operator &op,
                         double idCoeff) { return op + idCoeff; })
      .def("__radd__", [](const qukkos::Operator &op,
                          double idCoeff) { return op + idCoeff; })
      .def("__sub__", [](const qukkos::Operator &op,
                         double idCoeff) { return op - idCoeff; })
      .def("__rsub__", [](const qukkos::Operator &op,
                          double idCoeff) { return idCoeff - op; })
      .def(py::self * py::self)
      .def(py::self *= std::complex<double>())
      .def(py::self * std::complex<double>())
      .def(std::complex<double>() * py::self)
      .def(py::self -= py::self)
      .def(py::self - py::self)
      .def("__eq__", &qukkos::Operator::operator==)
      .def("__repr__", &qukkos::Operator::toString)
      .def("toString", &qukkos::Operator::toString)
      .def("toBinaryVectors", &qukkos::Operator::toBinaryVectors)
      .def("nQubits", &qukkos::Operator::nQubits)
      .def("coefficient", &qukkos::Operator::coefficient)
      .def("nBits", &qukkos::Operator::nBits)
      .def("getNonIdentitySubTerms", &qukkos::Operator::getNonIdentitySubTerms)
      .def("getIdentitySubTerm", &qukkos::Operator::getIdentitySubTerm)
      .def("to_numpy", [](qukkos::Operator &op) {
        auto mat_el = op.to_sparse_matrix();
        auto size = std::pow(2, op.nBits());
        Eigen::MatrixXcd mat = Eigen::MatrixXcd::Zero(size, size);
        for (auto el : mat_el) {
          mat(el.row(), el.col()) = el.coeff();
        }
        return mat;
      });

  py::class_<qukkos::CompositeInstruction,
             std::shared_ptr<qukkos::CompositeInstruction>>(
      m, "CompositeInstruction", "")
      .def("nLogicalBits", &qukkos::CompositeInstruction::nLogicalBits, "")
      .def("nPhysicalBits", &qukkos::CompositeInstruction::nPhysicalBits, "")
      .def("nInstructions", &qukkos::CompositeInstruction::nInstructions, "")
      .def("getInstruction", &qukkos::CompositeInstruction::getInstruction, "")
      .def("getInstructions", &qukkos::CompositeInstruction::getInstructions, "")
      // .def("removeInstruction",
      // &qukkos::CompositeInstruction::removeInstruction,
      //      "")
      // .def("replaceInstruction",
      //      &qukkos::CompositeInstruction::replaceInstruction, "")
      // .def("insertInstruction",
      // &qukkos::CompositeInstruction::insertInstruction,
      //      "")
      .def("__str__", &qukkos::CompositeInstruction::toString, "")
      .def("name", &qukkos::CompositeInstruction::name, "")
      .def("toString", &qukkos::CompositeInstruction::toString, "")
      .def("depth", &qukkos::CompositeInstruction::depth, "")
      .def("as_xacc", &qukkos::CompositeInstruction::as_xacc, "");

  // Expose QUKKOS API functions
  m.def(
      "createOptimizer",
      [](const std::string &name, PyHeterogeneousMap p = {}) {
        return qukkos::createOptimizer(name, heterogeneousMapConvert(p));
      },
      py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
      py::return_value_policy::reference,
      "Return the Optimizer with given name.");
  m.def("set_verbose", &qukkos::set_verbose, "");
  m.def("createTransformation", &qukkos::createTransformation, "");
  m.def(
      "set_qpu",
      [](const std::string &name, PyHeterogeneousMap p = {}) {
        xacc::internal_compiler::qpu =
            xacc::getAccelerator(name, heterogeneousMapConvert(p));
      },
      py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
      "Set the QPU backend.");

  m.def(
      "set_opt_level",
      [](int level) { xacc::internal_compiler::__opt_level = level; },
      py::arg("level"), "Set QUKKOS runtime optimization level.");

  m.def(
      "add_pass",
      [](const std::string &pass_name) {
        // Note: we expect __user_opt_passes to be a comma-separated list of
        // pass names.
        if (xacc::internal_compiler::__user_opt_passes.empty()) {
          xacc::internal_compiler::__user_opt_passes = pass_name;
        } else {
          xacc::internal_compiler::__user_opt_passes += ("," + pass_name);
        }
      },
      py::arg("pass_name"),
      "Add an optimization pass to be run by the PassManager.");

  m.def(
      "get_placement_names",
      []() {
        std::vector<std::string> result;
        auto ir_transforms = xacc::getServices<qukkos::IRTransformation>();
        for (const auto &plugin : ir_transforms) {
          if (plugin->type() == xacc::IRTransformationType::Placement) {
            result.emplace_back(plugin->name());
          }
        }
        return result;
      },
      "Get names of all available placement plugins.");

  m.def(
      "set_placement",
      [](const std::string &placement_name) {
        xacc::internal_compiler::__placement_name = placement_name;
      },
      py::arg("placement_name"), "Set the placement strategy.");

  m.def(
      "qalloc", [](int size) { return ::qalloc(size); },
      "Allocate qubit register.");
  m.def("set_shots", &qukkos::set_shots, "");
  m.def("set_validate",
        [](bool validate) {
          xacc::internal_compiler::__validate_nisq_execution = validate;
        },
        "Enable/disable backend execution validation.");
  py::class_<xacc::internal_compiler::qubit>(m, "qubit", "");
  py::class_<xacc::internal_compiler::qreg>(m, "qreg", "")
      .def("size", &xacc::internal_compiler::qreg::size, "")
      .def("print", &xacc::internal_compiler::qreg::print, "")
      .def("counts", &xacc::internal_compiler::qreg::counts, "")
      .def(
          "extract_range",
          [](xacc::internal_compiler::qreg &q, std::size_t start,
             std::size_t end) {
            std::vector<std::size_t> r{start, end};
            return q.extract_range(r);
          },
          "")
      .def(
          "head", [](xacc::internal_compiler::qreg &q) { return q.head(); }, "")
      .def(
          "head",
          [](xacc::internal_compiler::qreg &q, const std::size_t n) {
            return q.head(n);
          },
          "")
      .def(
          "tail", [](xacc::internal_compiler::qreg &q) { return q.tail(); }, "")
      .def(
          "tail",
          [](xacc::internal_compiler::qreg &q, const std::size_t n) {
            return q.tail(n);
          },
          "")
      .def("exp_val_z", &xacc::internal_compiler::qreg::exp_val_z, "")
      .def(
          "results",
          [](xacc::internal_compiler::qreg &q) {
            auto buffer = q.results_shared();
            return buffer;
          },
          "")
      .def(
          "getInformation",
          [](xacc::internal_compiler::qreg &q, const std::string &key) {
            return q.results()->getInformation(key);
          },
          "")
      .def(
          "__getitem__",
          [](xacc::internal_compiler::qreg &q, int index) { return q[index]; },
          "");
  // m.def("createObjectiveFunction", [](const std::string name, ))
  py::class_<qukkos::QJIT, std::shared_ptr<qukkos::QJIT>>(m, "QJIT", "")
      .def(py::init<>(), "")
      .def("write_cache", &qukkos::QJIT::write_cache, "")
      .def(
          "jit_compile",
          [](qukkos::QJIT &qjit, const std::string src) {
            bool turn_on_hetmap_kernel_ctor = true;
            qjit.jit_compile(src, turn_on_hetmap_kernel_ctor, {});
          },
          "")
      .def(
          "internal_python_jit_compile",
          [](qukkos::QJIT &qjit, const std::string src,
             const std::vector<std::string> &dependency = {},
             const std::string &extra_cpp_code = "",
             std::vector<std::string> extra_headers = {}) {
            bool turn_on_hetmap_kernel_ctor = true;
            qjit.jit_compile(src, turn_on_hetmap_kernel_ctor, dependency,
                             extra_cpp_code, extra_headers);
          },
          "")
      .def(
          "run_syntax_handler",
          [](qukkos::QJIT &qjit, const std::string src) {
            return qjit.run_syntax_handler(src, true);
          },
          "")
      .def(
          "invoke",
          [](qukkos::QJIT &qjit, const std::string name, KernelArgDict args) {
            xacc::HeterogeneousMap m;
            for (auto &item : args) {
              KernelArgDictToHeterogeneousMap vis(m, item.first);
              mpark::visit(vis, item.second);
            }
            qjit.invoke_with_hetmap(name, m);
          },
          "")

      .def("extract_composite",
           [](qukkos::QJIT &qjit, const std::string name, KernelArgDict args) {
             xacc::HeterogeneousMap m;
             for (auto &item : args) {
               KernelArgDictToHeterogeneousMap vis(m, item.first);
               mpark::visit(vis, item.second);
             }
             return qjit.extract_composite_with_hetmap(name, m);
           })
      .def("internal_as_unitary",
           [](qukkos::QJIT &qjit, const std::string name, KernelArgDict args) {
             xacc::HeterogeneousMap m;
             for (auto &item : args) {
               KernelArgDictToHeterogeneousMap vis(m, item.first);
               mpark::visit(vis, item.second);
             }
             auto composite = qjit.extract_composite_with_hetmap(name, m);
             return qukkos::__internal__::map_composite_to_unitary_matrix(
                 composite);
             //  auto n_qubits = composite->nLogicalBits();
             //  qukkos::KernelToUnitaryVisitor visitor(n_qubits);
             //  InstructionIterator iter(composite);
             //  while (iter.hasNext()) {
             //    auto inst = iter.next();
             //    if (!inst->isComposite() && inst->isEnabled()) {
             //      inst->accept(&visitor);
             //    }
             //  }
             //  return visitor.getMat();
           })
      .def(
          "get_kernel_function_ptr",
          [](qukkos::QJIT &qjit, const std::string &kernel_name) {
            return qjit.get_kernel_function_ptr(kernel_name);
          },
          "")
      .def(
          "get_native_code",
          [](qukkos::QJIT &qjit, const std::string name, KernelArgDict args, PyHeterogeneousMap options = {}) {
            xacc::HeterogeneousMap m;
            for (auto &item : args) {
              KernelArgDictToHeterogeneousMap vis(m, item.first);
              mpark::visit(vis, item.second);
            }
            auto program = qjit.extract_composite_with_hetmap(name, m);
            xacc::internal_compiler::execute_pass_manager(program);
            return xacc::internal_compiler::get_native_code(
                program, heterogeneousMapConvert(options));
          },
          "");

  py::class_<qukkos::ObjectiveFunction, std::shared_ptr<qukkos::ObjectiveFunction>>(
      m, "ObjectiveFunction", "")
      .def("dimensions", &qukkos::ObjectiveFunction::dimensions, "")
      .def(
          "__call__",
          [](qukkos::ObjectiveFunction &obj, std::vector<double> x) {
            return obj(x);
          },
          "")
      .def(
          "__call__",
          [](qukkos::ObjectiveFunction &obj, std::vector<double> x,
             std::vector<double> &dx) {
            auto val = obj(x, dx);
            return std::make_pair(val, dx);
          },
          "")
      .def("get_qreg", &qukkos::ObjectiveFunction::get_qreg, "");

  m.def(
      "createObjectiveFunction",
      [](py::object kernel, qukkos::Operator &obs, const int n_params) {
        auto q = ::qalloc(obs.nBits());
        std::shared_ptr<qukkos::ObjectiveFunction> obj =
            std::make_shared<qukkos::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe");
        return obj;
      },
      "");
  m.def(
      "createObjectiveFunction",
      [](py::object kernel, py::object &py_obs, const int n_params) {
        auto obs = convertToQUKKOSOperator(py_obs);
        auto q = ::qalloc(obs.nBits());
        std::shared_ptr<qukkos::ObjectiveFunction> obj =
            std::make_shared<qukkos::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe");
        return obj;
      },
      "");
  m.def(
      "createObjectiveFunction",
      [](py::object kernel, qukkos::Operator &obs, const int n_params,
         PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        auto q = ::qalloc(obs.nBits());
        std::shared_ptr<qukkos::ObjectiveFunction> obj =
            std::make_shared<qukkos::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe", nativeHetMap);
        return obj;
      },
      "");
  m.def(
      "createObjectiveFunction",
      [](py::object kernel, py::object &py_obs, const int n_params,
         PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        auto obs = convertToQUKKOSOperator(py_obs);
        auto q = ::qalloc(obs.nBits());
        std::shared_ptr<qukkos::ObjectiveFunction> obj =
            std::make_shared<qukkos::PyObjectiveFunction>(kernel, obs, n_params,
                                                        "vqe", nativeHetMap);
        return obj;
      },
      "");

  m.def(
      "createOperator",
      [](const std::string &repr) { return qukkos::createOperator(repr); }, "");
  m.def(
      "createOperator",
      [](const std::string &type, const std::string &repr) {
        return qukkos::createOperator(type, repr);
      },
      "");
  m.def(
      "createOperator",
      [](const std::string &type, PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        return qukkos::createOperator(type, nativeHetMap);
      },
      "");
  m.def("createOperator", [](const std::string &type, py::object pyobject) {
    return convertToQUKKOSOperator(pyobject, type == "fermion");
  });
  m.def(
      "createObservable",
      [](const std::string &repr) { return qukkos::createOperator(repr); }, "");
  m.def(
      "createObservable",
      [](const std::string &type, const std::string &repr) {
        return qukkos::createOperator(type, repr);
      },
      "");
  m.def(
      "createObservable",
      [](const std::string &type, PyHeterogeneousMap &options) {
        auto nativeHetMap = heterogeneousMapConvert(options);
        return qukkos::createOperator(type, nativeHetMap);
      },
      "");

  m.def(
      "operatorTransform",
      [](const std::string &type, qukkos::Operator &obs) {
        return qukkos::operatorTransform(type, obs);
      },
      "");
  m.def(
      "internal_observe",
      [](std::shared_ptr<qukkos::CompositeInstruction> kernel,
         qukkos::Operator &obs) {
        auto q = ::qalloc(obs.nBits());
        return qukkos::observe(kernel, obs, q);
      },
      "");
  m.def(
      "internal_observe",
      [](std::shared_ptr<qukkos::CompositeInstruction> kernel, py::object obs) {
        auto observable = convertToQUKKOSOperator(obs);
        auto q = ::qalloc(observable.nBits());
        return qukkos::observe(kernel, observable, q);
      },
      "");
  m.def("internal_observe",
        [](std::shared_ptr<qukkos::CompositeInstruction> kernel,
           qukkos::Operator &obs, xacc::internal_compiler::qreg &q) {
          return qukkos::observe(kernel, obs, q);
        },
        "");
  m.def("internal_observe",
        [](std::shared_ptr<qukkos::CompositeInstruction> kernel, py::object obs,
           xacc::internal_compiler::qreg &q) {
          auto observable = convertToQUKKOSOperator(obs);
          return qukkos::observe(kernel, observable, q);
        },
        "");
  m.def(
      "internal_autograd",
      [](py::function &kernel_eval, qukkos::Operator &obs,
         std::vector<double> x) -> std::tuple<double, std::vector<double>> {
        try {
          std::function<std::shared_ptr<qukkos::CompositeInstruction>(
              std::vector<double>)>
              cpp_kernel_eval = [&](std::vector<double> x_vec) {
                auto ret = kernel_eval(x_vec);
                auto kernel =
                    ret.cast<std::shared_ptr<qukkos::CompositeInstruction>>();
                return kernel;
              };

          auto gradiend_method = qukkos::__internal__::get_gradient_method(
              qukkos::__internal__::DEFAULT_GRADIENT_METHOD, cpp_kernel_eval,
              obs);

          auto program = cpp_kernel_eval(x);
          auto q = ::qalloc(
              std::max((int)obs.nBits(), (int)program->nPhysicalBits()));
          auto cost_val = qukkos::observe(program, obs, q);
          auto dx = (*gradiend_method)(x, cost_val);
          return std::make_tuple(cost_val, dx);
        } catch (std::exception &e) {
          qukkos::error("Invalid kernel evaluator.");
          return std::make_tuple(0.0, std::vector<double>{});
        }
      },
      "");

  m.def("internal_get_all_instructions", []() -> std::vector<py::tuple> {
    auto insts = xacc::getServices<xacc::Instruction>();
    std::vector<py::tuple> ret;
    for (auto inst : insts) {
      if (!inst->isComposite()) {
        ret.push_back(py::make_tuple(inst->name(), inst->nRequiredBits(),
                                     inst->isParameterized()));
      }
    }
    return ret;
  });

#ifdef QUKKOS_BUILD_MLIR_PYTHON_API
  m.def("openqasm_to_mlir",
        [](const std::string &oqasm_src, const std::string &kernel_name,
           bool add_entry_point, int opt_level, bool qiskit_compat) {
          std::map<std::string, std::string> extra_args;
          if (qiskit_compat) {
            extra_args.insert({"qiskit_compat", "true"});
          }
          return qukkos::mlir_compile(oqasm_src, kernel_name,
                                    qukkos::OutputType::MLIR, add_entry_point,
                                    opt_level, extra_args);
        });

  m.def("openqasm_to_llvm_mlir",
        [](const std::string &oqasm_src, const std::string &kernel_name,
           bool add_entry_point, int opt_level, bool qiskit_compat) {
          std::map<std::string, std::string> extra_args;
          if (qiskit_compat) {
            extra_args.insert({"qiskit_compat", "true"});
          }
          return qukkos::mlir_compile(oqasm_src, kernel_name,
                                    qukkos::OutputType::LLVMMLIR, add_entry_point,
                                    opt_level, extra_args);
        });

  m.def("openqasm_to_llvm_ir", [](const std::string &oqasm_src,
                                  const std::string &kernel_name,
                                  bool add_entry_point, int opt_level, bool qiskit_compat) {
    std::map<std::string, std::string> extra_args;
    if (qiskit_compat) {
      extra_args.insert({"qiskit_compat", "true"});
    }
    return qukkos::mlir_compile(oqasm_src, kernel_name, qukkos::OutputType::LLVMIR,
                              add_entry_point, opt_level, extra_args);
  });
#endif

  // QuaSiMo sub-module bindings:
  {
    py::module qsim =
        m.def_submodule("QuaSiMo", "QUKKOS's python QuaSiMo submodule");

    // QuantumSimulationModel bindings:
    py::class_<qukkos::QuaSiMo::QuantumSimulationModel>(
        qsim, "QuantumSimulationModel",
        "The QuantumSimulationModel captures the quantum simulation problem "
        "description.")
        .def(py::init<>())
        .def(
            "__str__",
            [](qukkos::QuaSiMo::QuantumSimulationModel &self) {
              std::stringstream ss;
              ss << "{ observable: " << self.observable->toString() << "}";
              return ss.str();
            },
            "");

    // ModelFactory bindings:
    py::class_<qukkos::QuaSiMo::ModelFactory>(
        qsim, "ModelFactory",
        "The ModelFactory interface provides methods to "
        "construct QuaSiMo problem models.")
        .def(py::init<>())
        .def(
            "createModel",
            [](qukkos::Operator &obs, qukkos::QuaSiMo::TdObservable ham_func)
                -> qukkos::QuaSiMo::QuantumSimulationModel {
              return qukkos::QuaSiMo::ModelFactory::createModel(obs, ham_func);
            },
            "Return the Model for a time-dependent problem.")
        .def(
            "createModel",
            [](py::object py_kernel, qukkos::Operator &obs,
               const int n_params) -> qukkos::QuaSiMo::QuantumSimulationModel {
              qukkos::QuaSiMo::QuantumSimulationModel model;
              auto nq = obs.nBits();
              auto kernel_functor = std::make_shared<qukkos::PyKernelFunctor>(
                  py_kernel, nq, n_params);
              model.observable = &obs;
              model.user_defined_ansatz = kernel_functor;
              return model;
            },
            "")
        .def(
            "createModel",
            [](py::object py_kernel, py::object &py_obs,
               const int n_params) -> qukkos::QuaSiMo::QuantumSimulationModel {
              qukkos::QuaSiMo::QuantumSimulationModel model;
              static auto obs = convertToQUKKOSOperator(py_obs);
              auto nq = obs.nBits();
              auto kernel_functor = std::make_shared<qukkos::PyKernelFunctor>(
                  py_kernel, nq, n_params);
              model.observable = &obs;
              model.user_defined_ansatz = kernel_functor;
              return model;
            },
            "")
        .def(
            "createModel",
            [](py::object py_kernel, qukkos::Operator &obs, const int n_qubits,
               const int n_params) -> qukkos::QuaSiMo::QuantumSimulationModel {
              qukkos::QuaSiMo::QuantumSimulationModel model;
              auto kernel_functor = std::make_shared<qukkos::PyKernelFunctor>(
                  py_kernel, n_qubits, n_params);
              model.observable = &obs;
              model.user_defined_ansatz = kernel_functor;
              return model;
            },
            "")
        .def(
            "createModel",
            [](py::object py_kernel, std::shared_ptr<qukkos::Operator> &obs,
               const int n_params) -> qukkos::QuaSiMo::QuantumSimulationModel {
              qukkos::QuaSiMo::QuantumSimulationModel model;
              auto nq = obs->nBits();
              auto kernel_functor = std::make_shared<qukkos::PyKernelFunctor>(
                  py_kernel, nq, n_params);
              model.observable = obs.get();
              model.user_defined_ansatz = kernel_functor;
              return model;
            },
            "")

        .def(
            "createModel",
            [](py::object py_kernel, std::shared_ptr<qukkos::Operator> &obs,
               const int n_qubits,
               const int n_params) -> qukkos::QuaSiMo::QuantumSimulationModel {
              qukkos::QuaSiMo::QuantumSimulationModel model;
              auto kernel_functor = std::make_shared<qukkos::PyKernelFunctor>(
                  py_kernel, n_qubits, n_params);
              model.observable = obs.get();
              model.user_defined_ansatz = kernel_functor;
              return model;
            },
            "")
        .def(
            "createModel",
            [](qukkos::Operator &obs) -> qukkos::QuaSiMo::QuantumSimulationModel {
              return qukkos::QuaSiMo::ModelFactory::createModel(obs);
            },
            "")
        .def(
            "createModel",
            [](py::object &py_obs) -> qukkos::QuaSiMo::QuantumSimulationModel {
              qukkos::QuaSiMo::QuantumSimulationModel model;
              static auto obs = convertToQUKKOSOperator(py_obs);
              model.observable = &obs;
              return model;
            },
            "")
        .def(
            "createModel",
            [](qukkos::QuaSiMo::ModelFactory::ModelType type,
               PyHeterogeneousMap &params)
                -> qukkos::QuaSiMo::QuantumSimulationModel {
              auto nativeHetMap = heterogeneousMapConvert(params);
              return qukkos::QuaSiMo::ModelFactory::createModel(type,
                                                              nativeHetMap);
            },
            "Create a model of a supported type.");
    py::enum_<qukkos::QuaSiMo::ModelFactory::ModelType>(m, "ModelType")
        .value("Heisenberg", qukkos::QuaSiMo::ModelFactory::ModelType::Heisenberg)
        .export_values();
    // CostFunctionEvaluator bindings
    py::class_<qukkos::QuaSiMo::CostFunctionEvaluator,
               std::shared_ptr<qukkos::QuaSiMo::CostFunctionEvaluator>,
               qukkos::QuaSiMo::PyCostFunctionEvaluator>(
        qsim, "CostFunctionEvaluator",
        "The CostFunctionEvaluator interface provides methods to "
        "evaluate the observable operator expectation value on quantum "
        "backends.")
        .def(py::init<>())
        .def(
            "initialize",
            [](qukkos::QuaSiMo::CostFunctionEvaluator &self,
               qukkos::Operator &obs) { return self.initialize(&obs); },
            "Initialize the evaluator")
        .def(
            "evaluate",
            [](qukkos::QuaSiMo::CostFunctionEvaluator &self,
               std::shared_ptr<qukkos::CompositeInstruction> state_prep)
                -> double { return self.evaluate(state_prep); },
            "Initialize the evaluator");
    qsim.def(
        "getObjEvaluator",
        [](qukkos::Operator &obs, const std::string &name = "default",
           py::dict p = {}) {
          return qukkos::QuaSiMo::getObjEvaluator(obs, name);
        },
        py::arg("obs"), py::arg("name") = "default", py::arg("p") = py::dict(),
        py::return_value_policy::reference,
        "Return the CostFunctionEvaluator.");

    // QuantumSimulationWorkflow bindings
    py::class_<qukkos::QuaSiMo::QuantumSimulationWorkflow,
               std::shared_ptr<qukkos::QuaSiMo::QuantumSimulationWorkflow>,
               qukkos::QuaSiMo::PyQuantumSimulationWorkflow>(
        qsim, "QuantumSimulationWorkflow",
        "The QuantumSimulationWorkflow interface provides methods to "
        "execute a quantum simulation workflow.")
        .def(py::init<>())
        .def(
            "execute",
            [](qukkos::QuaSiMo::QuantumSimulationWorkflow &self,
               const qukkos::QuaSiMo::QuantumSimulationModel &model)
                -> qukkos::QuaSiMo::QuantumSimulationResult {
              return self.execute(model);
            },
            "Execute the workflow for the input problem model.");
    qsim.def(
        "getWorkflow",
        [](const std::string &name, PyHeterogeneousMap p = {}) {
          auto nativeHetMap = heterogeneousMapConvert(p);
          return qukkos::QuaSiMo::getWorkflow(name, nativeHetMap);
        },
        py::arg("name"), py::arg("p") = PyHeterogeneousMap(),
        py::return_value_policy::reference,
        "Return the quantum simulation workflow.");
  }
    
}
