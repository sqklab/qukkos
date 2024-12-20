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
#ifndef XACC_QALLOC_HPP_
#define XACC_QALLOC_HPP_

#include <map>
#include <string>
#include <memory>
#include <vector>

namespace xacc {
class AcceleratorBuffer;
class Observable;
namespace internal_compiler {
// We treat a qubit as a pair { QREG_VAR_NAME, QREG_IDX }
// for use with qukkos QRT API we also need to keep track of the buffer
struct qubit;
struct AllocEventListener {
  virtual void onAllocate(qubit* qubit) = 0;
  virtual void onDealloc(qubit* qubit) = 0;
};

extern AllocEventListener *getGlobalQubitManager();
extern void setGlobalQubitManager(AllocEventListener *);

// This is the struct that will be embedded into the qubit
// to track the **true** lifetime of a qubit.
// It will notify the AllocEventListener once a qubit is deallocated
// (all copies go out of scope)
// Use case: scratch qubits created inside a quantum kernel. 
struct AllocTracker {
  qubit *m_qubit;
  AllocEventListener *listener;
  AllocTracker(qubit *q) : m_qubit(q), listener(getGlobalQubitManager()) {
    listener->onAllocate(m_qubit);
  }

  ~AllocTracker() { listener->onDealloc(m_qubit); }
};

// Qubit allocator interface:
struct QubitAllocator {
  virtual qubit allocate() = 0;
};

struct qubit {
  std::string first;
  std::size_t second;
  xacc::AcceleratorBuffer *buffer;
  xacc::AcceleratorBuffer *results() { return buffer; }

  // New allocation:
  qubit(const std::string &reg_name, size_t idx,
        xacc::AcceleratorBuffer *in_buffer = nullptr)
      : first(reg_name), second(idx), buffer(in_buffer) {
    tracker = std::make_shared<AllocTracker>(this);
  }
  qubit() = default;
  // Having this tracker as a shared_ptr so that we can follow the qubit
  // even if it is copied, e.g. via slicing.
  // Default copy and copy assign should just copy this tracker across.
  std::shared_ptr<AllocTracker> tracker;
};

class qreg;
// Classical register associated with a qubit register to store single-shot
// measurement results. i.e. after Measure(q[0]) => q.creg[0] will contain the
// boolean (true/false) result of the measurement.
class cReg {
public:
  cReg() = default;
  cReg(std::shared_ptr<AcceleratorBuffer> in_buffer);
  bool operator[](std::size_t i);

private:
  std::shared_ptr<AcceleratorBuffer> buffer;
};

// A qreg is a vector of qubits
class qreg {
private:
  std::vector<qubit> internal_qubits;
  std::string random_string(std::size_t length);

protected:
  std::shared_ptr<AcceleratorBuffer> buffer;
  bool been_named_and_stored = false;

public:
  struct Range {
    std::size_t start;
    std::size_t end;
    std::size_t step = 1;
    Range(std::vector<std::size_t> &s);
    Range(std::initializer_list<std::size_t> &&s);
  };

  qreg() = default;
  qreg(const int n);
  qreg(const qreg &other);
  qreg(std::vector<qubit> &qubits);
  auto begin() { return internal_qubits.begin(); }
  auto end() { return internal_qubits.end(); }
  qubit operator[](const std::size_t i);
  qreg extract_range(const std::size_t& start, const std::size_t& end);
  qreg extract_range(const Range &&range);
  qreg extract_qubits(const std::initializer_list<std::size_t> &&qbits);
  qreg head(const std::size_t n_qubits);
  qubit head();
  qubit tail();
  qreg tail(const std::size_t n_qubits);
  AcceleratorBuffer *results();
  std::shared_ptr<AcceleratorBuffer> results_shared();
  std::map<std::string, int> counts();
  double exp_val_z();
  void reset();
  int size();
  void addChild(qreg &q);
  void setName(const char *name);
  void setNameAndStore(const char *name);
  std::string name();
  void store();
  void print();
  double weighted_sum(Observable *obs);
  void write_file(const std::string &file_name);
  // friend std::ostream& operator<<(std::ostream& os, qreg& q);
  // Public member var to simplify the single measurement syntax:
  // i.e. we can access the single measurement results via the syntax:
  // q.creg[i] vs. (*q.results())[i]
  cReg creg;
};
// std::ostream& operator<<(std::ostream& os, qreg& q);
} // namespace internal_compiler
} // namespace xacc

// Optionally provide an allocator:
// The idea is that during code-gen (syntax handling)
// we can add in a special auxillary/ancilla allocator
// which allocate qubits from a shared pool rather than
// create new qubits.
xacc::internal_compiler::qreg
qalloc(const int n,
       xacc::internal_compiler::QubitAllocator *allocator = nullptr);

// __qpu__ indicates this functions is for the QUKKOS Clang Syntax Handler
// and annotated with quantum for the LLVM IR CodeGen
#define __qpu__ [[clang::syntax(qukkos)]] __attribute__((annotate("quantum")))

#endif