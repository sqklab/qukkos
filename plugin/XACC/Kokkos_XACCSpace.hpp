//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

#ifndef KOKKOS_IMPL_PUBLIC_INCLUDE
#include <Kokkos_Macros.hpp>
static_assert(false,
              "Including non-public Kokkos header files is not allowed.");
#endif

#ifndef KOKKOS_XACC_SPACE_HPP
#define KOKKOS_XACC_SPACE_HPP

#include <Kokkos_Concepts.hpp>
#include <impl/Kokkos_Tools.hpp>

#include <XACC.h>
#include <iosfwd>

namespace Kokkos::Experimental {

class XACC;

class XACCSpace {
 public:
  using memory_space    = XACCSpace;
  using execution_space = XACC;
  using device_type     = Kokkos::Device<execution_space, memory_space>;

  using size_type = size_t;

  XACCSpace() = default;

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const Kokkos::Experimental::XACC& exec_space,
                 const size_t arg_alloc_size) const;
  void* allocate(const Kokkos::Experimental::XACC& exec_space,
                 const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;
  void* allocate(const size_t arg_alloc_size) const;
  void* allocate(const char* arg_label, const size_t arg_alloc_size,
                 const size_t arg_logical_size = 0) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;
  void deallocate(const char* arg_label, void* const arg_alloc_ptr,
                  const size_t arg_alloc_size,
                  const size_t arg_logical_size = 0) const;

  static constexpr char const* name() { return "XACCSpace"; }

 private:
  void* impl_allocate(const Kokkos::Experimental::XACC& exec_space,
                      const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void* impl_allocate(const char* arg_label, const size_t arg_alloc_size,
                      const size_t arg_logical_size = 0,
                      const Kokkos::Tools::SpaceHandle =
                          Kokkos::Tools::make_space_handle(name())) const;
  void impl_deallocate(const char* arg_label, void* const arg_alloc_ptr,
                       const size_t arg_alloc_size,
                       const size_t arg_logical_size = 0,
                       const Kokkos::Tools::SpaceHandle =
                           Kokkos::Tools::make_space_handle(name())) const;
};

}  // namespace Kokkos::Experimental

/*--------------------------------------------------------------------------*/

template <>
struct Kokkos::Impl::MemorySpaceAccess<Kokkos::HostSpace,
                                       Kokkos::Experimental::XACCSpace> {
#if defined(KOKKOS_ENABLE_XACC_FORCE_HOST_AS_DEVICE)
  enum : bool{assignable = true};
  enum : bool{accessible = true};
#else
  enum : bool { assignable = false };
  enum : bool { accessible = false };
#endif
  enum : bool { deepcopy = true };
};

template <>
struct Kokkos::Impl::MemorySpaceAccess<Kokkos::Experimental::XACCSpace,
                                       Kokkos::HostSpace> {
#if defined(KOKKOS_ENABLE_XACC_FORCE_HOST_AS_DEVICE)
  enum : bool{assignable = true};
  enum : bool{accessible = true};
#else
  enum : bool { assignable = false };
  enum : bool { accessible = false };
#endif
  enum : bool { deepcopy = true };
};

template <>
struct Kokkos::Impl::MemorySpaceAccess<Kokkos::Experimental::XACCSpace,
                                       Kokkos::Experimental::XACCSpace> {
  enum : bool { assignable = true };
  enum : bool { accessible = true };
  enum : bool { deepcopy = true };
};
/*--------------------------------------------------------------------------*/

#endif
