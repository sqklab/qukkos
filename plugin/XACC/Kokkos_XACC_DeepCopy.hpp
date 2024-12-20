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

#ifndef KOKKOS_XACC_DEEP_COPY_HPP
#define KOKKOS_XACC_DEEP_COPY_HPP

#include <XACC/Kokkos_XACC.hpp>
#include <XACC/Kokkos_XACCSpace.hpp>

#include <Kokkos_Concepts.hpp>

#include <XACC.h>

template <>
struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::XACCSpace,
                              Kokkos::Experimental::XACCSpace,
                              Kokkos::Experimental::XACC> {
  DeepCopy(void* dst, const void* src, size_t n) {
    // The behavior of acc_memcpy_device when bytes argument is zero is
    // clarified only in the latest XACC specification (V3.2), and thus the
    // value checking is added as a safeguard. (The current NVHPC (V22.5)
    // supports XACC V2.7.)
    if (n > 0) {
      acc_memcpy_device_async(dst, const_cast<void*>(src), n, acc_async_noval);
    }
  }
  DeepCopy(const Kokkos::Experimental::XACC& exec, void* dst,
           const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_device_async(dst, const_cast<void*>(src), n,
                              exec.acc_async_queue());
    }
  }
};

template <class ExecutionSpace>
struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::XACCSpace,
                              Kokkos::Experimental::XACCSpace,
                              ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_device_async(dst, const_cast<void*>(src), n, acc_async_noval);
    }
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<XACCSpace, XACCSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    if (n > 0) {
      acc_memcpy_device_async(dst, const_cast<void*>(src), n, acc_async_noval);
    }
  }
};

template <>
struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::XACCSpace,
                              Kokkos::HostSpace,
                              Kokkos::Experimental::XACC> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0)
      acc_memcpy_to_device_async(dst, const_cast<void*>(src), n,
                                 acc_async_noval);
  }
  DeepCopy(const Kokkos::Experimental::XACC& exec, void* dst,
           const void* src, size_t n) {
    if (n > 0)
      acc_memcpy_to_device_async(dst, const_cast<void*>(src), n,
                                 exec.acc_async_queue());
  }
};

template <class ExecutionSpace>
struct Kokkos::Impl::DeepCopy<Kokkos::Experimental::XACCSpace,
                              Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_to_device_async(dst, const_cast<void*>(src), n,
                                 acc_async_noval);
    }
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<XACCSpace, HostSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    if (n > 0) {
      acc_memcpy_to_device_async(dst, const_cast<void*>(src), n,
                                 acc_async_noval);
    }
  }
};

template <>
struct Kokkos::Impl::DeepCopy<Kokkos::HostSpace,
                              Kokkos::Experimental::XACCSpace,
                              Kokkos::Experimental::XACC> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_from_device_async(dst, const_cast<void*>(src), n,
                                   acc_async_noval);
    }
  }
  DeepCopy(const Kokkos::Experimental::XACC& exec, void* dst,
           const void* src, size_t n) {
    if (n > 0) {
      acc_memcpy_from_device_async(dst, const_cast<void*>(src), n,
                                   exec.acc_async_queue());
    }
  }
};

template <class ExecutionSpace>
struct Kokkos::Impl::DeepCopy<
    Kokkos::HostSpace, Kokkos::Experimental::XACCSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
    if (n > 0)
      acc_memcpy_from_device_async(dst, const_cast<void*>(src), n,
                                   acc_async_noval);
  }
  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence(
        "Kokkos::Impl::DeepCopy<HostSpace, XACCSpace, "
        "ExecutionSpace>::DeepCopy: fence before copy");
    if (n > 0) {
      acc_memcpy_from_device_async(dst, const_cast<void*>(src), n,
                                   acc_async_noval);
    }
  }
};

#endif
