
configure_file("${CMAKE_SOURCE_DIR}/runtime/qukkos_version.in"
               "${CMAKE_BINARY_DIR}/runtime/qukkos_version")
install(FILES "${CMAKE_BINARY_DIR}/runtime/qukkos_version" DESTINATION include/qukkos)

set(LIBRARY_NAME qukkos)

file(GLOB SRC observable/qukkos_observable.cpp 
              optimizer/qukkos_optimizer.cpp 
              objectives/objective_function.cpp
              kernel/quantum_kernel.cpp
              utils/qukkos_utils.cpp
              objectives/gradient_function.cpp)

              #execution/taskInitiate.cpp
              #utils/qukkos_types.cpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_include_directories(${LIBRARY_NAME} PUBLIC . objectives qrt qrt/internal_compiler qrt/ir ${XACC_ROOT}/include/spdlog
                            observable 
                            optimizer 
                            jit
                            kernel 
                            objectives 
                            execution 
                            utils)

target_link_libraries(${LIBRARY_NAME} PUBLIC xacc::xacc xacc::quantum_gate qrt xacc::pauli xacc::fermion)

if(APPLE)
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES LINK_FLAGS "-undefined dynamic_lookup")
else()
  set_target_properties(${LIBRARY_NAME}
                        PROPERTIES INSTALL_RPATH "${XACC_ROOT}/lib")
  set_target_properties(${LIBRARY_NAME} PROPERTIES LINK_FLAGS "-shared")
endif()

file(GLOB HEADERS qukkos.hpp qukkos_lang_ext.hpp
                  observable/qukkos_observable.hpp 
                  optimizer/qukkos_optimizer.hpp 
                  kernel/quantum_kernel.hpp 
                  qrt/qrt.hpp qrt/ir/*.hpp 
                  qrt/internal_compiler/qalloc 
                  qrt/internal_compiler/*.hpp
                  objectives/objective_function.hpp 
                  objectives/gradient_function.hpp
                  execution/taskInitiate.hpp
                  utils/qukkos_utils.hpp
                  utils/qukkos_pimpl.hpp
                  jit/qukkos_jit.hpp)

install(FILES ${HEADERS} DESTINATION include/qukkos)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
set(EXTRA_INCLUDE "")
foreach(EXTRA_HEADER ${QUKKOS_EXTRA_HEADERS})
  set(EXTRA_INCLUDE "${EXTRA_INCLUDE} -I ${EXTRA_HEADER}")
endforeach()

if (EXTRA_INCLUDE)
  string(REGEX REPLACE "\"" "" EXTRA_INCLUDE ${EXTRA_INCLUDE})
  message(STATUS "Extra Includes for PCH-generation: ${EXTRA_INCLUDE}")
endif()

install(CODE "execute_process(COMMAND ${LLVM_INSTALL_PREFIX}/bin/clang++ -std=c++17 -x c++-header ${CMAKE_INSTALL_PREFIX}/include/qukkos/qukkos_lang_ext.hpp -o ${CMAKE_BINARY_DIR}/runtime/qukkos_lang_ext.hpp.pch -I ${XACC_ROOT}/include/xacc -I ${CMAKE_INSTALL_PREFIX}/include/qukkos -I ${XACC_ROOT}/include/eigen ${EXTRA_INCLUDE})")
install(FILES ${CMAKE_BINARY_DIR}/runtime/qukkos_lang_ext.hpp.pch DESTINATION include/qukkos)

if (QUKKOS_BUILD_TESTS)
  add_subdirectory(tests)
  add_subdirectory(observable/tests)
  add_subdirectory(optimizer/tests)
endif()

add_subdirectory(objectives)
add_subdirectory(qrt)
add_subdirectory(jit)
