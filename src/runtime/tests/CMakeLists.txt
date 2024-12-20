link_directories(${XACC_ROOT}/lib)
if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qukkos__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(QUKKOSTester QUKKOSTester.cpp)
add_test(NAME qukkos_QUKKOSTester COMMAND QUKKOSTester)
target_include_directories(QUKKOSTester PRIVATE ${XACC_ROOT}/include/gtest)
target_link_libraries(QUKKOSTester ${XACC_TEST_LIBRARIES} qukkos)


configure_file(InternalCompilerTester.in.cpp
               ${CMAKE_BINARY_DIR}/runtime/tests/InternalCompilerTester.cpp)
add_executable(InternalCompilerTester ${CMAKE_BINARY_DIR}/runtime/tests/InternalCompilerTester.cpp)
target_include_directories(InternalCompilerTester PRIVATE ${XACC_ROOT}/include/gtest ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler)
target_link_libraries(InternalCompilerTester PRIVATE xacc qrt ${XACC_TEST_LIBRARIES})
add_test(NAME qukkos_InternalCompilerTester COMMAND InternalCompilerTester)
target_compile_features(InternalCompilerTester PRIVATE cxx_std_17)


add_executable(QuilCompilerQallocTester QuilCompilerQallocTester.cpp)
add_test(NAME qukkos_QuilCompilerQallocTester COMMAND QuilCompilerQallocTester)
target_include_directories(QuilCompilerQallocTester PRIVATE ${XACC_ROOT}/include/gtest ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler)
target_link_libraries(QuilCompilerQallocTester ${XACC_TEST_LIBRARIES} qukkos qrt ${XACC_TEST_LIBRARIES})

add_executable(XASMCompilerQallocTester XASMCompilerQallocTester.cpp)
add_test(NAME qukkos_XASMCompilerQallocTester COMMAND XASMCompilerQallocTester)
target_include_directories(XASMCompilerQallocTester PRIVATE ${XACC_ROOT}/include/gtest ${CMAKE_SOURCE_DIR}/runtime/qrt/internal_compiler)
target_link_libraries(XASMCompilerQallocTester ${XACC_TEST_LIBRARIES} qukkos qrt ${XACC_TEST_LIBRARIES})