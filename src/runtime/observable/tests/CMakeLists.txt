link_directories(${XACC_ROOT}/lib)
if (NOT XACC_ROOT STREQUAL CMAKE_INSTALL_PREFIX)
  add_definitions(-D__internal__qukkos__compile__plugin__path="${CMAKE_INSTALL_PREFIX}/plugins")
endif()

add_executable(OperatorTester OperatorTester.cpp)
add_test(NAME qukkos_OperatorTester COMMAND QUKKOSTester)
target_include_directories(OperatorTester PRIVATE ${XACC_ROOT}/include/gtest)
target_link_libraries(OperatorTester ${XACC_TEST_LIBRARIES} qukkos)

