set(LIBRARY_NAME qukkos-quasimo)

file(GLOB SRC *.cpp)
file(GLOB HEADERS *.hpp)

add_library(${LIBRARY_NAME} SHARED ${SRC})

target_link_libraries(${LIBRARY_NAME} PUBLIC qukkos)

xacc_configure_library_rpath(${LIBRARY_NAME})

install(FILES ${HEADERS} DESTINATION include/qukkos)
install(TARGETS ${LIBRARY_NAME} DESTINATION lib)
