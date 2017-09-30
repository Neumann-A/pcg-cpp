cmake_minimum_required (VERSION 3.9 FATAL_ERROR)

project(pcg-cpp CXX)

set(HEADERS "include/pcg_extras.hpp;include/pcg_random.hpp;include/pcg_uint128.hpp")

SET(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(PCG_MAJOR_VERSION "1")
set(PCG_MINOR_VERSION "1")
set(PCG_PATCH_VERSION "0")
set(PCG_VERSION_STRING "${PCG_MAJOR_VERSION}.${PCG_MINOR_VERSION}.${PCG_PATCH_VERSION}")

add_library(${PROJECT_NAME} INTERFACE)

set_target_properties (${PROJECT_NAME} PROPERTIES FOLDER pcg)

target_compile_features(${PROJECT_NAME} PRIVATE cxx_constexpr)

#Export as Package

export(PACKAGE ${PROJECT_NAME})
CONFIGURE_FILE(${PROJECT_NAME}Config.cmake.in
    ${PROJECT_NAME}Config.cmake
    @ONLY)
CONFIGURE_FILE(${PROJECT_NAME}ConfigVersion.cmake.in
    ${PROJECT_NAME}ConfigVersion.cmake
    @ONLY)
INSTALL(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    DESTINATION "${CMAKE_INSTALL_DIR}"
    COMPONENT dev)