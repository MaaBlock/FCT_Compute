cmake_minimum_required(VERSION 3.26)

include(platform/platform.cmake)
include(../../Config/set_utf8_compiler_encoding.cmake OPTIONAL)

if(MSVC AND MSVC_VERSION GREATER_EQUAL 1940 AND DEFINED ENV{CUDA_PATH_V13_0})
    file(TO_CMAKE_PATH "$ENV{CUDA_PATH_V13_0}" _cuda_v13_root)
    if(EXISTS "${_cuda_v13_root}/bin/nvcc.exe")
        set(CUDAToolkit_ROOT "${_cuda_v13_root}" CACHE PATH "CUDA Toolkit root directory" FORCE)
        set(CMAKE_CUDA_COMPILER "${_cuda_v13_root}/bin/nvcc.exe" CACHE FILEPATH "CUDA compiler" FORCE)
    endif()
endif()

set(_fct_compute_prev_build_shared_libs_defined FALSE)
if(DEFINED BUILD_SHARED_LIBS)
    set(_fct_compute_prev_build_shared_libs_defined TRUE)
    set(_fct_compute_prev_build_shared_libs "${BUILD_SHARED_LIBS}")
endif()

set(LUISA_COMPUTE_ENABLE_DSL ON CACHE BOOL "Enable C++ DSL" FORCE)
set(LUISA_COMPUTE_BUILD_TESTS OFF CACHE BOOL "Build tests for LuisaCompute" FORCE)
set(LUISA_COMPUTE_ENABLE_GUI OFF CACHE BOOL "Enable GUI support" FORCE)
set(LUISA_COMPUTE_ENABLE_FALLBACK OFF CACHE BOOL "Enable Fallback backend" FORCE)
set(LUISA_COMPUTE_ENABLE_CLANG_CXX OFF CACHE BOOL "Enable ClangTooling-based C++ shading language" FORCE)
set(LUISA_COMPUTE_ENABLE_HIP OFF CACHE BOOL "Enable HIP backend" FORCE)
set(LUISA_COMPUTE_USE_SYSTEM_STL ON CACHE BOOL "Use system STL" FORCE)
set(LUISA_COMPUTE_USE_SYSTEM_SPDLOG ON CACHE BOOL "Use system spdlog" FORCE)
set(LUISA_COMPUTE_CHECK_BACKEND_DEPENDENCIES ON CACHE BOOL "Check backend dependencies" FORCE)

if(NOT TARGET luisa::compute)
    add_subdirectory(ThirdParty/LuisaCompute EXCLUDE_FROM_ALL)
endif()

if(_fct_compute_prev_build_shared_libs_defined)
    set(BUILD_SHARED_LIBS "${_fct_compute_prev_build_shared_libs}" CACHE BOOL "" FORCE)
else()
    unset(BUILD_SHARED_LIBS CACHE)
    unset(BUILD_SHARED_LIBS)
endif()

file(GLOB_RECURSE FCT_COMPUTE_SOURCES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "./src/*.cpp"
)

file(GLOB_RECURSE FCT_COMPUTE_HEADERS
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    "./src/*.h"
    "./src/*.hpp"
)

add_library(FCT_Compute STATIC
    ${FCT_COMPUTE_SOURCES}
    ${FCT_COMPUTE_HEADERS}
)

if(COMMAND set_utf8_for_target)
    set_utf8_for_target(FCT_Compute)
endif()

set_target_properties(FCT_Compute PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/FCT_Compute)

foreach(header_file ${FCT_COMPUTE_HEADERS})
    file(RELATIVE_PATH relative_path ${CMAKE_CURRENT_SOURCE_DIR}/src ${header_file})
    get_filename_component(header_dir ${relative_path} DIRECTORY)

    if(header_dir)
        file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/FCT_Compute/${header_dir})
    endif()

    configure_file(${header_file} ${CMAKE_CURRENT_SOURCE_DIR}/include/FCT_Compute/${relative_path} COPYONLY)
endforeach()

target_include_directories(FCT_Compute PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../FCT/include/FCT>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../FCT/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(FCT_Compute PUBLIC
    FCT
    luisa::compute
)

function(target_link_fct_compute_libraries target_name)
    target_link_libraries(${target_name} FCT_Compute)
endfunction()
