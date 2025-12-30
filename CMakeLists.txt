cmake_minimum_required(VERSION 3.13)
project(nnmatch)

#============= OPTIONS ===============
if(NOT DEFINED NM_TESTING)
    option(NM_TESTING "" ON)
endif()

if(NOT DEFINED NM_APP)
    option(NM_APP "" ON)
endif()

option(NM_CLANG_TIDY "Run clang-tidy during compilation" OFF)
option(NM_VULKAN "Enable Vulkan GPU acceleration" OFF)

# =====================================

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ======= DEPENDENCIES ================
find_package(OpenCV REQUIRED)
include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})

find_package(onnxruntime REQUIRED)

find_package(spdlog REQUIRED)

if(NM_VULKAN)
    find_package(Vulkan REQUIRED)
    add_compile_definitions(NM_VULKAN_ENABLED)
endif()
# =======================================

if(MSVC)
    set(CMAKE_CXX_FLAGS " /W4 /WX")
else()
    set(CMAKE_CXX_FLAGS " -Wall -Wextra -pedantic -Werror -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_ASAN "-fsanitize=address -g -O2 -DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_ASAN "-fsanitize=address -g")
    set(CMAKE_CXX_FLAGS_COVERAGE "--coverage -fprofile-arcs -ftest-coverage -Og -g -fprofile-abs-path -fprofile-update=atomic"
                    CACHE STRING "Flags used by the C++ compiler during coverage builds" FORCE)
    set(CMAKE_EXE_LINKER_FLAGS_COVERAGE "--coverage -ftest-coverage -lgcov"
                        CACHE STRING "Flags used for linking binaries during coverage builds" FORCE)
endif()

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

include_directories(include)

if(NM_CLANG_TIDY)
    find_program(CLANG_TIDY_EXE
        NAMES clang-tidy-18 clang-tidy-17 clang-tidy-16 clang-tidy-15 clang-tidy-14 clang-tidy
        REQUIRED)
    message(STATUS "clang-tidy: ${CLANG_TIDY_EXE}")
    set(CMAKE_CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
endif()

add_subdirectory(src)

message("-- NM_TESTING: ${NM_TESTING}")
message("-- NM_APP: ${NM_APP}")

if (NM_APP)
    add_subdirectory(app)
endif()

if (NM_TESTING)
    add_subdirectory(test)
endif()

# ======= INSTALL =======================
include(GNUInstallDirs)

install(TARGETS nm_xfeat nm_lightglue
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(DIRECTORY include/nnmatch
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

if (NM_APP)
    install(TARGETS match_images
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

set(_required_models
    models/xfeat.onnx
    models/lightglue.onnx)

foreach(_model ${_required_models})
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${_model}")
        message(FATAL_ERROR "Required model not found: ${_model}\n"
            "Run the export scripts first:\n"
            "  python tools/export/1_export_xfeat_onnx.py\n"
            "  python tools/export/4_export_lightglue_onnx.py")
    endif()
endforeach()

install(FILES
    models/xfeat.onnx
    models/lightglue.onnx
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nnmatch/models)

# External data files (large models split by ONNX exporter)
install(FILES
    models/xfeat.onnx.data
    models/lightglue.onnx.data
    DESTINATION ${CMAKE_INSTALL_DATADIR}/nnmatch/models
    OPTIONAL)

# ======= CPACK ========================
set(CPACK_PACKAGE_NAME "nnmatch")
set(CPACK_PACKAGE_VERSION "0.1.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "XFeat + LightGlue feature matching")
set(CPACK_GENERATOR "TGZ")
include(CPack)
