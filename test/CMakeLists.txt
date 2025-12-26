find_package(GTest REQUIRED)

add_executable(nm_tests
    gtest_main.cpp
    test_types.cpp
    test_postprocess.cpp
    test_xfeat.cpp
    test_lightglue.cpp
)

target_include_directories(nm_tests PRIVATE ${CMAKE_SOURCE_DIR}/src/xfeat)

target_link_libraries(nm_tests PRIVATE nm_xfeat nm_lightglue GTest::gtest opencv_imgcodecs)

target_compile_definitions(nm_tests PRIVATE
    TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/test_data")

add_custom_target(test
                  USES_TERMINAL
                  COMMAND ${CMAKE_BINARY_DIR}/test/nm_tests
                  DEPENDS nm_tests)

if ("${CMAKE_BUILD_TYPE}" STREQUAL "Coverage")

    add_custom_target(clean_coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMAND ${CMAKE_SOURCE_DIR}/external/venv/bin/fastcov -z
        COMMAND rm -r coverage_html || echo "no existing coverage report"
        DEPENDS nm_tests)

    add_dependencies(test clean_coverage)

    add_custom_target(test_coverage
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
        COMMAND ${CMAKE_SOURCE_DIR}/external/venv/bin/fastcov -i `realpath ${CMAKE_SOURCE_DIR}` -e external/ -l -o clean_coverage.info
        COMMAND lcov --summary clean_coverage.info --ignore-errors mismatch,inconsistent
        DEPENDS test)

    add_custom_target(test_coverage_html
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
        COMMAND genhtml clean_coverage.info --output-directory coverage_html --branch-coverage --ignore-errors mismatch,inconsistent
        DEPENDS test_coverage)
endif()
