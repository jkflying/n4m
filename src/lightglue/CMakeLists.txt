add_library(nm_lightglue lightglue.cpp)

target_link_libraries(nm_lightglue PUBLIC nm_xfeat onnxruntime::onnxruntime spdlog::spdlog)

if(NM_VULKAN)
    target_link_libraries(nm_lightglue PUBLIC Vulkan::Vulkan)
endif()
