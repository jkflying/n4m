add_library(nm_xfeat xfeat.cpp)

target_link_libraries(nm_xfeat PUBLIC opencv_core ncnn spdlog::spdlog)

if(NM_VULKAN)
    target_link_libraries(nm_xfeat PUBLIC Vulkan::Vulkan)
endif()
