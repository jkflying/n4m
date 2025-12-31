add_library(n4m_xfeat xfeat.cpp)

target_link_libraries(n4m_xfeat PUBLIC opencv_core opencv_imgproc onnxruntime::onnxruntime spdlog::spdlog)
