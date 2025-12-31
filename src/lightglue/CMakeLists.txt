add_library(n4m_lightglue lightglue.cpp)

target_link_libraries(n4m_lightglue PUBLIC n4m_xfeat onnxruntime::onnxruntime spdlog::spdlog)
