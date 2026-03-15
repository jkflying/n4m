#pragma once

#include <n4m/types.hpp>

#include <onnxruntime_cxx_api.h>

#include <string>
#include <vector>

namespace n4m
{

/// Returns the execution provider name used by ONNX Runtime for the given backend.
inline const char *backend_provider_name(Backend b)
{
    switch (b)
    {
    case Backend::cpu:
        return "CPUExecutionProvider";
    case Backend::cuda:
        return "CUDAExecutionProvider";
    case Backend::tensorrt:
        return "TensorrtExecutionProvider";
    case Backend::coreml:
        return "CoreMLExecutionProvider";
    case Backend::directml:
        return "DmlExecutionProvider";
    case Backend::rocm:
        return "ROCMExecutionProvider";
    case Backend::openvino:
        return "OpenVINOExecutionProvider";
    }
    return "";
}

/// Query which backends are available in the current ONNX Runtime build.
inline std::vector<Backend> available_backends()
{
    auto providers = Ort::GetAvailableProviders();

    // All backends in enum order, paired with their ORT provider name.
    static constexpr Backend all_backends[] = {
        Backend::cpu,      Backend::cuda, Backend::tensorrt, Backend::coreml,
        Backend::directml, Backend::rocm, Backend::openvino,
    };

    std::vector<Backend> result;
    for (Backend b : all_backends)
    {
        const char *name = backend_provider_name(b);
        for (const auto &p : providers)
        {
            if (p == name)
            {
                result.push_back(b);
                break;
            }
        }
    }
    return result;
}

} // namespace n4m
