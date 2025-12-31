#pragma once

#include <onnxruntime_cxx_api.h>
#include <spdlog/spdlog.h>

#include <cstdio>
#include <fcntl.h>
#include <string>
#include <thread>
#include <unistd.h>

namespace n4m
{
namespace detail
{

inline void ort_logging_function(void * /*param*/, OrtLoggingLevel severity, const char * /*category*/,
                                 const char * /*logid*/, const char * /*code_location*/, const char *message)
{
    switch (severity)
    {
    case ORT_LOGGING_LEVEL_VERBOSE:
        spdlog::trace("{}", message);
        break;
    case ORT_LOGGING_LEVEL_INFO:
        spdlog::info("{}", message);
        break;
    case ORT_LOGGING_LEVEL_WARNING:
        spdlog::warn("{}", message);
        break;
    case ORT_LOGGING_LEVEL_ERROR:
        spdlog::error("{}", message);
        break;
    case ORT_LOGGING_LEVEL_FATAL:
        spdlog::critical("{}", message);
        break;
    }
}

inline Ort::Env create_ort_env(const char *logid)
{
    return Ort::Env(ORT_LOGGING_LEVEL_WARNING, logid, ort_logging_function, nullptr);
}

// Redirects stderr to spdlog during the lifetime of this object.
// Used to capture ONNX schema registration messages that bypass ORT's logging.
class StderrToSpdlog
{
  public:
    StderrToSpdlog()
    {
        std::fflush(stderr);
        saved_fd_ = ::dup(STDERR_FILENO);
        if (saved_fd_ < 0)
        {
            return;
        }

        int pipefd[2];
        if (::pipe(pipefd) != 0)
        {
            ::close(saved_fd_);
            saved_fd_ = -1;
            return;
        }
        ::dup2(pipefd[1], STDERR_FILENO);
        ::close(pipefd[1]);

        reader_ = std::thread([read_fd = pipefd[0]]() {
            char buf[4096];
            std::string line;
            ssize_t n;
            while ((n = ::read(read_fd, buf, sizeof(buf))) > 0)
            {
                for (ssize_t i = 0; i < n; ++i)
                {
                    if (buf[i] == '\n')
                    {
                        if (!line.empty())
                        {
                            spdlog::debug("{}", line);
                        }
                        line.clear();
                    }
                    else
                    {
                        line += buf[i];
                    }
                }
            }
            if (!line.empty())
            {
                spdlog::debug("{}", line);
            }
            ::close(read_fd);
        });
    }

    ~StderrToSpdlog()
    {
        if (saved_fd_ < 0)
        {
            return;
        }
        std::fflush(stderr);
        ::dup2(saved_fd_, STDERR_FILENO);
        ::close(saved_fd_);
        reader_.join();
    }

    StderrToSpdlog(const StderrToSpdlog &) = delete;
    StderrToSpdlog &operator=(const StderrToSpdlog &) = delete;

  private:
    int saved_fd_ = -1;
    std::thread reader_;
};

inline Ort::Session create_ort_session(Ort::Env &env, const char *model_path, Ort::SessionOptions &opts)
{
    StderrToSpdlog redirect;
    return Ort::Session(env, model_path, opts);
}

} // namespace detail
} // namespace n4m
