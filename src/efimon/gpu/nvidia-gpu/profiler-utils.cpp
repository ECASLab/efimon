/**
 * @file profiler-utils.cpp
 * @brief Helpers shared by the modules of the NVIDIA process metrics library
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <pthread.h>

#include <cstdio>
#include <ctime>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>

namespace efimon {
namespace nvidia {

std::string EscapeCsv(const std::string& value) {
  std::string escaped = "\"";

  for (const char character : value) {
    if (character == '\"') escaped += '\"';
    escaped += character;
  }

  escaped += "\"";
  return escaped;
}

void PrintCudaError(const char* operation, CUresult result) {
  const char* name = "CUDA_ERROR_UNKNOWN";
  const char* message = "unknown CUDA driver error";
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &message);
  std::fprintf(stderr, "[CUDA] %s: %s (%s)\n", operation, name, message);
}

void PrintCuptiError(const char* operation, CUptiResult result) {
  const char* message = "unknown CUPTI error";
  cuptiGetResultString(result, &message);
  std::fprintf(stderr, "[CUPTI] %s: %s\n", operation, message);
}

bool FinishCsvFile(std::ofstream* file, const std::filesystem::path& path) {
  file->flush();
  if (!*file) {
    std::fprintf(stderr, "[Profiler] Cannot write CSV file %s\n", path.c_str());
    file->close();
    return false;
  }
  file->close();
  if (!*file) {
    std::fprintf(stderr, "[Profiler] Cannot close CSV file %s\n", path.c_str());
    return false;
  }
  return true;
}

double ThreadCpuMilliseconds() {
  timespec cpu_time{};
  if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &cpu_time) != 0) return 0.0;
  return cpu_time.tv_sec * 1e3 + cpu_time.tv_nsec / 1e6;
}

void NameProfilerThread(const char* name) {
  pthread_setname_np(pthread_self(), name);
}

} /* namespace nvidia */
} /* namespace efimon */
