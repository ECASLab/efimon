/**
 * @file profiler-options.cpp
 * @brief Settings of the NVIDIA process metrics library, read from the
 * NVIDIA_METRICS_* environment variables
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <unistd.h>

#include <cerrno>
#include <chrono>  // NOLINT
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <efimon/gpu/nvidia-gpu/profiler-options.hpp>
#include <filesystem>  // NOLINT
#include <limits>
#include <string>
#include <system_error>  // NOLINT

namespace efimon {
namespace nvidia {
namespace {

/** CUPTI holds several buffers at once; on a V100 at 45,000 to 90,000
    kernels per second it used up to 18 */
constexpr size_t kDefaultActivityPoolBuffers = 20;
/** Upper bound of NVIDIA_METRICS_ACTIVITY_POOL_BUFFERS */
constexpr size_t kMaxActivityPoolBuffers = 256;
/** Default of NVIDIA_METRICS_ACTIVITY_FLUSH_MS */
constexpr uint32_t kDefaultActivityFlushMs = 1000;
/** Default of NVIDIA_METRICS_SLOW_READ_MS */
constexpr uint32_t kDefaultSlowReadMs = 200;
/** Upper bound of NVIDIA_METRICS_SLOW_READ_MS */
constexpr uint32_t kMaxSlowReadMs = 10000;
/** Upper bound of NVIDIA_METRICS_TAIL_MS */
constexpr uint64_t kMaxTailMs = 60000;

/**
 * @brief Reads a 0/1 switch
 *
 * @param name environment variable
 * @param fallback value when the variable is unset or invalid
 * @return switch value
 */
bool GetEnvSwitch(const char* name, bool fallback) {
  const char* configured = std::getenv(name);
  if (configured == nullptr || configured[0] == '\0') return fallback;
  if (std::strcmp(configured, "0") == 0) return false;
  if (std::strcmp(configured, "1") == 0) return true;

  std::fprintf(stderr, "[Profiler] Invalid %s: %s; using %d\n", name,
               configured, fallback ? 1 : 0);
  return fallback;
}

/**
 * @brief Reads a non-negative integer
 *
 * @param name environment variable
 * @param fallback value when the variable is unset or invalid
 * @param max_value largest valid value
 * @return integer value
 */
uint64_t GetEnvInteger(const char* name, uint64_t fallback,
                       uint64_t max_value) {
  const char* configured = std::getenv(name);
  if (configured == nullptr || configured[0] == '\0') return fallback;

  char* end = nullptr;
  errno = 0;
  const uint64_t parsed = std::strtoull(configured, &end, 10);
  if (configured[0] == '-' || errno != 0 || end == configured || *end != '\0' ||
      parsed > max_value) {
    std::fprintf(stderr, "[Profiler] Invalid %s: %s; using %" PRIu64 "\n", name,
                 configured, fallback);
    return fallback;
  }
  return parsed;
}

/**
 * @brief Gets the absolute, normalized form of a directory, which identifies
 * a profiling session
 *
 * @param directory output directory
 * @return absolute directory, or the normalized input if it cannot be made
 * absolute
 */
std::string SessionDirectory(const std::filesystem::path& directory) {
  std::error_code error;
  const std::filesystem::path absolute =
      std::filesystem::absolute(directory, error);
  return (error ? directory : absolute).lexically_normal().string();
}

} /* namespace */

ProfilerOptions LoadProfilerOptions() {
  ProfilerOptions options;
  options.follow_children =
      GetEnvSwitch("NVIDIA_METRICS_FOLLOW_CHILDREN", false);
  options.activity_worker =
      GetEnvSwitch("NVIDIA_METRICS_ACTIVITY_WORKER", true);
  options.activity_pool_buffers = static_cast<size_t>(
      GetEnvInteger("NVIDIA_METRICS_ACTIVITY_POOL_BUFFERS",
                    kDefaultActivityPoolBuffers, kMaxActivityPoolBuffers));
  options.activity_flush_ms = static_cast<uint32_t>(
      GetEnvInteger("NVIDIA_METRICS_ACTIVITY_FLUSH_MS", kDefaultActivityFlushMs,
                    std::numeric_limits<uint32_t>::max()));
  options.kernel_journal = GetEnvSwitch("NVIDIA_METRICS_KERNEL_JOURNAL", true);
  options.memcpy_csv = GetEnvSwitch("NVIDIA_METRICS_MEMCPY_CSV", true);
  options.nvml_fields = GetEnvSwitch("NVIDIA_METRICS_NVML_FIELDS", true);
  options.power_bracket = GetEnvSwitch("NVIDIA_METRICS_POWER_BRACKET", true);
  options.power_samples = GetEnvSwitch("NVIDIA_METRICS_POWER_SAMPLES", true);
  options.fixed_period = GetEnvSwitch("NVIDIA_METRICS_FIXED_PERIOD", true);
  options.slow_read_ms = static_cast<uint32_t>(GetEnvInteger(
      "NVIDIA_METRICS_SLOW_READ_MS", kDefaultSlowReadMs, kMaxSlowReadMs));
  options.all_devices = GetEnvSwitch("NVIDIA_METRICS_ALL_DEVICES", false);
  return options;
}

bool ConfigureProcessScope(bool follow_children,
                           std::filesystem::path* output_directory) {
  const char* configured = std::getenv("NVIDIA_METRICS_OUTPUT_DIR");
  if (configured != nullptr && configured[0] != '\0') {
    *output_directory = configured;
  }

  const std::string own_pid = std::to_string(getpid());
  const std::string session_directory = SessionDirectory(*output_directory);
  /* Files opened at start and files written at exit must land in the same
     directory even if the process changes its working directory. */
  *output_directory = session_directory;
  const char* owner = std::getenv("NVIDIA_METRICS_OWNER_PID");
  const char* owner_directory = std::getenv("NVIDIA_METRICS_OWNER_OUTPUT_DIR");
  if (owner == nullptr || owner[0] == '\0' || owner_directory == nullptr ||
      session_directory != owner_directory) {
    setenv("NVIDIA_METRICS_OWNER_PID", own_pid.c_str(), 1);
    setenv("NVIDIA_METRICS_OWNER_OUTPUT_DIR", session_directory.c_str(), 1);
    setenv("NVIDIA_METRICS_OUTPUT_DIR", session_directory.c_str(), 1);
    return true;
  }
  if (own_pid == owner) return true;

  if (!follow_children) return false;
  *output_directory = *output_directory / "children" / own_pid;
  return true;
}

bool GetConfiguredDeviceIndex(int* device_index) {
  const char* configured = std::getenv("NVIDIA_METRICS_DEVICE");

  if (configured == nullptr || configured[0] == '\0') {
    *device_index = 0;
    return true;
  }

  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(configured, &end, 10);

  if (errno != 0 || end == configured || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "[Profiler] Invalid NVIDIA_METRICS_DEVICE: %s\n",
                 configured);
    return false;
  }

  *device_index = static_cast<int>(parsed);
  return true;
}

bool GetConfiguredWindowDuration(std::chrono::milliseconds* window_duration) {
  const char* configured = std::getenv("NVIDIA_METRICS_WINDOW_MS");

  if (configured == nullptr || configured[0] == '\0') {
    *window_duration = std::chrono::milliseconds(kDefaultMetricsWindowMs);
    return true;
  }

  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(configured, &end, 10);

  if (errno != 0 || end == configured || *end != '\0' || parsed <= 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(
        stderr,
        "[Profiler] Invalid NVIDIA_METRICS_WINDOW_MS: %s; using %d ms\n",
        configured, kDefaultMetricsWindowMs);
    *window_duration = std::chrono::milliseconds(kDefaultMetricsWindowMs);
    return false;
  }

  *window_duration = std::chrono::milliseconds(parsed);
  return true;
}

std::chrono::milliseconds GetConfiguredTailDuration(
    std::chrono::milliseconds window_duration) {
  return std::chrono::milliseconds(GetEnvInteger(
      "NVIDIA_METRICS_TAIL_MS", static_cast<uint64_t>(window_duration.count()),
      kMaxTailMs));
}

} /* namespace nvidia */
} /* namespace efimon */
