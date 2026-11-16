/**
 * @file profiler-options.hpp
 * @brief Settings of the NVIDIA process metrics library, read from the
 * NVIDIA_METRICS_* environment variables
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_PROFILER_OPTIONS_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_PROFILER_OPTIONS_HPP_

#include <chrono>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT

namespace efimon {
namespace nvidia {

/** Kernel window used when NVIDIA_METRICS_WINDOW_MS is unset or invalid */
constexpr int kDefaultMetricsWindowMs = 200;

/**
 * @brief Runtime switches read from the NVIDIA_METRICS_* variables
 *
 * Every behavior added by the profiler audit can be disabled to compare
 * against the previous one.
 */
struct ProfilerOptions {
  /** Profile child processes too; each writes under children/<pid> */
  bool follow_children = false;
  /** Parse CUPTI buffers in a dedicated thread instead of the CUPTI
      callback */
  bool activity_worker = true;
  /** Preallocated CUPTI activity buffers; 0 allocates one per request */
  size_t activity_pool_buffers = 20;
  /** CUPTI periodic flush in milliseconds; 0 keeps the CUPTI heuristics */
  uint32_t activity_flush_ms = 1000;
  /** Append kernels to the binary kernel_activity.journal instead of keeping
      them in memory until exit */
  bool kernel_journal = true;
  /** Write individual memory copies to memcpy_activity.csv */
  bool memcpy_csv = true;
  /** Read the NVML energy counter next to every power sample */
  bool nvml_fields = true;
  /** Bracket only the power and energy reads with the query timestamps */
  bool power_bracket = true;
  /** Write the power samples that the driver buffers to
      gpu_power_samples.csv */
  bool power_samples = true;
  /** Schedule telemetry reads every 50 ms from a fixed start instead of
      sleeping 50 ms after each read, which drifts by the read time */
  bool fixed_period = true;
  /** Period of the reads that need not run every 50 ms: the energy counter,
      temperature, utilization and the driver's power samples. 0 reads them
      at every telemetry row */
  uint32_t slow_read_ms = 200;
  /** Keep kernels and memory copies of every device, not only of the device
      whose power NVML reads */
  bool all_devices = false;
};

/**
 * @brief Reads the runtime switches from the environment
 *
 * Unset or invalid variables keep their defaults; invalid ones are reported
 * on stderr.
 *
 * @return options of this process
 */
ProfilerOptions LoadProfilerOptions();

/**
 * @brief Decides whether this process profiles and where it writes
 *
 * The first process that loads the library for an output directory owns that
 * session: it records its PID and directory in NVIDIA_METRICS_OWNER_PID and
 * NVIDIA_METRICS_OWNER_OUTPUT_DIR, and exports the absolute directory so that
 * descendants inherit the same session. A wrapper that execs the target keeps
 * the PID and stays the owner. Other processes of the session inherit
 * LD_PRELOAD but stay idle, so that they never overwrite the owner's CSVs,
 * unless follow_children sends them to children/<pid>. A process that asks
 * for a different output directory starts its own session.
 *
 * @param follow_children whether child processes profile too
 * @param output_directory receives the absolute output directory; must not
 * be nullptr
 * @return false when this process must not profile
 */
bool ConfigureProcessScope(bool follow_children,
                           std::filesystem::path* output_directory);

/**
 * @brief Reads the CUDA device from NVIDIA_METRICS_DEVICE
 *
 * @param device_index receives the device, 0 when the variable is unset; must
 * not be nullptr
 * @return false if the variable is invalid
 */
bool GetConfiguredDeviceIndex(int* device_index);

/**
 * @brief Reads the kernel window from NVIDIA_METRICS_WINDOW_MS
 *
 * @param window_duration receives the window, kDefaultMetricsWindowMs when
 * the variable is unset or invalid; must not be nullptr
 * @return false if the variable is invalid
 */
bool GetConfiguredWindowDuration(std::chrono::milliseconds* window_duration);

/**
 * @brief Reads the telemetry kept after the last kernel from
 * NVIDIA_METRICS_TAIL_MS
 *
 * The tail gives the last kernel window real power samples.
 *
 * @param window_duration kernel window, used when the variable is unset or
 * invalid
 * @return tail duration, at most 60 s
 */
std::chrono::milliseconds GetConfiguredTailDuration(
    std::chrono::milliseconds window_duration);

} /* namespace nvidia */
} /* namespace efimon */

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_PROFILER_OPTIONS_HPP_
