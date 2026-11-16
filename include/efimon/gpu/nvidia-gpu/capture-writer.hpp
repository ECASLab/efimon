/**
 * @file capture-writer.hpp
 * @brief Files written when the profiled process exits: the sorted kernel and
 * memory copy CSVs and capture_summary.txt
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_CAPTURE_WRITER_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_CAPTURE_WRITER_HPP_

#include <sys/types.h>

#include <cstddef>
#include <cstdint>
#include <efimon/gpu/nvidia-gpu/activity-collector.hpp>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <filesystem>  // NOLINT

namespace efimon {
namespace nvidia {

/**
 * @brief Counters of capture_summary.txt
 *
 * They tell whether a capture is complete, plus the CPU time of the
 * profiler's own threads. CUPTI's internal threads are not included.
 */
struct CaptureSummary {
  /** Profiled process */
  pid_t pid = 0;
  /** Kernels written to kernel_activity.csv */
  uint64_t kernels = 0;
  /** Distinct kernel names */
  size_t distinct_kernel_names = 0;
  /** Memory copies kept */
  size_t memcpy_records = 0;
  /** Records that CUPTI dropped */
  size_t dropped_activity_records = 0;
  /** Incomplete records left out */
  size_t invalid_activity_records = 0;
  /** Records of other devices left out */
  size_t other_device_activity_records = 0;
  /** Records that arrived after the collection closed */
  size_t late_activity_records = 0;
  /** Buffers allocated because the pool was empty */
  size_t activity_buffers_beyond_pool = 0;
  /** Rows of gpu_telemetry.csv */
  uint64_t telemetry_rows = 0;
  /** Rows of gpu_power_samples.csv */
  uint64_t power_sample_rows = 0;
  /** "failed", "used" or "disabled" */
  const char* kernel_journal = "disabled";
  /** CPU time of the telemetry thread */
  double telemetry_thread_cpu_ms = 0.0;
  /** CPU time of the activity worker */
  double activity_thread_cpu_ms = 0.0;
  /** CPU time spent writing the activity CSVs */
  double exit_write_cpu_ms = 0.0;
  /** Wall time spent writing the activity CSVs */
  double exit_write_wall_ms = 0.0;
  /** Wall time of the telemetry kept after the last kernel */
  double tail_wait_ms = 0.0;
  /** Whether the profiler failed */
  bool profiler_failed = false;
  /** "failed", "partial" or "complete" */
  const char* capture_status = "complete";
};

/**
 * @brief Writes kernel_activity.csv and, if enabled, memcpy_activity.csv
 *
 * Kernels are written in start order and added to the windows. Errors are
 * reported on stderr.
 *
 * @param directory output directory
 * @param pid profiled process
 * @param memcpy_csv whether memcpy_activity.csv is written
 * @param activity stopped collector; must not be nullptr
 * @param windows windows that receive every kernel; must not be nullptr
 * @param kernel_count receives the number of kernels; must not be nullptr
 * @return false if a CSV or the journal cannot be read or written
 */
bool WriteActivityCsvFiles(const std::filesystem::path& directory, pid_t pid,
                           bool memcpy_csv, ActivityCollector* activity,
                           ActiveTimeWindows* windows, uint64_t* kernel_count);

/**
 * @brief Writes capture_summary.txt, one key=value per line
 *
 * @param path file to write
 * @param summary counters of the capture
 * @return false if the file cannot be opened or written
 */
bool WriteCaptureSummary(const std::filesystem::path& path,
                         const CaptureSummary& summary);

} /* namespace nvidia */
} /* namespace efimon */

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_CAPTURE_WRITER_HPP_
