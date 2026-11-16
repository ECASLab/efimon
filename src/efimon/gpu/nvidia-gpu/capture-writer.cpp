/**
 * @file capture-writer.cpp
 * @brief Files written when the profiled process exits: the sorted kernel and
 * memory copy CSVs and capture_summary.txt
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cupti.h>

#include <cstdint>
#include <cstdio>
#include <efimon/gpu/nvidia-gpu/capture-writer.hpp>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iomanip>
#include <string>

namespace efimon {
namespace nvidia {
namespace {

/**
 * @brief Gets the name of a CUPTI memory copy kind
 *
 * @param copy_kind CUpti_ActivityMemcpyKind value
 * @return short name, or "unknown"
 */
const char* MemcpyKindName(uint8_t copy_kind) {
  switch (copy_kind) {
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
      return "HtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
      return "DtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
      return "HtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
      return "AtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
      return "AtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
      return "AtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
      return "DtoA";
    case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
      return "DtoD";
    case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
      return "HtoH";
    case CUPTI_ACTIVITY_MEMCPY_KIND_PTOP:
      return "PtoP";
    default:
      return "unknown";
  }
}

/**
 * @brief Writes every kernel to a CSV in start order and adds it to windows
 *
 * @param file open kernel_activity.csv; must not be nullptr
 * @param pid profiled process
 * @param activity stopped collector; must not be nullptr
 * @param windows windows that receive every kernel; must not be nullptr
 * @param kernel_count receives the number of kernels; must not be nullptr
 * @return false if the journal cannot be read
 */
bool WriteSortedKernels(std::ofstream* file, pid_t pid,
                        ActivityCollector* activity, ActiveTimeWindows* windows,
                        uint64_t* kernel_count) {
  /* Rows are formatted with std::to_chars into one buffer and written in
     large blocks; the text is the same as with stream formatting. */
  constexpr size_t kCsvBlockBytes = 1 << 20;
  std::string rows;
  rows.reserve(kCsvBlockBytes + 512);
  uint64_t event_id = 0;
  auto write = [&](const KernelRecord& kernel) {
    AppendKernelCsvRow(event_id++, pid, kernel,
                       activity->escaped_name(kernel.name_id), &rows);
    windows->Add(kernel.start_ns, kernel.end_ns);
    if (rows.size() >= kCsvBlockBytes) {
      file->write(rows.data(), static_cast<std::streamsize>(rows.size()));
      rows.clear();
    }
  };

  const bool journal_read = activity->VisitKernelsSorted(write);
  file->write(rows.data(), static_cast<std::streamsize>(rows.size()));

  windows->Finish();
  *kernel_count = event_id;
  return journal_read;
}

} /* namespace */

bool WriteActivityCsvFiles(const std::filesystem::path& directory, pid_t pid,
                           bool memcpy_csv, ActivityCollector* activity,
                           ActiveTimeWindows* windows, uint64_t* kernel_count) {
  /* Header of kernel_activity.csv */
  constexpr char kKernelCsvHeader[] =
      "event_id,pid,device_id,context_id,stream_id,correlation_id,"
      "start_ns,end_ns,duration_ns,kernel_name\n";
  bool success = true;

  const auto kernels_path = directory / "kernel_activity.csv";
  std::ofstream kernels_file(kernels_path);
  if (!kernels_file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 kernels_path.c_str());
    success = false;
  } else {
    kernels_file << kKernelCsvHeader;
    if (!WriteSortedKernels(&kernels_file, pid, activity, windows,
                            kernel_count)) {
      success = false;
    }
    if (!FinishCsvFile(&kernels_file, kernels_path)) success = false;
  }

  if (memcpy_csv) {
    const auto memcpy_path = directory / "memcpy_activity.csv";
    std::ofstream memcpy_file(memcpy_path);
    if (!memcpy_file.is_open()) {
      std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                   memcpy_path.c_str());
      success = false;
    } else {
      memcpy_file << "event_id,pid,device_id,context_id,stream_id,"
                     "correlation_id,start_ns,end_ns,duration_ns,bytes,"
                     "copy_kind\n";
      const auto& memcpy_events = activity->memcpy_events();
      for (size_t index = 0; index < memcpy_events.size(); ++index) {
        const auto& copy = memcpy_events[index];
        memcpy_file << index << ',' << pid << ',' << copy.device_id << ','
                    << copy.context_id << ',' << copy.stream_id << ','
                    << copy.correlation_id << ',' << copy.start << ','
                    << copy.end << ',' << (copy.end - copy.start) << ','
                    << copy.bytes << ',' << MemcpyKindName(copy.copy_kind)
                    << '\n';
      }
      if (!FinishCsvFile(&memcpy_file, memcpy_path)) success = false;
    }
  }
  return success;
}

bool WriteCaptureSummary(const std::filesystem::path& path,
                         const CaptureSummary& summary) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open %s\n", path.c_str());
    return false;
  }

  file << "pid=" << summary.pid << '\n'
       << "kernels=" << summary.kernels << '\n'
       << "distinct_kernel_names=" << summary.distinct_kernel_names << '\n'
       << "memcpy_records=" << summary.memcpy_records << '\n'
       << "dropped_activity_records=" << summary.dropped_activity_records
       << '\n'
       << "invalid_activity_records=" << summary.invalid_activity_records
       << '\n'
       << "other_device_activity_records="
       << summary.other_device_activity_records << '\n'
       << "late_activity_records=" << summary.late_activity_records << '\n'
       << "activity_buffers_beyond_pool="
       << summary.activity_buffers_beyond_pool << '\n'
       << "telemetry_rows=" << summary.telemetry_rows << '\n'
       << "power_sample_rows=" << summary.power_sample_rows << '\n'
       << "kernel_journal=" << summary.kernel_journal << '\n'
       << std::fixed << std::setprecision(3)
       << "telemetry_thread_cpu_ms=" << summary.telemetry_thread_cpu_ms << '\n'
       << "activity_thread_cpu_ms=" << summary.activity_thread_cpu_ms << '\n'
       << "exit_write_cpu_ms=" << summary.exit_write_cpu_ms << '\n'
       << "exit_write_wall_ms=" << summary.exit_write_wall_ms << '\n'
       << "tail_wait_ms=" << summary.tail_wait_ms << '\n'
       << "profiler_failed=" << (summary.profiler_failed ? 1 : 0) << '\n'
       << "capture_status=" << summary.capture_status << '\n';
  return FinishCsvFile(&file, path);
}

} /* namespace nvidia */
} /* namespace efimon */
