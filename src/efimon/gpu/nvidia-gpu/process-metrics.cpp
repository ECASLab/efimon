/**
 * @file process-metrics.cpp
 * @brief Preloaded collector for NVML telemetry and CUPTI kernel activity
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cuda.h>
#include <cupti.h>
#include <fcntl.h>
#include <pthread.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <efimon/gpu/nvidia-gpu/activity-collector.hpp>
#include <efimon/gpu/nvidia-gpu/capture-writer.hpp>
#include <efimon/gpu/nvidia-gpu/gpu-telemetry.hpp>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-options.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <exception>
#include <filesystem>  // NOLINT
#include <fstream>
#include <limits>
#include <string>
#include <thread>
#include <vector>

/* Kernel names are interned: name_id indexes kernel_names. */
using KernelEvent = efimon::nvidia::KernelRecord;

using efimon::nvidia::ConfigureProcessScope;
using efimon::nvidia::EscapeCsv;
using efimon::nvidia::FinishCsvFile;
using efimon::nvidia::GetConfiguredDeviceIndex;
using efimon::nvidia::GetConfiguredTailDuration;
using efimon::nvidia::GetConfiguredWindowDuration;
using efimon::nvidia::kDefaultMetricsWindowMs;
using efimon::nvidia::LoadProfilerOptions;
using efimon::nvidia::NameProfilerThread;
using efimon::nvidia::PrintCudaError;
using efimon::nvidia::PrintCuptiError;
using efimon::nvidia::ProfilerOptions;
using efimon::nvidia::ThreadCpuMilliseconds;

/* Profiler state is global because the library has no entry point other than
   its load-time constructor and exit-time destructor. */

/* CUPTI kernel and memory copy activity of the profiled process. */
static efimon::nvidia::ActivityCollector activity;
/* NVML device, sampling thread and telemetry CSVs of the measured GPU. */
static efimon::nvidia::GpuTelemetry telemetry;
static pid_t profiler_owner_pid = -1;
/* False in processes that only inherited LD_PRELOAD; see ConfigureProcessScope.
 */
static bool profiler_active = false;
static int profiler_completion_fd = -1;
static std::atomic<bool> profiler_failed{false};

static ProfilerOptions options;
static std::filesystem::path output_directory(".");
static bool output_directory_ready = false;

constexpr char PROFILER_COMPLETION_SUCCESS = '0';
constexpr char PROFILER_COMPLETION_FAILURE = '1';
/* The CSVs were written but lack data; see CaptureIsPartial(). */
constexpr char PROFILER_COMPLETION_PARTIAL = '2';

static std::chrono::milliseconds metrics_window_duration(
    kDefaultMetricsWindowMs);

/* Telemetry kept after the last kernel at exit, so that the last kernel window
   has real power samples. NVIDIA_METRICS_TAIL_MS sets it; one kernel window by
   default. */
static std::chrono::milliseconds tail_duration(0);
static double tail_wait_ms = 0.0;

static void ProfilerStop();

static void InitializeCompletionChannel() {
  profiler_owner_pid = getpid();
  const char* configured = std::getenv("NVIDIA_METRICS_COMPLETION_FD");
  if (configured == nullptr || configured[0] == '\0') return;

  char* end = nullptr;
  errno = 0;
  const int64_t parsed = std::strtoll(configured, &end, 10);
  unsetenv("NVIDIA_METRICS_COMPLETION_FD");
  if (errno != 0 || end == configured || *end != '\0' || parsed < 0 ||
      parsed > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "[Profiler] Invalid completion descriptor: %s\n",
                 configured);
    return;
  }

  const int descriptor = static_cast<int>(parsed);
  const int flags = fcntl(descriptor, F_GETFD);
  if (flags < 0 || fcntl(descriptor, F_SETFD, flags | FD_CLOEXEC) != 0) {
    std::fprintf(stderr,
                 "[Profiler] Cannot configure completion descriptor: %s\n",
                 std::strerror(errno));
    return;
  }
  profiler_completion_fd = descriptor;
}

/* Returns whether the capture misses data although every file was written:
   no NVML telemetry, or kernels and memory copies that CUPTI dropped, did not
   complete or returned after the CSVs started. */
static bool CaptureIsPartial() {
  return !telemetry.initialized() || activity.partial();
}

/* Returns whether the capture failed, in the profiler or in the activity
   collection. */
static bool ProfilerFailed() {
  return profiler_failed.load(std::memory_order_acquire) || activity.failed();
}

static const char* CaptureStatus() {
  if (ProfilerFailed()) return "failed";
  return CaptureIsPartial() ? "partial" : "complete";
}

static void ReportCompletion() {
  if (profiler_completion_fd < 0 || getpid() != profiler_owner_pid) return;

  char status = PROFILER_COMPLETION_SUCCESS;
  if (ProfilerFailed()) {
    status = PROFILER_COMPLETION_FAILURE;
  } else if (CaptureIsPartial()) {
    status = PROFILER_COMPLETION_PARTIAL;
  }
  ssize_t written = -1;
  do {
    written = write(profiler_completion_fd, &status, sizeof(status));
  } while (written < 0 && errno == EINTR);
  if (written != sizeof(status)) {
    std::fprintf(stderr, "[Profiler] Cannot report completion: %s\n",
                 written < 0 ? std::strerror(errno) : "short write");
  }
  close(profiler_completion_fd);
  profiler_completion_fd = -1;
}

static bool GetCudaPciBusId(int device_index, std::string* pci_bus_id) {
  CUresult result = cuInit(0);
  if (result != CUDA_SUCCESS) {
    PrintCudaError("Cannot initialize the CUDA driver", result);
    return false;
  }

  CUdevice cuda_device;
  result = cuDeviceGet(&cuda_device, device_index);
  if (result != CUDA_SUCCESS) {
    PrintCudaError("Cannot select the configured CUDA device", result);
    return false;
  }

  char identifier[32]{};
  result = cuDeviceGetPCIBusId(identifier, sizeof(identifier), cuda_device);
  if (result != CUDA_SUCCESS) {
    PrintCudaError("Cannot resolve the CUDA device PCI bus ID", result);
    return false;
  }

  *pci_bus_id = identifier;
  return true;
}

/* Creates the output directory and opens the files that are written while
   the profiled process runs. */
static bool PrepareOutputFiles() {
  std::error_code error;
  std::filesystem::create_directories(output_directory, error);
  if (error) {
    std::fprintf(stderr, "[Profiler] Cannot create output directory %s: %s\n",
                 output_directory.c_str(), error.message().c_str());
    return false;
  }
  output_directory_ready = true;

  if (!telemetry.OpenTelemetryFile(output_directory)) return false;

  activity.OpenJournal(profiler_owner_pid);

  telemetry.OpenPowerSamplesFile();
  return true;
}

/* Writes the sorted kernel and memory copy CSVs that the analyzers read. */
static bool SaveActivityCsvFiles(efimon::nvidia::ActiveTimeWindows* windows,
                                 uint64_t* kernel_count) {
  if (!output_directory_ready) return false;

  const bool success = efimon::nvidia::WriteActivityCsvFiles(
      output_directory, profiler_owner_pid, options.memcpy_csv, &activity,
      windows, kernel_count);
  if (success) {
    std::printf("Timeline CSV: %s\nKernel CSV: %s\n",
                telemetry.telemetry_path().c_str(),
                (output_directory / "kernel_activity.csv").c_str());
  }
  return success;
}

/* ------------------------------------------------ */
/* Library loaded                                   */
/* ------------------------------------------------ */

static void ProfilerStart() {
  options = LoadProfilerOptions();
  if (!ConfigureProcessScope(options.follow_children, &output_directory)) {
    return;
  }
  profiler_active = true;
  InitializeCompletionChannel();

  std::printf("\n=== NVIDIA Process Profiler ===\n");

  std::printf("PID: %d\n", getpid());

  GetConfiguredWindowDuration(&metrics_window_duration);
  tail_duration = GetConfiguredTailDuration(metrics_window_duration);
  std::printf("Kernel window: %" PRId64 " ms; final telemetry: %" PRId64
              " ms\n",
              static_cast<int64_t>(metrics_window_duration.count()),
              static_cast<int64_t>(tail_duration.count()));
  std::printf(
      "Options: follow_children=%d activity_worker=%d pool_buffers=%zu "
      "flush_ms=%" PRIu32
      " kernel_journal=%d memcpy_csv=%d nvml_fields=%d "
      "power_bracket=%d power_samples=%d fixed_period=%d "
      "slow_read_ms=%" PRIu32 " all_devices=%d\n",
      options.follow_children, options.activity_worker,
      options.activity_pool_buffers, options.activity_flush_ms,
      options.kernel_journal, options.memcpy_csv, options.nvml_fields,
      options.power_bracket, options.power_samples, options.fixed_period,
      options.slow_read_ms, options.all_devices);

  telemetry.Configure(options);
  activity.Configure(options, output_directory);
  if (!PrepareOutputFiles()) profiler_failed = true;

  /* NVML */

  int device_index = 0;

  if (GetConfiguredDeviceIndex(&device_index)) {
    std::string pci_bus_id;
    if (GetCudaPciBusId(device_index, &pci_bus_id)) {
      if (!options.all_devices) activity.set_measured_device(device_index);
      telemetry.InitializeDevice(device_index, pci_bus_id);
    }
  }

  /* CUPTI */

  switch (activity.Start()) {
    case efimon::nvidia::ActivityCollector::StartResult::kStarted:
      break;
    case efimon::nvidia::ActivityCollector::StartResult::kNotRegistered:
      profiler_failed = true;
      /* NVML telemetry does not depend on CUPTI activity, so keep it. */
      std::printf("\n--- Live GPU telemetry ---\n");
      telemetry.Start();
      return;
    case efimon::nvidia::ActivityCollector::StartResult::kNotEnabled:
      profiler_failed = true;
      return;
  }

  std::printf("\n--- Live GPU telemetry ---\n");

  telemetry.Start();
}

/* ------------------------------------------------ */
/* Profiler shutdown                                */
/* ------------------------------------------------ */

/* Writes capture_summary.txt with the counters of the capture. */
static void WriteCaptureSummary(uint64_t kernel_count, double exit_write_cpu_ms,
                                double exit_write_wall_ms) {
  if (!output_directory_ready) return;

  efimon::nvidia::CaptureSummary summary;
  summary.pid = profiler_owner_pid;
  summary.kernels = kernel_count;
  summary.distinct_kernel_names = activity.kernel_name_count();
  summary.memcpy_records = activity.memcpy_events().size();
  summary.dropped_activity_records = activity.dropped_records();
  summary.invalid_activity_records = activity.invalid_records();
  summary.other_device_activity_records = activity.other_device_records();
  summary.late_activity_records = activity.late_records();
  summary.activity_buffers_beyond_pool = activity.pool_misses();
  summary.telemetry_rows = telemetry.rows();
  summary.power_sample_rows = telemetry.power_sample_rows();
  summary.kernel_journal = activity.journal_state();
  summary.telemetry_thread_cpu_ms = telemetry.thread_cpu_ms();
  summary.activity_thread_cpu_ms = activity.thread_cpu_ms();
  summary.exit_write_cpu_ms = exit_write_cpu_ms;
  summary.exit_write_wall_ms = exit_write_wall_ms;
  summary.tail_wait_ms = tail_wait_ms;
  summary.profiler_failed = ProfilerFailed();
  summary.capture_status = CaptureStatus();
  if (!efimon::nvidia::WriteCaptureSummary(
          output_directory / "capture_summary.txt", summary)) {
    profiler_failed = true;
  }
}

static void ProfilerStop() {
  if (!profiler_active || getpid() != profiler_owner_pid) return;

  if (!activity.Stop()) profiler_failed = true;

  /* Keep real power samples beyond the last kernel for a full final window. */
  if (telemetry.sampling() && tail_duration.count() > 0) {
    const auto tail_start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(tail_duration);
    tail_wait_ms = std::chrono::duration<double, std::milli>(
                       std::chrono::steady_clock::now() - tail_start)
                       .count();
  }

  telemetry.Stop();
  if (telemetry.failed()) profiler_failed = true;

  if (!telemetry.FinishFiles()) profiler_failed = true;

  activity.SortMemcpyEvents();

  const uint64_t window_size_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(
          metrics_window_duration)
          .count());
  efimon::nvidia::ActiveTimeWindows windows(window_size_ns);
  uint64_t kernel_count = 0;
  const double exit_cpu_start_ms = ThreadCpuMilliseconds();
  const auto exit_wall_start = std::chrono::steady_clock::now();
  const bool activity_saved = SaveActivityCsvFiles(&windows, &kernel_count);
  if (!activity_saved) profiler_failed = true;
  const double exit_write_cpu_ms = ThreadCpuMilliseconds() - exit_cpu_start_ms;
  const double exit_write_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - exit_wall_start)
          .count();

  activity.FinishJournal(activity_saved);

  const uint64_t memory_bytes = telemetry.ProcessMemory(getpid());

  WriteCaptureSummary(kernel_count, exit_write_cpu_ms, exit_write_wall_ms);

  std::printf("\n--- Process Summary ---\n");

  std::printf("PID: %d\n", getpid());

  std::printf("Kernels total: %" PRIu64 "\n", kernel_count);
  std::printf("Distinct kernel names: %zu\n", activity.kernel_name_count());

  const size_t dropped = activity.dropped_records();

  if (dropped > 0) {
    std::fprintf(stderr, "[CUPTI] WARNING: %zu activity records were dropped\n",
                 dropped);
  }

  const size_t pool_misses = activity.pool_misses();
  if (pool_misses > 0) {
    std::printf("Activity buffers allocated beyond the pool: %zu\n",
                pool_misses);
  }

  std::printf("GPU memory: %.2f MB\n", memory_bytes / (1024.0 * 1024.0));

  /* ---------------------------------------------- */
  /* Configurable kernel windows                    */
  /* ---------------------------------------------- */

  if (!windows.empty()) {
    std::printf("\n--- %" PRId64 " ms Kernel Windows ---\n",
                static_cast<int64_t>(metrics_window_duration.count()));

    const std::vector<uint64_t> active_ns = windows.ActiveNs();
    for (size_t window_number = 0; window_number < active_ns.size();
         ++window_number) {
      const double relative_start_ms =
          static_cast<double>(window_number * window_size_ns) / 1e6;
      const double relative_end_ms = relative_start_ms + window_size_ns / 1e6;
      const double utilization = static_cast<double>(active_ns[window_number]) /
                                 static_cast<double>(window_size_ns) * 100.0;

      std::printf(
          "Window %zu [%.0f-%.0f ms] | Active: %.3f ms | "
          "Utilization: %.2f %%\n",
          window_number, relative_start_ms, relative_end_ms,
          active_ns[window_number] / 1e6, utilization);
    }
  } else {
    std::printf("\nNo kernels captured\n");
    if (!options.follow_children) {
      std::printf(
          "If the command only starts the CUDA program, set "
          "NVIDIA_METRICS_FOLLOW_CHILDREN=1\n");
    }
  }

  /* ---------------------------------------------- */
  /* Memory copies                                  */
  /* ---------------------------------------------- */

  std::printf("\n--- Memory copies ---\n");

  std::printf("H2D: %.2f MB\n", activity.bytes_h2d() / (1024.0 * 1024.0));

  std::printf("D2H: %.2f MB\n", activity.bytes_d2h() / (1024.0 * 1024.0));

  std::printf("D2D: %.2f MB\n", activity.bytes_d2d() / (1024.0 * 1024.0));

  telemetry.Shutdown();

  std::printf("=======================\n");

  ReportCompletion();
}

class ProfilerLifecycle {
 public:
  ProfilerLifecycle() { ProfilerStart(); }

  ~ProfilerLifecycle() { ProfilerStop(); }

  ProfilerLifecycle(const ProfilerLifecycle&) = delete;
  ProfilerLifecycle& operator=(const ProfilerLifecycle&) = delete;
};

/* This object must remain after the profiler state declarations in this
   translation unit. C++ destroys it first, ensuring ProfilerStop() joins the
   sampling and activity threads before the containers and synchronization
   objects are destroyed. */
static ProfilerLifecycle profiler_lifecycle;
