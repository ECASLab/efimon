/**
 * @file process-metrics.cpp
 * @brief Preloaded collector for NVML telemetry and CUPTI kernel activity
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 *
 * The library is loaded with LD_PRELOAD into the profiled process. Its
 * load-time constructor starts the profiler and its exit-time destructor
 * writes the session files; the modules do the work:
 *
 * - profiler-options: NVIDIA_METRICS_* settings and the session directory.
 * - gpu-telemetry: NVML sampling thread and the telemetry CSVs.
 * - activity-collector: CUPTI kernel and memory copy activity.
 * - capture-writer: activity CSVs and capture_summary.txt at exit.
 */

#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>  // NOLINT
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <efimon/gpu/nvidia-gpu/activity-collector.hpp>
#include <efimon/gpu/nvidia-gpu/capture-writer.hpp>
#include <efimon/gpu/nvidia-gpu/gpu-telemetry.hpp>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-options.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <filesystem>  // NOLINT
#include <limits>
#include <string>
#include <system_error>  // NOLINT
#include <thread>        // NOLINT
#include <vector>

namespace efimon {
namespace nvidia {
namespace {

/** Completion byte: the capture is complete */
constexpr char kCompletionSuccess = '0';
/** Completion byte: the profiler failed */
constexpr char kCompletionFailure = '1';
/** Completion byte: the CSVs were written but lack data; see
    Profiler::CaptureIsPartial() */
constexpr char kCompletionPartial = '2';

/**
 * @brief Gets the PCI bus id of a CUDA device
 *
 * @param device_index CUDA device
 * @param pci_bus_id receives the PCI bus id; must not be nullptr
 * @return false if the CUDA driver cannot resolve the device
 */
bool GetCudaPciBusId(int device_index, std::string* pci_bus_id) {
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

/**
 * @brief Profiler of the process that loaded the library
 *
 * It owns the telemetry and the activity collection of one session. Start()
 * runs once from the load-time constructor and Stop() once from the
 * exit-time destructor; the other methods are only called from them.
 */
class Profiler {
 public:
  /**
   * @brief Construct a new Profiler object, not started
   */
  Profiler() = default;

  Profiler(const Profiler&) = delete;
  Profiler& operator=(const Profiler&) = delete;

  /**
   * @brief Reads the settings, opens the session files and starts the
   * telemetry and the CUPTI activity
   *
   * A process that only inherited LD_PRELOAD stays idle.
   */
  void Start();

  /**
   * @brief Stops the collection and writes the session files and summaries
   *
   * It does nothing in idle processes and in children that did not exec.
   */
  void Stop();

 private:
  void InitializeCompletionChannel();
  bool CaptureIsPartial() const;
  bool Failed() const;
  const char* CaptureStatus() const;
  void ReportCompletion();
  bool PrepareOutputFiles();
  bool SaveActivityCsvFiles(ActiveTimeWindows* windows, uint64_t* kernel_count);
  void WriteCaptureSummary(uint64_t kernel_count, double exit_write_cpu_ms,
                           double exit_write_wall_ms);
  void PrintProcessSummary(uint64_t kernel_count, uint64_t memory_bytes,
                           const ActiveTimeWindows& windows) const;

  /** CUPTI kernel and memory copy activity of the profiled process */
  ActivityCollector activity_;
  /** NVML device, sampling thread and telemetry CSVs of the measured GPU */
  GpuTelemetry telemetry_;
  /** Settings read at start */
  ProfilerOptions options_;
  /** Session directory */
  std::filesystem::path output_directory_{"."};
  /** True once the session directory exists */
  bool output_directory_ready_ = false;
  /** PID of the process that owns the session */
  pid_t owner_pid_ = -1;
  /** False in processes that only inherited LD_PRELOAD; see
      ConfigureProcessScope() */
  bool active_ = false;
  /** Descriptor that receives the completion byte; -1 when unused */
  int completion_fd_ = -1;
  /** Set when the capture fails outside the activity collection */
  std::atomic<bool> failed_{false};
  /** Kernel window */
  std::chrono::milliseconds window_duration_{kDefaultMetricsWindowMs};
  /** Telemetry kept after the last kernel at exit, so that the last kernel
      window has real power samples */
  std::chrono::milliseconds tail_duration_{0};
  /** Wall time of that tail */
  double tail_wait_ms_ = 0.0;
};

void Profiler::InitializeCompletionChannel() {
  owner_pid_ = getpid();
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
  completion_fd_ = descriptor;
}

/* The capture misses data although every file was written: no NVML
   telemetry, or kernels and memory copies that CUPTI dropped, did not
   complete or returned after the CSVs started. */
bool Profiler::CaptureIsPartial() const {
  return !telemetry_.initialized() || activity_.partial();
}

bool Profiler::Failed() const {
  return failed_.load(std::memory_order_acquire) || activity_.failed();
}

const char* Profiler::CaptureStatus() const {
  if (Failed()) return "failed";
  return CaptureIsPartial() ? "partial" : "complete";
}

void Profiler::ReportCompletion() {
  if (completion_fd_ < 0 || getpid() != owner_pid_) return;

  char status = kCompletionSuccess;
  if (Failed()) {
    status = kCompletionFailure;
  } else if (CaptureIsPartial()) {
    status = kCompletionPartial;
  }
  ssize_t written = -1;
  do {
    written = write(completion_fd_, &status, sizeof(status));
  } while (written < 0 && errno == EINTR);
  if (written != sizeof(status)) {
    std::fprintf(stderr, "[Profiler] Cannot report completion: %s\n",
                 written < 0 ? std::strerror(errno) : "short write");
  }
  close(completion_fd_);
  completion_fd_ = -1;
}

/* Creates the output directory and opens the files that are written while
   the profiled process runs. */
bool Profiler::PrepareOutputFiles() {
  std::error_code error;
  std::filesystem::create_directories(output_directory_, error);
  if (error) {
    std::fprintf(stderr, "[Profiler] Cannot create output directory %s: %s\n",
                 output_directory_.c_str(), error.message().c_str());
    return false;
  }
  output_directory_ready_ = true;

  if (!telemetry_.OpenTelemetryFile(output_directory_)) return false;

  activity_.OpenJournal(owner_pid_);

  telemetry_.OpenPowerSamplesFile();
  return true;
}

/* Writes the sorted kernel and memory copy CSVs that the analyzers read. */
bool Profiler::SaveActivityCsvFiles(ActiveTimeWindows* windows,
                                    uint64_t* kernel_count) {
  if (!output_directory_ready_) return false;

  const bool success =
      WriteActivityCsvFiles(output_directory_, owner_pid_, options_.memcpy_csv,
                            &activity_, windows, kernel_count);
  if (success) {
    std::printf("Timeline CSV: %s\nKernel CSV: %s\n",
                telemetry_.telemetry_path().c_str(),
                (output_directory_ / "kernel_activity.csv").c_str());
  }
  return success;
}

void Profiler::Start() {
  options_ = LoadProfilerOptions();
  if (!ConfigureProcessScope(options_.follow_children, &output_directory_)) {
    return;
  }
  active_ = true;
  InitializeCompletionChannel();

  std::printf("\n=== NVIDIA Process Profiler ===\n");

  std::printf("PID: %d\n", getpid());

  GetConfiguredWindowDuration(&window_duration_);
  tail_duration_ = GetConfiguredTailDuration(window_duration_);
  std::printf("Kernel window: %" PRId64 " ms; final telemetry: %" PRId64
              " ms\n",
              static_cast<int64_t>(window_duration_.count()),
              static_cast<int64_t>(tail_duration_.count()));
  std::printf(
      "Options: follow_children=%d activity_worker=%d pool_buffers=%zu "
      "flush_ms=%" PRIu32
      " kernel_journal=%d memcpy_csv=%d nvml_fields=%d "
      "power_bracket=%d power_samples=%d fixed_period=%d "
      "slow_read_ms=%" PRIu32 " all_devices=%d\n",
      options_.follow_children, options_.activity_worker,
      options_.activity_pool_buffers, options_.activity_flush_ms,
      options_.kernel_journal, options_.memcpy_csv, options_.nvml_fields,
      options_.power_bracket, options_.power_samples, options_.fixed_period,
      options_.slow_read_ms, options_.all_devices);

  telemetry_.Configure(options_);
  activity_.Configure(options_, output_directory_);
  if (!PrepareOutputFiles()) failed_ = true;

  /* NVML */

  int device_index = 0;

  if (GetConfiguredDeviceIndex(&device_index)) {
    std::string pci_bus_id;
    if (GetCudaPciBusId(device_index, &pci_bus_id)) {
      if (!options_.all_devices) activity_.set_measured_device(device_index);
      telemetry_.InitializeDevice(device_index, pci_bus_id);
    }
  }

  /* CUPTI */

  switch (activity_.Start()) {
    case ActivityCollector::StartResult::kStarted:
      break;
    case ActivityCollector::StartResult::kNotRegistered:
      failed_ = true;
      /* NVML telemetry does not depend on CUPTI activity, so keep it. */
      std::printf("\n--- Live GPU telemetry ---\n");
      telemetry_.Start();
      return;
    case ActivityCollector::StartResult::kNotEnabled:
      failed_ = true;
      return;
  }

  std::printf("\n--- Live GPU telemetry ---\n");

  telemetry_.Start();
}

/* Writes capture_summary.txt with the counters of the capture. */
void Profiler::WriteCaptureSummary(uint64_t kernel_count,
                                   double exit_write_cpu_ms,
                                   double exit_write_wall_ms) {
  if (!output_directory_ready_) return;

  CaptureSummary summary;
  summary.pid = owner_pid_;
  summary.kernels = kernel_count;
  summary.distinct_kernel_names = activity_.kernel_name_count();
  summary.memcpy_records = activity_.memcpy_events().size();
  summary.dropped_activity_records = activity_.dropped_records();
  summary.invalid_activity_records = activity_.invalid_records();
  summary.other_device_activity_records = activity_.other_device_records();
  summary.late_activity_records = activity_.late_records();
  summary.activity_buffers_beyond_pool = activity_.pool_misses();
  summary.telemetry_rows = telemetry_.rows();
  summary.power_sample_rows = telemetry_.power_sample_rows();
  summary.kernel_journal = activity_.journal_state();
  summary.telemetry_thread_cpu_ms = telemetry_.thread_cpu_ms();
  summary.activity_thread_cpu_ms = activity_.thread_cpu_ms();
  summary.exit_write_cpu_ms = exit_write_cpu_ms;
  summary.exit_write_wall_ms = exit_write_wall_ms;
  summary.tail_wait_ms = tail_wait_ms_;
  summary.profiler_failed = Failed();
  summary.capture_status = CaptureStatus();
  if (!efimon::nvidia::WriteCaptureSummary(
          output_directory_ / "capture_summary.txt", summary)) {
    failed_ = true;
  }
}

void Profiler::Stop() {
  if (!active_ || getpid() != owner_pid_) return;

  if (!activity_.Stop()) failed_ = true;

  /* Keep real power samples beyond the last kernel for a full final window. */
  if (telemetry_.sampling() && tail_duration_.count() > 0) {
    const auto tail_start = std::chrono::steady_clock::now();
    std::this_thread::sleep_for(tail_duration_);
    tail_wait_ms_ = std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - tail_start)
                        .count();
  }

  telemetry_.Stop();
  if (telemetry_.failed()) failed_ = true;

  if (!telemetry_.FinishFiles()) failed_ = true;

  activity_.SortMemcpyEvents();

  const uint64_t window_size_ns = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(window_duration_)
          .count());
  ActiveTimeWindows windows(window_size_ns);
  uint64_t kernel_count = 0;
  const double exit_cpu_start_ms = ThreadCpuMilliseconds();
  const auto exit_wall_start = std::chrono::steady_clock::now();
  const bool activity_saved = SaveActivityCsvFiles(&windows, &kernel_count);
  if (!activity_saved) failed_ = true;
  const double exit_write_cpu_ms = ThreadCpuMilliseconds() - exit_cpu_start_ms;
  const double exit_write_wall_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - exit_wall_start)
          .count();

  activity_.FinishJournal(activity_saved);

  const uint64_t memory_bytes = telemetry_.ProcessMemory(getpid());

  WriteCaptureSummary(kernel_count, exit_write_cpu_ms, exit_write_wall_ms);

  PrintProcessSummary(kernel_count, memory_bytes, windows);

  telemetry_.Shutdown();

  std::printf("=======================\n");

  ReportCompletion();
}

/* Prints the kernel counts, the kernel windows and the memory copies. */
void Profiler::PrintProcessSummary(uint64_t kernel_count, uint64_t memory_bytes,
                                   const ActiveTimeWindows& windows) const {
  std::printf("\n--- Process Summary ---\n");

  std::printf("PID: %d\n", getpid());

  std::printf("Kernels total: %" PRIu64 "\n", kernel_count);
  std::printf("Distinct kernel names: %zu\n", activity_.kernel_name_count());

  const size_t dropped = activity_.dropped_records();

  if (dropped > 0) {
    std::fprintf(stderr, "[CUPTI] WARNING: %zu activity records were dropped\n",
                 dropped);
  }

  const size_t pool_misses = activity_.pool_misses();
  if (pool_misses > 0) {
    std::printf("Activity buffers allocated beyond the pool: %zu\n",
                pool_misses);
  }

  std::printf("GPU memory: %.2f MB\n", memory_bytes / (1024.0 * 1024.0));

  if (!windows.empty()) {
    std::printf("\n--- %" PRId64 " ms Kernel Windows ---\n",
                static_cast<int64_t>(window_duration_.count()));

    const uint64_t window_size_ns = windows.window_ns();
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
    if (!options_.follow_children) {
      std::printf(
          "If the command only starts the CUDA program, set "
          "NVIDIA_METRICS_FOLLOW_CHILDREN=1\n");
    }
  }

  std::printf("\n--- Memory copies ---\n");

  std::printf("H2D: %.2f MB\n", activity_.bytes_h2d() / (1024.0 * 1024.0));

  std::printf("D2H: %.2f MB\n", activity_.bytes_d2h() / (1024.0 * 1024.0));

  std::printf("D2D: %.2f MB\n", activity_.bytes_d2d() / (1024.0 * 1024.0));
}

/**
 * @brief Starts the profiler when the library loads and stops it at exit
 *
 * The profiler is created with new and never destroyed, so CUPTI callbacks
 * that arrive while the process exits never reach a destroyed object.
 */
class ProfilerLifecycle {
 public:
  /**
   * @brief Creates and starts the profiler
   */
  ProfilerLifecycle() { profiler_->Start(); }

  /**
   * @brief Stops the profiler and writes the session files
   */
  ~ProfilerLifecycle() { profiler_->Stop(); }

  ProfilerLifecycle(const ProfilerLifecycle&) = delete;
  ProfilerLifecycle& operator=(const ProfilerLifecycle&) = delete;

 private:
  /** Profiler of this process; never deleted */
  Profiler* const profiler_ = new Profiler;
};

/* The only hook of a preloaded library is a static object: its constructor
   runs when the library loads and its destructor at exit. This is a
   deliberate exception to the rule that static objects are trivially
   destructible; all the state it uses lives in the never-deleted Profiler. */
ProfilerLifecycle profiler_lifecycle;

} /* namespace */
} /* namespace nvidia */
} /* namespace efimon */
