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
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <deque>
#include <efimon/gpu/nvidia-gpu/gpu-telemetry.hpp>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-options.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <exception>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
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

struct MemcpyEvent {
  uint64_t start;
  uint64_t end;
  uint64_t bytes;
  uint32_t device_id;
  uint32_t context_id;
  uint32_t stream_id;
  uint32_t correlation_id;
  uint8_t copy_kind;
};

struct CompletedActivityBuffer {
  uint8_t* buffer;
  size_t valid_size;
};

/* Profiler state is global because the library has no entry point other than
   its load-time constructor and exit-time destructor. */

/* Kernels kept in memory when the journal is disabled or cannot be written. A
   deque never copies existing records when it grows. Guarded by
   activity_mutex. */
static std::deque<KernelEvent> kernels;
/* Deque elements stay in place, so kernel_name_ids can view their characters.
 */
static std::deque<std::string> kernel_names;
static std::vector<std::string> escaped_kernel_names;
static std::unordered_map<std::string_view, uint32_t> kernel_name_ids;
static std::deque<MemcpyEvent> memcpy_events;
static std::mutex activity_mutex;
static std::atomic<size_t> dropped_activity_records{0};
static std::atomic<bool> dropped_records_query_failed{false};
static std::atomic<bool> activity_record_error_reported{false};
/* Records left out of the CSVs: incomplete ones, whose timestamps are zero or
   out of order, and those of a device other than measured_device_id. */
static std::atomic<size_t> invalid_activity_records{0};
static std::atomic<size_t> other_device_activity_records{0};
/* CUDA device ordinal selected for NVML, as in CUPTI's deviceId; -1 keeps the
   records of every device, also when the selected device does not exist. */
static int64_t measured_device_id = -1;
/* Set under activity_mutex once the CSVs start to be written. Buffers that
   CUPTI returns later are only counted, so the export reads the containers
   without the lock. */
static bool activity_closed = false;
static std::atomic<size_t> late_activity_records{0};

static uint64_t bytes_h2d = 0;
static uint64_t bytes_d2h = 0;
static uint64_t bytes_d2d = 0;

/* NVML device, sampling thread and telemetry CSVs of the measured GPU. */
static efimon::nvidia::GpuTelemetry telemetry;
static pid_t profiler_owner_pid = -1;
/* False in processes that only inherited LD_PRELOAD; see ConfigureProcessScope.
 */
static bool profiler_active = false;
/* True once CUPTI accepted the activity callbacks. Another CUPTI client, such
   as Nsight Systems or compute-sanitizer, makes the registration fail; the
   activity API must then not be used again. */
static bool activity_registered = false;
static int profiler_completion_fd = -1;
static std::atomic<bool> profiler_failed{false};
static std::atomic<bool> activity_allocation_error_reported{false};

static ProfilerOptions options;
static std::filesystem::path output_directory(".");
static bool output_directory_ready = false;
/* Kernels of the CUPTI buffer being parsed and the journal that receives
   them, both guarded by activity_mutex. */
static std::vector<KernelEvent> parsed_kernels;
static efimon::nvidia::KernelJournal kernel_journal;

/* Counters for capture_summary.txt. Each has one writer thread and is read
   after that thread has joined. */
static double activity_thread_cpu_ms = 0.0;

/* Every pooled buffer, and the subset that CUPTI does not currently hold. */
static std::mutex activity_pool_mutex;
static std::vector<uint8_t*> pooled_activity_buffers;
static std::vector<uint8_t*> free_activity_buffers;
static std::atomic<size_t> activity_pool_misses{0};

/* Buffers handed over by BufferCompleted() for the activity worker. */
static std::mutex completed_buffers_mutex;
static std::condition_variable completed_buffers_ready;
static std::deque<CompletedActivityBuffer> completed_buffers;
static bool activity_worker_accepting = false;
static std::thread* activity_worker_thread = nullptr;

constexpr char PROFILER_COMPLETION_SUCCESS = '0';
constexpr char PROFILER_COMPLETION_FAILURE = '1';
/* The CSVs were written but lack data; see CaptureIsPartial(). */
constexpr char PROFILER_COMPLETION_PARTIAL = '2';

static std::chrono::milliseconds metrics_window_duration(
    kDefaultMetricsWindowMs);

constexpr size_t BUFFER_SIZE = 1024 * 1024;
constexpr size_t ACTIVITY_BUFFER_ALIGNMENT = 8;
constexpr size_t MIN_QUEUED_ACTIVITY_BUFFERS = 2;
/* Telemetry kept after the last kernel at exit, so that the last kernel window
   has real power samples. NVIDIA_METRICS_TAIL_MS sets it; one kernel window by
   default. */
static std::chrono::milliseconds tail_duration(0);
static double tail_wait_ms = 0.0;

constexpr char KERNEL_CSV_HEADER[] =
    "event_id,pid,device_id,context_id,stream_id,correlation_id,"
    "start_ns,end_ns,duration_ns,kernel_name\n";

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
  return !telemetry.initialized() ||
         dropped_activity_records.load(std::memory_order_relaxed) > 0 ||
         invalid_activity_records.load(std::memory_order_relaxed) > 0 ||
         late_activity_records.load(std::memory_order_relaxed) > 0;
}

static const char* CaptureStatus() {
  if (profiler_failed.load(std::memory_order_acquire)) return "failed";
  return CaptureIsPartial() ? "partial" : "complete";
}

static void ReportCompletion() {
  if (profiler_completion_fd < 0 || getpid() != profiler_owner_pid) return;

  char status = PROFILER_COMPLETION_SUCCESS;
  if (profiler_failed.load(std::memory_order_acquire)) {
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

static std::filesystem::path KernelJournalPath() {
  return output_directory / "kernel_activity.journal";
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

  if (options.kernel_journal &&
      !kernel_journal.Open(KernelJournalPath(), profiler_owner_pid)) {
    std::fprintf(stderr,
                 "[Profiler] Cannot open %s; kernels are kept in memory "
                 "until exit\n",
                 KernelJournalPath().c_str());
    options.kernel_journal = false;
  }

  telemetry.OpenPowerSamplesFile();
  return true;
}

static const char* MemcpyKindName(uint8_t copy_kind) {
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

/* Writes every kernel to file in start order and adds it to windows: the
   journal runs merged with the kernels kept in memory. Returns false if the
   journal cannot be read. */
static bool WriteSortedKernels(std::ofstream* file,
                               efimon::nvidia::ActiveTimeWindows* windows,
                               uint64_t* kernel_count) {
  std::sort(kernels.begin(), kernels.end(), efimon::nvidia::KernelRecordLess);

  /* Rows are formatted with std::to_chars into one buffer and written in
     large blocks; the text is the same as with stream formatting. */
  constexpr size_t kCsvBlockBytes = 1 << 20;
  std::string rows;
  rows.reserve(kCsvBlockBytes + 512);
  uint64_t event_id = 0;
  auto write = [&](const KernelEvent& kernel) {
    efimon::nvidia::AppendKernelCsvRow(event_id++, profiler_owner_pid, kernel,
                                       escaped_kernel_names[kernel.name_id],
                                       &rows);
    windows->Add(kernel.start_ns, kernel.end_ns);
    if (rows.size() >= kCsvBlockBytes) {
      file->write(rows.data(), static_cast<std::streamsize>(rows.size()));
      rows.clear();
    }
  };

  size_t memory_index = 0;
  bool journal_read = true;
  if (kernel_journal.record_count() > 0) {
    journal_read = kernel_journal.ForEachSorted([&](const KernelEvent& kernel) {
      while (memory_index < kernels.size() &&
             efimon::nvidia::KernelRecordLess(kernels[memory_index], kernel)) {
        write(kernels[memory_index++]);
      }
      write(kernel);
    });
  }
  while (memory_index < kernels.size()) write(kernels[memory_index++]);
  file->write(rows.data(), static_cast<std::streamsize>(rows.size()));

  windows->Finish();
  *kernel_count = event_id;
  if (!journal_read) {
    std::fprintf(stderr, "[Profiler] Cannot read %s\n",
                 KernelJournalPath().c_str());
  }
  return journal_read;
}

/* Writes the sorted kernel and memory copy CSVs that the analyzers read. */
static bool SaveActivityCsvFiles(efimon::nvidia::ActiveTimeWindows* windows,
                                 uint64_t* kernel_count) {
  if (!output_directory_ready) return false;

  bool success = true;

  const auto kernels_path = output_directory / "kernel_activity.csv";
  std::ofstream kernels_file(kernels_path);
  if (!kernels_file.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 kernels_path.c_str());
    success = false;
  } else {
    kernels_file << KERNEL_CSV_HEADER;
    if (!WriteSortedKernels(&kernels_file, windows, kernel_count)) {
      success = false;
    }
    if (!FinishCsvFile(&kernels_file, kernels_path)) success = false;
  }

  if (options.memcpy_csv) {
    const auto memcpy_path = output_directory / "memcpy_activity.csv";
    std::ofstream memcpy_file(memcpy_path);
    if (!memcpy_file.is_open()) {
      std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                   memcpy_path.c_str());
      success = false;
    } else {
      memcpy_file << "event_id,pid,device_id,context_id,stream_id,"
                     "correlation_id,start_ns,end_ns,duration_ns,bytes,"
                     "copy_kind\n";
      for (size_t index = 0; index < memcpy_events.size(); ++index) {
        const auto& copy = memcpy_events[index];
        memcpy_file << index << ',' << profiler_owner_pid << ','
                    << copy.device_id << ',' << copy.context_id << ','
                    << copy.stream_id << ',' << copy.correlation_id << ','
                    << copy.start << ',' << copy.end << ','
                    << (copy.end - copy.start) << ',' << copy.bytes << ','
                    << MemcpyKindName(copy.copy_kind) << '\n';
      }
      if (!FinishCsvFile(&memcpy_file, memcpy_path)) success = false;
    }
  }

  if (success) {
    std::printf("Timeline CSV: %s\nKernel CSV: %s\n",
                telemetry.telemetry_path().c_str(), kernels_path.c_str());
  }
  return success;
}

/* ------------------------------------------------ */
/* CUPTI buffer                                     */
/* ------------------------------------------------ */

static void PreallocateActivityBuffers() {
  std::lock_guard<std::mutex> lock(activity_pool_mutex);
  for (size_t index = 0; index < options.activity_pool_buffers; ++index) {
    auto* buffer = static_cast<uint8_t*>(
        std::aligned_alloc(ACTIVITY_BUFFER_ALIGNMENT, BUFFER_SIZE));
    if (buffer == nullptr) {
      std::fprintf(stderr,
                   "[CUPTI] Preallocated %zu of %zu activity buffers; the "
                   "rest are allocated on demand\n",
                   index, options.activity_pool_buffers);
      break;
    }
    pooled_activity_buffers.push_back(buffer);
    free_activity_buffers.push_back(buffer);
  }
}

/* Returns a pooled buffer to the pool, or frees one allocated on demand. */
static void ReleaseActivityBuffer(uint8_t* buffer) {
  {
    std::lock_guard<std::mutex> lock(activity_pool_mutex);
    if (std::find(pooled_activity_buffers.begin(),
                  pooled_activity_buffers.end(),
                  buffer) != pooled_activity_buffers.end()) {
      free_activity_buffers.push_back(buffer);
      return;
    }
  }
  std::free(buffer);
}

/* Frees the pooled buffers that CUPTI returned. A buffer CUPTI still holds is
   no longer pooled, so ReleaseActivityBuffer() frees it when it comes back,
   and any later request allocates on demand. */
static void ReleaseActivityBufferPool() {
  std::lock_guard<std::mutex> lock(activity_pool_mutex);
  for (uint8_t* buffer : free_activity_buffers) std::free(buffer);
  free_activity_buffers.clear();
  pooled_activity_buffers.clear();
}

void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size,
                              size_t* max_records) {
  *buffer = nullptr;
  *size = 0;
  *max_records = 0;

  uint8_t* requested_buffer = nullptr;
  if (options.activity_pool_buffers > 0) {
    std::lock_guard<std::mutex> lock(activity_pool_mutex);
    if (!free_activity_buffers.empty()) {
      requested_buffer = free_activity_buffers.back();
      free_activity_buffers.pop_back();
    }
  }

  if (requested_buffer == nullptr) {
    if (options.activity_pool_buffers > 0) {
      activity_pool_misses.fetch_add(1, std::memory_order_relaxed);
    }
    requested_buffer = static_cast<uint8_t*>(
        std::aligned_alloc(ACTIVITY_BUFFER_ALIGNMENT, BUFFER_SIZE));
  }

  if (requested_buffer == nullptr) {
    profiler_failed.store(true, std::memory_order_release);
    if (!activity_allocation_error_reported.exchange(
            true, std::memory_order_relaxed)) {
      std::fprintf(stderr,
                   "[CUPTI] Cannot allocate activity buffer; profiling data "
                   "will be incomplete\n");
    }
    return;
  }

  *buffer = requested_buffer;
  *size = BUFFER_SIZE;
}

/* ------------------------------------------------ */
/* CUPTI records                                    */
/* ------------------------------------------------ */

/* Returns the id of name, adding it on first use. Caller holds activity_mutex.
 */
static uint32_t InternKernelName(const char* name) {
  const std::string_view key = name != nullptr ? name : "unknown";
  const auto found = kernel_name_ids.find(key);
  if (found != kernel_name_ids.end()) return found->second;

  const auto id = static_cast<uint32_t>(kernel_names.size());
  kernel_names.emplace_back(key);
  escaped_kernel_names.push_back(EscapeCsv(kernel_names.back()));
  kernel_name_ids.emplace(std::string_view(kernel_names.back()), id);
  /* A write error surfaces when the run of this buffer is appended. */
  if (options.kernel_journal)
    kernel_journal.AppendName(id, kernel_names.back());
  return id;
}

/* Returns whether a kernel or memory copy record goes to the CSVs. A forced
   flush can return records that are not complete, with zero timestamps, which
   would move the start of the kernel windows; records of another device would
   be attributed the energy of the measured one. */
static bool KeepActivityRecord(uint32_t device_id, uint64_t start_ns,
                               uint64_t end_ns) {
  if (start_ns == 0 || end_ns < start_ns) {
    invalid_activity_records.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (measured_device_id >= 0 &&
      device_id != static_cast<uint64_t>(measured_device_id)) {
    other_device_activity_records.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

/* Copies the needed fields of every record in a completed CUPTI buffer. */
static void ParseActivityBuffer(uint8_t* buffer, size_t valid_size) {
  std::lock_guard<std::mutex> lock(activity_mutex);

  CUpti_Activity* record = nullptr;
  CUptiResult result = CUPTI_SUCCESS;
  if (activity_closed) {
    size_t late = 0;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
           CUPTI_SUCCESS) {
      ++late;
    }
    late_activity_records.fetch_add(late, std::memory_order_relaxed);
    return;
  }
  while (CUPTI_SUCCESS ==
         (result = cuptiActivityGetNextRecord(buffer, valid_size, &record))) {
    /* Kernel */
    if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
      auto* kernel =
#if CUPTI_API_VERSION <= 26
          reinterpret_cast<CUpti_ActivityKernel9*>(
#else
          reinterpret_cast<CUpti_ActivityKernel10*>(
#endif
              record);
      if (!KeepActivityRecord(kernel->deviceId, kernel->start, kernel->end)) {
        continue;
      }

      const KernelEvent event{kernel->start,
                              kernel->end,
                              kernel->deviceId,
                              kernel->contextId,
                              kernel->streamId,
                              kernel->correlationId,
                              InternKernelName(kernel->name)};
      if (options.kernel_journal) {
        parsed_kernels.push_back(event);
      } else {
        kernels.push_back(event);
      }
    }

    /* Memory copies */
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* copy = reinterpret_cast<CUpti_ActivityMemcpy*>(record);
      if (!KeepActivityRecord(copy->deviceId, copy->start, copy->end)) {
        continue;
      }

      switch (copy->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
          bytes_h2d += copy->bytes;
          break;

        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
          bytes_d2h += copy->bytes;
          break;

        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
          bytes_d2d += copy->bytes;
          break;

        default:
          break;
      }

      if (options.memcpy_csv) {
        memcpy_events.push_back({copy->start, copy->end, copy->bytes,
                                 copy->deviceId, copy->contextId,
                                 copy->streamId, copy->correlationId,
                                 copy->copyKind});
      }
    }
  }

  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED &&
      !activity_record_error_reported.exchange(true,
                                               std::memory_order_relaxed)) {
    PrintCuptiError("Failed to read an activity record", result);
    profiler_failed.store(true, std::memory_order_release);
  }

  /* AppendRun() flushes every run, so an abnormal exit keeps the completed
     buffers. After a write error the kernels of this buffer stay in
     parsed_kernels, and the runs already written are still merged at exit. */
  if (options.kernel_journal && !parsed_kernels.empty() &&
      !kernel_journal.AppendRun(&parsed_kernels)) {
    std::fprintf(stderr,
                 "[Profiler] Cannot write %s; later kernels are kept in "
                 "memory until exit\n",
                 KernelJournalPath().c_str());
    kernels.insert(kernels.end(), parsed_kernels.begin(), parsed_kernels.end());
    parsed_kernels.clear();
    options.kernel_journal = false;
  }
}

static void ProcessCompletedBuffers() {
  NameProfilerThread("nvpm-activity");
  while (true) {
    CompletedActivityBuffer completed{};
    {
      std::unique_lock<std::mutex> lock(completed_buffers_mutex);
      completed_buffers_ready.wait(lock, [] {
        return !completed_buffers.empty() || !activity_worker_accepting;
      });
      if (completed_buffers.empty()) {
        activity_thread_cpu_ms = ThreadCpuMilliseconds();
        return;
      }
      completed = completed_buffers.front();
      completed_buffers.pop_front();
    }
    ParseActivityBuffer(completed.buffer, completed.valid_size);
    ReleaseActivityBuffer(completed.buffer);
  }
}

static void StartActivityWorker() {
  if (!options.activity_worker) return;

  {
    std::lock_guard<std::mutex> lock(completed_buffers_mutex);
    activity_worker_accepting = true;
  }
  try {
    activity_worker_thread = new std::thread(ProcessCompletedBuffers);
  } catch (const std::exception& error) {
    {
      std::lock_guard<std::mutex> lock(completed_buffers_mutex);
      activity_worker_accepting = false;
    }
    std::fprintf(stderr,
                 "[CUPTI] Cannot start the activity worker; buffers are "
                 "parsed in the CUPTI callback: %s\n",
                 error.what());
  }
}

static size_t MaxQueuedActivityBuffers() {
  return std::max(options.activity_pool_buffers, MIN_QUEUED_ACTIVITY_BUFFERS);
}

/* Stops accepting buffers and waits until the worker parsed every queued one.
 */
static void StopActivityWorker() {
  {
    std::lock_guard<std::mutex> lock(completed_buffers_mutex);
    activity_worker_accepting = false;
  }
  completed_buffers_ready.notify_all();

  if (activity_worker_thread == nullptr) return;
  if (activity_worker_thread->joinable()) activity_worker_thread->join();
  delete activity_worker_thread;
  activity_worker_thread = nullptr;
}

void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id,
                              uint8_t* buffer, size_t, size_t valid_size) {
  size_t dropped = 0;
  const CUptiResult dropped_result =
      cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);

  if (dropped_result == CUPTI_SUCCESS) {
    dropped_activity_records.fetch_add(dropped, std::memory_order_relaxed);
  } else if (!dropped_records_query_failed.exchange(
                 true, std::memory_order_relaxed)) {
    PrintCuptiError("Failed to query dropped activity records", dropped_result);
  }

  if (buffer == nullptr) return;

  /* Hand the buffer to the worker so that CUPTI's thread only enqueues it.
     A full queue means the worker is behind; parsing here then bounds the
     memory held by queued buffers, as before the worker existed. */
  {
    std::lock_guard<std::mutex> lock(completed_buffers_mutex);
    if (activity_worker_accepting &&
        completed_buffers.size() < MaxQueuedActivityBuffers()) {
      completed_buffers.push_back({buffer, valid_size});
      completed_buffers_ready.notify_one();
      return;
    }
  }

  ParseActivityBuffer(buffer, valid_size);
  ReleaseActivityBuffer(buffer);
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
  if (!PrepareOutputFiles()) profiler_failed = true;

  /* NVML */

  int device_index = 0;

  if (GetConfiguredDeviceIndex(&device_index)) {
    std::string pci_bus_id;
    if (GetCudaPciBusId(device_index, &pci_bus_id)) {
      if (!options.all_devices) measured_device_id = device_index;
      telemetry.InitializeDevice(device_index, pci_bus_id);
    }
  }

  /* CUPTI */

  PreallocateActivityBuffers();
  StartActivityWorker();

  CUptiResult cupti_result =
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError(
        "Cannot register activity callbacks; kernel and memory copy "
        "activity are disabled. Another CUPTI client (Nsight Systems, Nsight "
        "Compute or compute-sanitizer) may be attached",
        cupti_result);
    profiler_failed = true;
    /* NVML telemetry does not depend on CUPTI activity, so keep it. */
    std::printf("\n--- Live GPU telemetry ---\n");
    telemetry.Start();
    return;
  }
  activity_registered = true;

  cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError("Failed to enable concurrent kernel activity",
                    cupti_result);
    profiler_failed = true;
    return;
  }

  cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError("Failed to enable memory copy activity", cupti_result);
    profiler_failed = true;
    return;
  }

  /* The periodic flush only returns full buffers; exit still flushes all. */
  if (options.activity_flush_ms > 0) {
    cupti_result = cuptiActivityFlushPeriod(options.activity_flush_ms);
    if (CUPTI_SUCCESS != cupti_result) {
      PrintCuptiError("Failed to set the activity flush period", cupti_result);
    }
  }

  std::printf("\n--- Live GPU telemetry ---\n");

  telemetry.Start();
}

/* ------------------------------------------------ */
/* Profiler shutdown                                */
/* ------------------------------------------------ */

/* Writes capture_summary.txt: one key=value per line with the counters that
   tell whether a capture is complete, plus the CPU time of the profiler's
   own threads. CUPTI's internal threads are not included. */
static void WriteCaptureSummary(uint64_t kernel_count, double exit_write_cpu_ms,
                                double exit_write_wall_ms) {
  if (!output_directory_ready) return;

  const std::filesystem::path path = output_directory / "capture_summary.txt";
  std::ofstream summary(path);
  if (!summary.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open %s\n", path.c_str());
    profiler_failed = true;
    return;
  }

  const char* journal_state = "disabled";
  if (kernel_journal.failed()) {
    journal_state = "failed";
  } else if (kernel_journal.is_open() || kernel_journal.record_count() > 0) {
    journal_state = "used";
  }

  summary << "pid=" << profiler_owner_pid << '\n'
          << "kernels=" << kernel_count << '\n'
          << "distinct_kernel_names=" << kernel_names.size() << '\n'
          << "memcpy_records=" << memcpy_events.size() << '\n'
          << "dropped_activity_records="
          << dropped_activity_records.load(std::memory_order_relaxed) << '\n'
          << "invalid_activity_records="
          << invalid_activity_records.load(std::memory_order_relaxed) << '\n'
          << "other_device_activity_records="
          << other_device_activity_records.load(std::memory_order_relaxed)
          << '\n'
          << "late_activity_records="
          << late_activity_records.load(std::memory_order_relaxed) << '\n'
          << "activity_buffers_beyond_pool="
          << activity_pool_misses.load(std::memory_order_relaxed) << '\n'
          << "telemetry_rows=" << telemetry.rows() << '\n'
          << "power_sample_rows=" << telemetry.power_sample_rows() << '\n'
          << "kernel_journal=" << journal_state << '\n'
          << std::fixed << std::setprecision(3)
          << "telemetry_thread_cpu_ms=" << telemetry.thread_cpu_ms() << '\n'
          << "activity_thread_cpu_ms=" << activity_thread_cpu_ms << '\n'
          << "exit_write_cpu_ms=" << exit_write_cpu_ms << '\n'
          << "exit_write_wall_ms=" << exit_write_wall_ms << '\n'
          << "tail_wait_ms=" << tail_wait_ms << '\n'
          << "profiler_failed="
          << (profiler_failed.load(std::memory_order_acquire) ? 1 : 0) << '\n'
          << "capture_status=" << CaptureStatus() << '\n';
  if (!FinishCsvFile(&summary, path)) profiler_failed = true;
}

static void ProfilerStop() {
  if (!profiler_active || getpid() != profiler_owner_pid) return;

  if (activity_registered) {
    /* Stop new records before the final flush. */
    for (const CUpti_ActivityKind kind :
         {CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL, CUPTI_ACTIVITY_KIND_MEMCPY}) {
      const CUptiResult disable_result = cuptiActivityDisable(kind);
      if (disable_result != CUPTI_SUCCESS) {
        PrintCuptiError("Failed to disable activity collection",
                        disable_result);
      }
    }
    const CUptiResult flush_result =
        cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);

    if (flush_result != CUPTI_SUCCESS) {
      PrintCuptiError("Failed to flush activity records", flush_result);
      profiler_failed = true;
    }
  }

  StopActivityWorker();
  {
    std::lock_guard<std::mutex> lock(activity_mutex);
    activity_closed = true;
  }
  ReleaseActivityBufferPool();

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

  std::sort(memcpy_events.begin(), memcpy_events.end(),
            [](const MemcpyEvent& left, const MemcpyEvent& right) {
              return left.start < right.start;
            });

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

  /* The sorted CSV supersedes the journal; keep the journal if it failed. */
  if (kernel_journal.is_open()) {
    if (!activity_saved) {
      kernel_journal.Flush();
      std::fprintf(stderr,
                   "[Profiler] Kept %s; recover it with "
                   "the journal recovery of efimon-analyser\n",
                   KernelJournalPath().c_str());
    } else if (!kernel_journal.Remove()) {
      std::fprintf(stderr, "[Profiler] Cannot remove %s\n",
                   KernelJournalPath().c_str());
    }
  }

  const uint64_t memory_bytes = telemetry.ProcessMemory(getpid());

  WriteCaptureSummary(kernel_count, exit_write_cpu_ms, exit_write_wall_ms);

  std::printf("\n--- Process Summary ---\n");

  std::printf("PID: %d\n", getpid());

  std::printf("Kernels total: %" PRIu64 "\n", kernel_count);
  std::printf("Distinct kernel names: %zu\n", kernel_names.size());

  const size_t dropped =
      dropped_activity_records.load(std::memory_order_relaxed);

  if (dropped > 0) {
    std::fprintf(stderr, "[CUPTI] WARNING: %zu activity records were dropped\n",
                 dropped);
  }

  const size_t pool_misses =
      activity_pool_misses.load(std::memory_order_relaxed);
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

  std::printf("H2D: %.2f MB\n", bytes_h2d / (1024.0 * 1024.0));

  std::printf("D2H: %.2f MB\n", bytes_d2h / (1024.0 * 1024.0));

  std::printf("D2D: %.2f MB\n", bytes_d2d / (1024.0 * 1024.0));

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
