/**
 * @file activity-collector.cpp
 * @brief Collection of the CUPTI kernel and memory copy activity of the
 * profiled process
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cupti.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <efimon/gpu/nvidia-gpu/activity-collector.hpp>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <exception>
#include <filesystem>  // NOLINT
#include <functional>
#include <mutex>  // NOLINT
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <vector>

namespace efimon {
namespace nvidia {
namespace {

/** Size of each CUPTI activity buffer */
constexpr size_t kBufferSize = 1024 * 1024;
/** Alignment that CUPTI requires for the activity buffers */
constexpr size_t kBufferAlignment = 8;
/** Smallest queue of completed buffers, also without a pool */
constexpr size_t kMinQueuedBuffers = 2;

/** Collector that receives the CUPTI callbacks. CUPTI gives the callbacks no
    user data, so they reach the collector through this pointer, which never
    owns it */
ActivityCollector* active_collector = nullptr;

/**
 * @brief CUPTI callback that requests an activity buffer
 *
 * @param buffer receives the buffer
 * @param size receives the buffer size
 * @param max_records receives the record limit
 */
void CUPTIAPI BufferRequested(uint8_t** buffer, size_t* size,
                              size_t* max_records) {
  active_collector->RequestBuffer(buffer, size, max_records);
}

/**
 * @brief CUPTI callback that returns a completed activity buffer
 *
 * @param context CUDA context of the buffer
 * @param stream_id stream of the buffer
 * @param buffer completed buffer
 * @param valid_size bytes of valid records
 */
void CUPTIAPI BufferCompleted(CUcontext context, uint32_t stream_id,
                              uint8_t* buffer, size_t /*size*/,
                              size_t valid_size) {
  active_collector->CompleteBuffer(context, stream_id, buffer, valid_size);
}

} /* namespace */

void ActivityCollector::Configure(const ProfilerOptions& options,
                                  const std::filesystem::path& directory) {
  options_ = options;
  journal_path_ = directory / "kernel_activity.journal";
}

void ActivityCollector::OpenJournal(pid_t pid) {
  if (options_.kernel_journal && !journal_.Open(journal_path_, pid)) {
    std::fprintf(stderr,
                 "[Profiler] Cannot open %s; kernels are kept in memory "
                 "until exit\n",
                 journal_path_.c_str());
    options_.kernel_journal = false;
  }
}

ActivityCollector::StartResult ActivityCollector::Start() {
  PreallocateBuffers();
  StartWorker();

  active_collector = this;
  CUptiResult cupti_result =
      cuptiActivityRegisterCallbacks(BufferRequested, BufferCompleted);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError(
        "Cannot register activity callbacks; kernel and memory copy "
        "activity are disabled. Another CUPTI client (Nsight Systems, Nsight "
        "Compute or compute-sanitizer) may be attached",
        cupti_result);
    return StartResult::kNotRegistered;
  }
  registered_ = true;

  cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError("Failed to enable concurrent kernel activity",
                    cupti_result);
    return StartResult::kNotEnabled;
  }

  cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY);

  if (CUPTI_SUCCESS != cupti_result) {
    PrintCuptiError("Failed to enable memory copy activity", cupti_result);
    return StartResult::kNotEnabled;
  }

  /* The periodic flush only returns full buffers; exit still flushes all. */
  if (options_.activity_flush_ms > 0) {
    cupti_result = cuptiActivityFlushPeriod(options_.activity_flush_ms);
    if (CUPTI_SUCCESS != cupti_result) {
      PrintCuptiError("Failed to set the activity flush period", cupti_result);
    }
  }
  return StartResult::kStarted;
}

bool ActivityCollector::Stop() {
  bool flushed = true;
  if (registered_) {
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
      flushed = false;
    }
  }

  StopWorker();
  {
    std::lock_guard<std::mutex> lock(mutex_);
    closed_ = true;
  }
  ReleaseBufferPool();
  return flushed;
}

bool ActivityCollector::VisitKernelsSorted(
    const std::function<void(const KernelRecord&)>& visit) {
  std::sort(kernels_.begin(), kernels_.end(), KernelRecordLess);

  size_t memory_index = 0;
  bool journal_read = true;
  if (journal_.record_count() > 0) {
    journal_read = journal_.ForEachSorted([&](const KernelRecord& kernel) {
      while (memory_index < kernels_.size() &&
             KernelRecordLess(kernels_[memory_index], kernel)) {
        visit(kernels_[memory_index++]);
      }
      visit(kernel);
    });
  }
  while (memory_index < kernels_.size()) visit(kernels_[memory_index++]);

  if (!journal_read) {
    std::fprintf(stderr, "[Profiler] Cannot read %s\n", journal_path_.c_str());
  }
  return journal_read;
}

void ActivityCollector::SortMemcpyEvents() {
  std::sort(memcpy_events_.begin(), memcpy_events_.end(),
            [](const MemcpyEvent& left, const MemcpyEvent& right) {
              return left.start < right.start;
            });
}

void ActivityCollector::FinishJournal(bool csv_written) {
  if (!journal_.is_open()) return;
  if (!csv_written) {
    journal_.Flush();
    std::fprintf(stderr,
                 "[Profiler] Kept %s; recover it with "
                 "the journal recovery of efimon-analyser\n",
                 journal_path_.c_str());
  } else if (!journal_.Remove()) {
    std::fprintf(stderr, "[Profiler] Cannot remove %s\n",
                 journal_path_.c_str());
  }
}

const char* ActivityCollector::journal_state() const {
  if (journal_.failed()) return "failed";
  if (journal_.is_open() || journal_.record_count() > 0) return "used";
  return "disabled";
}

bool ActivityCollector::partial() const {
  return dropped_records() > 0 || invalid_records() > 0 || late_records() > 0;
}

void ActivityCollector::PreallocateBuffers() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  for (size_t index = 0; index < options_.activity_pool_buffers; ++index) {
    auto* buffer = static_cast<uint8_t*>(
        std::aligned_alloc(kBufferAlignment, kBufferSize));
    if (buffer == nullptr) {
      std::fprintf(stderr,
                   "[CUPTI] Preallocated %zu of %zu activity buffers; the "
                   "rest are allocated on demand\n",
                   index, options_.activity_pool_buffers);
      break;
    }
    pooled_buffers_.push_back(buffer);
    free_buffers_.push_back(buffer);
  }
}

/* Returns a pooled buffer to the pool, or frees one allocated on demand. */
void ActivityCollector::ReleaseBuffer(uint8_t* buffer) {
  {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (std::find(pooled_buffers_.begin(), pooled_buffers_.end(), buffer) !=
        pooled_buffers_.end()) {
      free_buffers_.push_back(buffer);
      return;
    }
  }
  std::free(buffer);
}

/* Frees the pooled buffers that CUPTI returned. A buffer CUPTI still holds is
   no longer pooled, so ReleaseBuffer() frees it when it comes back, and any
   later request allocates on demand. */
void ActivityCollector::ReleaseBufferPool() {
  std::lock_guard<std::mutex> lock(pool_mutex_);
  for (uint8_t* buffer : free_buffers_) std::free(buffer);
  free_buffers_.clear();
  pooled_buffers_.clear();
}

void ActivityCollector::RequestBuffer(uint8_t** buffer, size_t* size,
                                      size_t* max_records) {
  *buffer = nullptr;
  *size = 0;
  *max_records = 0;

  uint8_t* requested_buffer = nullptr;
  if (options_.activity_pool_buffers > 0) {
    std::lock_guard<std::mutex> lock(pool_mutex_);
    if (!free_buffers_.empty()) {
      requested_buffer = free_buffers_.back();
      free_buffers_.pop_back();
    }
  }

  if (requested_buffer == nullptr) {
    if (options_.activity_pool_buffers > 0) {
      pool_misses_.fetch_add(1, std::memory_order_relaxed);
    }
    requested_buffer = static_cast<uint8_t*>(
        std::aligned_alloc(kBufferAlignment, kBufferSize));
  }

  if (requested_buffer == nullptr) {
    failed_.store(true, std::memory_order_release);
    if (!allocation_error_reported_.exchange(true, std::memory_order_relaxed)) {
      std::fprintf(stderr,
                   "[CUPTI] Cannot allocate activity buffer; profiling data "
                   "will be incomplete\n");
    }
    return;
  }

  *buffer = requested_buffer;
  *size = kBufferSize;
}

/* Returns the id of name, adding it on first use. Caller holds mutex_. */
uint32_t ActivityCollector::InternKernelName(const char* name) {
  const std::string_view key = name != nullptr ? name : "unknown";
  const auto found = kernel_name_ids_.find(key);
  if (found != kernel_name_ids_.end()) return found->second;

  const auto id = static_cast<uint32_t>(kernel_names_.size());
  kernel_names_.emplace_back(key);
  escaped_kernel_names_.push_back(EscapeCsv(kernel_names_.back()));
  kernel_name_ids_.emplace(std::string_view(kernel_names_.back()), id);
  /* A write error surfaces when the run of this buffer is appended. */
  if (options_.kernel_journal) journal_.AppendName(id, kernel_names_.back());
  return id;
}

/* Returns whether a kernel or memory copy record goes to the CSVs. A forced
   flush can return records that are not complete, with zero timestamps, which
   would move the start of the kernel windows; records of another device would
   be attributed the energy of the measured one. */
bool ActivityCollector::KeepRecord(uint32_t device_id, uint64_t start_ns,
                                   uint64_t end_ns) {
  if (start_ns == 0 || end_ns < start_ns) {
    invalid_records_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  if (measured_device_id_ >= 0 &&
      device_id != static_cast<uint64_t>(measured_device_id_)) {
    other_device_records_.fetch_add(1, std::memory_order_relaxed);
    return false;
  }
  return true;
}

/* Copies the needed fields of every record in a completed CUPTI buffer. */
void ActivityCollector::ParseBuffer(uint8_t* buffer, size_t valid_size) {
  std::lock_guard<std::mutex> lock(mutex_);

  CUpti_Activity* record = nullptr;
  CUptiResult result = CUPTI_SUCCESS;
  if (closed_) {
    size_t late = 0;
    while (cuptiActivityGetNextRecord(buffer, valid_size, &record) ==
           CUPTI_SUCCESS) {
      ++late;
    }
    late_records_.fetch_add(late, std::memory_order_relaxed);
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
      if (!KeepRecord(kernel->deviceId, kernel->start, kernel->end)) {
        continue;
      }

      const KernelRecord event{kernel->start,
                               kernel->end,
                               kernel->deviceId,
                               kernel->contextId,
                               kernel->streamId,
                               kernel->correlationId,
                               InternKernelName(kernel->name)};
      if (options_.kernel_journal) {
        parsed_kernels_.push_back(event);
      } else {
        kernels_.push_back(event);
      }
    }

    /* Memory copies */
    if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
      auto* copy = reinterpret_cast<CUpti_ActivityMemcpy*>(record);
      if (!KeepRecord(copy->deviceId, copy->start, copy->end)) {
        continue;
      }

      switch (copy->copyKind) {
        case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
          bytes_h2d_ += copy->bytes;
          break;

        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
          bytes_d2h_ += copy->bytes;
          break;

        case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
          bytes_d2d_ += copy->bytes;
          break;

        default:
          break;
      }

      if (options_.memcpy_csv) {
        memcpy_events_.push_back({copy->start, copy->end, copy->bytes,
                                  copy->deviceId, copy->contextId,
                                  copy->streamId, copy->correlationId,
                                  copy->copyKind});
      }
    }
  }

  if (result != CUPTI_ERROR_MAX_LIMIT_REACHED &&
      !record_error_reported_.exchange(true, std::memory_order_relaxed)) {
    PrintCuptiError("Failed to read an activity record", result);
    failed_.store(true, std::memory_order_release);
  }

  /* AppendRun() flushes every run, so an abnormal exit keeps the completed
     buffers. After a write error the kernels of this buffer stay in
     parsed_kernels_, and the runs already written are still merged at
     exit. */
  if (options_.kernel_journal && !parsed_kernels_.empty() &&
      !journal_.AppendRun(&parsed_kernels_)) {
    std::fprintf(stderr,
                 "[Profiler] Cannot write %s; later kernels are kept in "
                 "memory until exit\n",
                 journal_path_.c_str());
    kernels_.insert(kernels_.end(), parsed_kernels_.begin(),
                    parsed_kernels_.end());
    parsed_kernels_.clear();
    options_.kernel_journal = false;
  }
}

void ActivityCollector::ProcessCompletedBuffers() {
  NameProfilerThread("nvpm-activity");
  while (true) {
    CompletedBuffer completed{};
    {
      std::unique_lock<std::mutex> lock(completed_mutex_);
      completed_ready_.wait(lock, [this] {
        return !completed_buffers_.empty() || !worker_accepting_;
      });
      if (completed_buffers_.empty()) {
        thread_cpu_ms_ = ThreadCpuMilliseconds();
        return;
      }
      completed = completed_buffers_.front();
      completed_buffers_.pop_front();
    }
    ParseBuffer(completed.buffer, completed.valid_size);
    ReleaseBuffer(completed.buffer);
  }
}

void ActivityCollector::StartWorker() {
  if (!options_.activity_worker) return;

  {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    worker_accepting_ = true;
  }
  /* std::thread reports a failure to start only with an exception; catching
     it keeps the profiled process alive and parses in the callback. */
  try {
    worker_ =
        new std::thread(&ActivityCollector::ProcessCompletedBuffers, this);
  } catch (const std::exception& error) {
    {
      std::lock_guard<std::mutex> lock(completed_mutex_);
      worker_accepting_ = false;
    }
    std::fprintf(stderr,
                 "[CUPTI] Cannot start the activity worker; buffers are "
                 "parsed in the CUPTI callback: %s\n",
                 error.what());
  }
}

size_t ActivityCollector::MaxQueuedBuffers() const {
  return std::max(options_.activity_pool_buffers, kMinQueuedBuffers);
}

/* Stops accepting buffers and waits until the worker parsed every queued
   one. */
void ActivityCollector::StopWorker() {
  {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    worker_accepting_ = false;
  }
  completed_ready_.notify_all();

  if (worker_ == nullptr) return;
  if (worker_->joinable()) worker_->join();
  delete worker_;
  worker_ = nullptr;
}

void ActivityCollector::CompleteBuffer(CUcontext context, uint32_t stream_id,
                                       uint8_t* buffer, size_t valid_size) {
  size_t dropped = 0;
  const CUptiResult dropped_result =
      cuptiActivityGetNumDroppedRecords(context, stream_id, &dropped);

  if (dropped_result == CUPTI_SUCCESS) {
    dropped_records_.fetch_add(dropped, std::memory_order_relaxed);
  } else if (!dropped_query_failed_.exchange(true, std::memory_order_relaxed)) {
    PrintCuptiError("Failed to query dropped activity records", dropped_result);
  }

  if (buffer == nullptr) return;

  /* Hand the buffer to the worker so that CUPTI's thread only enqueues it.
     A full queue means the worker is behind; parsing here then bounds the
     memory held by queued buffers, as before the worker existed. */
  {
    std::lock_guard<std::mutex> lock(completed_mutex_);
    if (worker_accepting_ && completed_buffers_.size() < MaxQueuedBuffers()) {
      completed_buffers_.push_back({buffer, valid_size});
      completed_ready_.notify_one();
      return;
    }
  }

  ParseBuffer(buffer, valid_size);
  ReleaseBuffer(buffer);
}

} /* namespace nvidia */
} /* namespace efimon */
