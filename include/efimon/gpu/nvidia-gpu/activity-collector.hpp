/**
 * @file activity-collector.hpp
 * @brief Collection of the CUPTI kernel and memory copy activity of the
 * profiled process
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_ACTIVITY_COLLECTOR_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_ACTIVITY_COLLECTOR_HPP_

#include <cuda.h>
#include <sys/types.h>

#include <atomic>
#include <condition_variable>  // NOLINT
#include <cstddef>
#include <cstdint>
#include <deque>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-options.hpp>
#include <filesystem>  // NOLINT
#include <functional>
#include <mutex>  // NOLINT
#include <string>
#include <string_view>
#include <thread>  // NOLINT
#include <unordered_map>
#include <vector>

namespace efimon {
namespace nvidia {

/**
 * @brief Fields of one CUPTI memory copy record
 */
struct MemcpyEvent {
  /** Copy start, in CUPTI nanoseconds */
  uint64_t start;
  /** Copy end, in CUPTI nanoseconds */
  uint64_t end;
  /** Copied bytes */
  uint64_t bytes;
  /** CUDA device id */
  uint32_t device_id;
  /** CUDA context id */
  uint32_t context_id;
  /** CUDA stream id */
  uint32_t stream_id;
  /** CUPTI correlation id */
  uint32_t correlation_id;
  /** CUpti_ActivityMemcpyKind of the copy */
  uint8_t copy_kind;
};

/**
 * @brief Collects the CUPTI kernel and memory copy activity of the process
 *
 * It hands CUPTI the activity buffers, parses the completed ones in a worker
 * thread (or in the CUPTI callback when the worker is disabled or behind) and
 * keeps the kernels in the binary journal or in memory, and the memory copies
 * in memory. The CUPTI callbacks reach the collector through a process-wide
 * pointer, so only one collector may be started. Configure(), OpenJournal()
 * and Start() run before any CUDA activity; the accessors of the collected
 * data are valid after Stop().
 */
class ActivityCollector {
 public:
  /** Outcome of Start() */
  enum class StartResult {
    kStarted,       /** Callbacks registered and activity enabled */
    kNotRegistered, /** CUPTI rejected the callbacks */
    kNotEnabled,    /** Callbacks registered, but an activity kind failed */
  };

  /**
   * @brief Construct a new Activity Collector object, not started
   */
  ActivityCollector() = default;

  ActivityCollector(const ActivityCollector&) = delete;
  ActivityCollector& operator=(const ActivityCollector&) = delete;

  /**
   * @brief Takes the activity switches of the profiler
   *
   * @param options profiler options; activity_worker, activity_pool_buffers,
   * activity_flush_ms, kernel_journal and memcpy_csv are used
   * @param directory output directory of the journal
   */
  void Configure(const ProfilerOptions& options,
                 const std::filesystem::path& directory);

  /**
   * @brief Keeps only the records of one device
   *
   * Without this call, the records of every device are kept.
   *
   * @param device_id CUDA device ordinal, as in CUPTI's deviceId
   */
  void set_measured_device(int64_t device_id) {
    measured_device_id_ = device_id;
  }

  /**
   * @brief Opens kernel_activity.journal when the journal is enabled
   *
   * If the journal cannot be opened, the kernels are kept in memory.
   *
   * @param pid profiled process, stored in the journal header
   */
  void OpenJournal(pid_t pid);

  /**
   * @brief Preallocates the buffers, starts the worker and enables CUPTI
   * kernel and memory copy activity
   *
   * Errors are reported on stderr.
   *
   * @return StartResult::kStarted on success
   */
  StartResult Start();

  /**
   * @brief Disables the activity, flushes CUPTI, stops the worker and closes
   * the collection
   *
   * Buffers that CUPTI returns afterwards are only counted as late.
   *
   * @return false if the final CUPTI flush failed
   */
  bool Stop();

  /**
   * @brief Visits every kernel in KernelRecordLess order
   *
   * It merges the journal runs with the kernels kept in memory.
   *
   * @param visit function called once per kernel
   * @return false if the journal cannot be read
   */
  bool VisitKernelsSorted(
      const std::function<void(const KernelRecord&)>& visit);

  /**
   * @brief Sorts the memory copies by start time
   */
  void SortMemcpyEvents();

  /**
   * @brief Keeps or deletes the journal after the CSVs were written
   *
   * The sorted CSV supersedes the journal, which is kept only if the CSVs
   * failed.
   *
   * @param csv_written whether the activity CSVs were written
   */
  void FinishJournal(bool csv_written);

  /**
   * @brief Gets a kernel name quoted for CSV
   *
   * @param name_id id from KernelRecord::name_id
   * @return quoted name
   */
  const std::string& escaped_name(uint32_t name_id) const {
    return escaped_kernel_names_[name_id];
  }

  /**
   * @brief Gets the memory copies
   *
   * @return copies, sorted after SortMemcpyEvents()
   */
  const std::deque<MemcpyEvent>& memcpy_events() const {
    return memcpy_events_;
  }

  /**
   * @brief Gets the state of the journal for capture_summary.txt
   *
   * @return "failed", "used" or "disabled"
   */
  const char* journal_state() const;

  /**
   * @brief Checks if records are missing from the CSVs
   *
   * @return true if CUPTI dropped records, or records were incomplete or
   * late
   */
  bool partial() const;

  /**
   * @brief Checks if an activity buffer or record failed
   *
   * @return true after an allocation or record error
   */
  bool failed() const { return failed_.load(std::memory_order_acquire); }

  /**
   * @brief Gets the number of distinct kernel names
   *
   * @return names seen so far
   */
  size_t kernel_name_count() const { return kernel_names_.size(); }

  /**
   * @brief Gets the records that CUPTI dropped
   *
   * @return dropped records
   */
  size_t dropped_records() const {
    return dropped_records_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Gets the incomplete records left out of the CSVs
   *
   * @return records with zero or out-of-order timestamps
   */
  size_t invalid_records() const {
    return invalid_records_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Gets the records of other devices left out of the CSVs
   *
   * @return records of devices other than the measured one
   */
  size_t other_device_records() const {
    return other_device_records_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Gets the records that arrived after Stop()
   *
   * @return late records
   */
  size_t late_records() const {
    return late_records_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Gets the buffers allocated because the pool was empty
   *
   * @return buffers allocated beyond the pool
   */
  size_t pool_misses() const {
    return pool_misses_.load(std::memory_order_relaxed);
  }

  /**
   * @brief Gets the CPU time of the worker thread
   *
   * @return milliseconds; read it after Stop()
   */
  double thread_cpu_ms() const { return thread_cpu_ms_; }

  /**
   * @brief Gets the bytes copied from host to device
   *
   * @return bytes
   */
  uint64_t bytes_h2d() const { return bytes_h2d_; }

  /**
   * @brief Gets the bytes copied from device to host
   *
   * @return bytes
   */
  uint64_t bytes_d2h() const { return bytes_d2h_; }

  /**
   * @brief Gets the bytes copied from device to device
   *
   * @return bytes
   */
  uint64_t bytes_d2d() const { return bytes_d2d_; }

  /**
   * @brief Hands CUPTI an activity buffer; called by the CUPTI callback
   *
   * @param buffer receives the buffer, or nullptr if none can be allocated;
   * must not be nullptr
   * @param size receives the buffer size; must not be nullptr
   * @param max_records receives 0, no record limit; must not be nullptr
   */
  void RequestBuffer(uint8_t** buffer, size_t* size, size_t* max_records);

  /**
   * @brief Takes a completed activity buffer; called by the CUPTI callback
   *
   * @param context CUDA context of the buffer
   * @param stream_id stream of the buffer
   * @param buffer completed buffer, owned by the collector from now on; may
   * be nullptr
   * @param valid_size bytes of valid records
   */
  void CompleteBuffer(CUcontext context, uint32_t stream_id, uint8_t* buffer,
                      size_t valid_size);

 private:
  /** Completed buffer waiting for the worker */
  struct CompletedBuffer {
    /** Activity buffer */
    uint8_t* buffer;
    /** Bytes of valid records */
    size_t valid_size;
  };

  void PreallocateBuffers();
  void ReleaseBuffer(uint8_t* buffer);
  void ReleaseBufferPool();
  uint32_t InternKernelName(const char* name);
  bool KeepRecord(uint32_t device_id, uint64_t start_ns, uint64_t end_ns);
  void ParseBuffer(uint8_t* buffer, size_t valid_size);
  void ProcessCompletedBuffers();
  void StartWorker();
  size_t MaxQueuedBuffers() const;
  void StopWorker();

  /** Copy of the profiler options; kernel_journal is cleared when the
      journal cannot be written */
  ProfilerOptions options_;
  /** Path of kernel_activity.journal */
  std::filesystem::path journal_path_;
  /** CUDA device whose records are kept; -1 keeps every device, also when
      the selected device does not exist */
  int64_t measured_device_id_ = -1;
  /** True once CUPTI accepted the callbacks. Another CUPTI client, such as
      Nsight Systems or compute-sanitizer, makes the registration fail; the
      activity API must then not be used again */
  bool registered_ = false;
  /** Set after an allocation or record error */
  std::atomic<bool> failed_{false};

  /** Guards the kernels, names, journal and closed_ */
  std::mutex mutex_;
  /** Kernels kept in memory when the journal is disabled or cannot be
      written. A deque never copies existing records when it grows */
  std::deque<KernelRecord> kernels_;
  /** Kernel names; deque elements stay in place, so kernel_name_ids_ can
      view their characters */
  std::deque<std::string> kernel_names_;
  /** Kernel names quoted for CSV */
  std::vector<std::string> escaped_kernel_names_;
  /** Id of each kernel name */
  std::unordered_map<std::string_view, uint32_t> kernel_name_ids_;
  /** Memory copies */
  std::deque<MemcpyEvent> memcpy_events_;
  /** Kernels of the buffer being parsed */
  std::vector<KernelRecord> parsed_kernels_;
  /** Journal that receives the parsed kernels */
  KernelJournal journal_;
  /** Set once Stop() closes the collection. Later buffers are only counted,
      so the export reads the containers without the lock */
  bool closed_ = false;
  /** Bytes copied from host to device */
  uint64_t bytes_h2d_ = 0;
  /** Bytes copied from device to host */
  uint64_t bytes_d2h_ = 0;
  /** Bytes copied from device to device */
  uint64_t bytes_d2d_ = 0;

  /** Records that CUPTI dropped */
  std::atomic<size_t> dropped_records_{0};
  /** Set once a failed query of dropped records was reported */
  std::atomic<bool> dropped_query_failed_{false};
  /** Set once a record error was reported */
  std::atomic<bool> record_error_reported_{false};
  /** Set once an allocation error was reported */
  std::atomic<bool> allocation_error_reported_{false};
  /** Incomplete records: zero or out-of-order timestamps */
  std::atomic<size_t> invalid_records_{0};
  /** Records of a device other than measured_device_id_ */
  std::atomic<size_t> other_device_records_{0};
  /** Records that arrived after Stop() */
  std::atomic<size_t> late_records_{0};

  /** Guards the buffer pool */
  std::mutex pool_mutex_;
  /** Every pooled buffer */
  std::vector<uint8_t*> pooled_buffers_;
  /** Pooled buffers that CUPTI does not currently hold */
  std::vector<uint8_t*> free_buffers_;
  /** Buffers allocated because the pool was empty */
  std::atomic<size_t> pool_misses_{0};

  /** Guards the completed buffers and worker_accepting_ */
  std::mutex completed_mutex_;
  /** Signals new completed buffers or the end of the worker */
  std::condition_variable completed_ready_;
  /** Buffers handed over by CompleteBuffer() for the worker */
  std::deque<CompletedBuffer> completed_buffers_;
  /** True while the worker takes buffers */
  bool worker_accepting_ = false;
  /** Worker thread; nullptr when not running */
  std::thread* worker_ = nullptr;
  /** CPU time of the worker thread */
  double thread_cpu_ms_ = 0.0;
};

} /* namespace nvidia */
} /* namespace efimon */

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_ACTIVITY_COLLECTOR_HPP_
