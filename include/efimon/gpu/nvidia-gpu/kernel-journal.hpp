/**
 * @file kernel-journal.hpp
 * @brief Binary journal of CUPTI kernel records and kernel active-time windows
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_KERNEL_JOURNAL_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_KERNEL_JOURNAL_HPP_

#include <cstdint>
#include <cstdio>
#include <filesystem>  // NOLINT
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace efimon {
namespace nvidia {

/**
 * @brief Fields of one CUPTI concurrent-kernel record
 *
 * The name_id indexes the kernel names that the journal stores separately.
 */
struct KernelRecord {
  /** Kernel start, in CUPTI nanoseconds */
  uint64_t start_ns = 0;
  /** Kernel end, in CUPTI nanoseconds */
  uint64_t end_ns = 0;
  /** CUDA device id */
  uint32_t device_id = 0;
  /** CUDA context id */
  uint32_t context_id = 0;
  /** CUDA stream id */
  uint32_t stream_id = 0;
  /** CUPTI correlation id, which follows the launch order */
  uint32_t correlation_id = 0;
  /** Index of the kernel name */
  uint32_t name_id = 0;
};

/**
 * @brief Orders kernel records by start time, then by correlation id
 *
 * The correlation id follows the launch order, so records with the same start
 * always come out in the same order, whatever the order in which CUPTI
 * delivered them.
 *
 * @param left first record
 * @param right second record
 * @return true if left goes before right
 */
bool KernelRecordLess(const KernelRecord& left, const KernelRecord& right);

/**
 * @brief Binary journal of the kernel records of a profiled process
 *
 * It appends kernel names and runs of kernel records to a binary file while
 * the profiled process runs, then merges the runs back in start order.
 *
 * File layout, little-endian: the 8-byte magic "NVPMKJ01", a uint32 format
 * version and the int32 PID; then chunks tagged 'N' (uint32 id, uint32 length,
 * name bytes) or 'R' (uint32 count, then count records of kRecordBytes with
 * start, end, device, context, stream, correlation and name id). Every run is
 * sorted with KernelRecordLess. The journal recovery of efimon-analyser reads
 * the same layout.
 *
 * Not thread-safe: callers serialize every call.
 */
class KernelJournal {
 public:
  /** Version written in the header */
  static constexpr uint32_t kFormatVersion = 1;
  /** Size of one encoded record */
  static constexpr size_t kRecordBytes = 36;

  /**
   * @brief Construct a new, closed Kernel Journal object
   */
  KernelJournal() = default;

  /**
   * @brief Destroy the Kernel Journal object and close its file
   */
  ~KernelJournal();

  KernelJournal(const KernelJournal&) = delete;
  KernelJournal& operator=(const KernelJournal&) = delete;

  /**
   * @brief Creates or truncates the journal and writes its header
   *
   * @param path journal file
   * @param pid PID of the profiled process, stored in the header
   * @return false if the file cannot be written
   */
  bool Open(const std::filesystem::path& path, int32_t pid);

  /**
   * @brief Records the name of a kernel id
   *
   * @param name_id kernel id
   * @param name kernel name
   * @return false after a write error
   */
  bool AppendName(uint32_t name_id, std::string_view name);

  /**
   * @brief Appends the records as one sorted run and flushes the file
   *
   * The records are sorted with KernelRecordLess first. On success, records is
   * cleared and keeps its capacity for reuse. After a write error, records is
   * unchanged apart from its order and every later call also fails.
   *
   * @param records records of the run; must not be nullptr
   * @return false after a write error
   */
  bool AppendRun(std::vector<KernelRecord>* records);

  /**
   * @brief Flushes the buffered chunks to the file
   *
   * @return false after a write error
   */
  bool Flush();

  /**
   * @brief Visits every record of the written runs in start order
   *
   * It includes the runs written before a write error. The read buffers of
   * all runs take 4 MiB, or 16 records per run beyond 7,281 runs.
   *
   * @param visit function called once per record, in KernelRecordLess order
   * @return false if the journal cannot be read
   */
  bool ForEachSorted(const std::function<void(const KernelRecord&)>& visit);

  /**
   * @brief Closes and deletes the journal file
   *
   * @return false if the file cannot be removed
   */
  bool Remove();

  /**
   * @brief Checks if the journal file is open
   *
   * @return true if Open() succeeded and the file is not closed yet
   */
  bool is_open() const { return file_ != nullptr; }

  /**
   * @brief Checks if a write error happened
   *
   * @return true after a write error
   */
  bool failed() const { return failed_; }

  /**
   * @brief Gets the number of records written in runs
   *
   * @return number of records
   */
  uint64_t record_count() const { return record_count_; }

  /**
   * @brief Gets the journal file
   *
   * @return path given to Open()
   */
  const std::filesystem::path& path() const { return path_; }

 private:
  /** Position of one run of records in the file */
  struct Run {
    /** Offset of the first record */
    uint64_t offset;
    /** Number of records */
    uint64_t count;
  };

  bool Write(const void* data, size_t size);
  void Close();

  /** Open journal file; nullptr when closed */
  std::FILE* file_ = nullptr;
  /** Journal file */
  std::filesystem::path path_;
  /** Runs written so far */
  std::vector<Run> runs_;
  /** Reusable encoding buffer */
  std::vector<unsigned char> encoded_;
  /** Bytes written so far */
  uint64_t offset_ = 0;
  /** Records written in runs */
  uint64_t record_count_ = 0;
  /** Set after the first write error */
  bool failed_ = false;
};

/**
 * @brief Appends one kernel_activity.csv row
 *
 * The row holds the event id, PID, device, context, stream and correlation
 * ids, start, end, duration and the kernel name. The text matches what
 * std::ostream writes for the same values.
 *
 * @param event_id row number
 * @param pid profiled process
 * @param record kernel record
 * @param quoted_name kernel name, already quoted for CSV
 * @param out string that receives the row; must not be nullptr
 */
void AppendKernelCsvRow(uint64_t event_id, int64_t pid,
                        const KernelRecord& record,
                        std::string_view quoted_name, std::string* out);

/**
 * @brief Active time of the kernels in fixed windows
 *
 * It accumulates the union of kernel intervals into fixed windows that start
 * at the first kernel, without storing the kernels. Records must arrive in
 * start order. Intervals that touch are merged, and empty intervals are
 * ignored. Records with a zero start or an end before the start are not
 * counted, so they cannot move the first window.
 */
class ActiveTimeWindows {
 public:
  /**
   * @brief Construct a new Active Time Windows object
   *
   * @param window_ns window length in nanoseconds; 0 is taken as 1
   */
  explicit ActiveTimeWindows(uint64_t window_ns);

  /**
   * @brief Adds one kernel interval
   *
   * @param start_ns kernel start
   * @param end_ns kernel end
   */
  void Add(uint64_t start_ns, uint64_t end_ns);

  /**
   * @brief Adds the last merged interval
   *
   * Call it once after the final Add().
   */
  void Finish();

  /**
   * @brief Checks if no valid kernel was added
   *
   * @return true if no kernel was counted
   */
  bool empty() const { return kernel_count_ == 0; }

  /**
   * @brief Gets the start of the first window
   *
   * @return start of the first counted kernel
   */
  uint64_t first_start_ns() const { return first_start_ns_; }

  /**
   * @brief Gets the latest kernel end
   *
   * @return latest end of the counted kernels
   */
  uint64_t last_end_ns() const { return last_end_ns_; }

  /**
   * @brief Gets the window length
   *
   * @return window length in nanoseconds
   */
  uint64_t window_ns() const { return window_ns_; }

  /**
   * @brief Gets the active time of each window
   *
   * @return active nanoseconds of each window from first_start_ns() until the
   * last window that starts before last_end_ns()
   */
  std::vector<uint64_t> ActiveNs() const;

 private:
  void AddMergedInterval(uint64_t start_ns, uint64_t end_ns);

  /** Window length */
  uint64_t window_ns_;
  /** Counted kernels */
  uint64_t kernel_count_ = 0;
  /** Start of the first counted kernel */
  uint64_t first_start_ns_ = 0;
  /** Latest end of the counted kernels */
  uint64_t last_end_ns_ = 0;
  /** True while an interval is being merged */
  bool has_interval_ = false;
  /** Start of the interval being merged */
  uint64_t interval_start_ns_ = 0;
  /** End of the interval being merged */
  uint64_t interval_end_ns_ = 0;
  /** Active time of each window filled so far */
  std::vector<uint64_t> active_ns_;
};

} /* namespace nvidia */
} /* namespace efimon */

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_KERNEL_JOURNAL_HPP_
