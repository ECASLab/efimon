/**
 * @file kernel-journal-testing.cpp
 * @brief Example and checks of the kernel journal and the active-time windows
 *
 * It writes kernel runs to a journal, merges them back in start order and
 * computes the active time of the kernels in fixed windows. It exits with a
 * non-zero code if any check fails.
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <sys/resource.h>
#include <unistd.h>

#include <algorithm>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <efimon/gpu/nvidia-gpu/kernel-journal.hpp>
#include <filesystem>  // NOLINT
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace efimon {
namespace nvidia {
namespace {

/** Number of failed checks */
int failures = 0;

/**
 * @brief Reports a failed check and counts it
 *
 * @param condition checked condition
 * @param expression text of the condition
 * @param line source line of the check
 */
void Check(bool condition, const char* expression, int line) {
  if (condition) return;
  std::fprintf(stderr,
               "kernel-journal-testing.cpp:%d: EFIMON_VERIFY(%s) failed\n",
               line, expression);
  ++failures;
}

/** Checks a condition and reports its text and line if it fails */
#define EFIMON_VERIFY(condition) Check((condition), #condition, __LINE__)

/**
 * @brief Builds a journal path in the temporary directory
 *
 * @param name file prefix
 * @return path unique to this process
 */
std::filesystem::path TemporaryPath(const char* name) {
  return std::filesystem::temp_directory_path() /
         (std::string(name) + "-" + std::to_string(getpid()) + ".journal");
}

/**
 * @brief Compares every field of two records
 *
 * @param left first record
 * @param right second record
 * @return true if all the fields are equal
 */
bool SameRecord(const KernelRecord& left, const KernelRecord& right) {
  return left.start_ns == right.start_ns && left.end_ns == right.end_ns &&
         left.device_id == right.device_id &&
         left.context_id == right.context_id &&
         left.stream_id == right.stream_id &&
         left.correlation_id == right.correlation_id &&
         left.name_id == right.name_id;
}

/**
 * @brief Window computation of the original collector, used as the reference
 *
 * @param kernels kernel records, in any order; must not be empty
 * @param window_ns window length
 * @return active nanoseconds of each window
 */
std::vector<uint64_t> ReferenceWindows(std::vector<KernelRecord> kernels,
                                       uint64_t window_ns) {
  std::stable_sort(kernels.begin(), kernels.end(),
                   [](const KernelRecord& left, const KernelRecord& right) {
                     return left.start_ns < right.start_ns;
                   });
  std::vector<std::pair<uint64_t, uint64_t>> intervals;
  for (const KernelRecord& kernel : kernels) {
    if (kernel.start_ns >= kernel.end_ns) continue;
    if (intervals.empty() || kernel.start_ns > intervals.back().second) {
      intervals.push_back({kernel.start_ns, kernel.end_ns});
    } else {
      intervals.back().second =
          std::max(intervals.back().second, kernel.end_ns);
    }
  }

  uint64_t first = kernels[0].start_ns;
  uint64_t last = kernels[0].end_ns;
  for (const KernelRecord& kernel : kernels) {
    first = std::min(first, kernel.start_ns);
    last = std::max(last, kernel.end_ns);
  }

  std::vector<uint64_t> windows;
  for (uint64_t start = first; start < last; start += window_ns) {
    const uint64_t end = start + window_ns;
    uint64_t active = 0;
    for (const auto& interval : intervals) {
      const uint64_t overlap_start = std::max(interval.first, start);
      const uint64_t overlap_end = std::min(interval.second, end);
      if (overlap_end > overlap_start) active += overlap_end - overlap_start;
    }
    windows.push_back(active);
  }
  return windows;
}

/**
 * @brief Generates random kernel records
 *
 * @param random random generator; must not be nullptr
 * @param count number of records
 * @return records with unique correlation ids
 */
std::vector<KernelRecord> RandomKernels(std::mt19937_64* random, size_t count) {
  std::uniform_int_distribution<uint64_t> start(1'000'000, 9'000'000);
  std::uniform_int_distribution<uint64_t> duration(0, 40'000);
  std::uniform_int_distribution<uint32_t> small(0, 7);
  std::vector<KernelRecord> kernels(count);
  for (size_t index = 0; index < count; ++index) {
    KernelRecord& kernel = kernels[index];
    /* Coarse starts create ties between runs. */
    kernel.start_ns = start(*random) / 1000 * 1000;
    kernel.end_ns = kernel.start_ns + duration(*random);
    kernel.device_id = small(*random);
    kernel.context_id = small(*random);
    kernel.stream_id = small(*random);
    kernel.correlation_id = static_cast<uint32_t>(index);
    kernel.name_id = small(*random);
  }
  return kernels;
}

/**
 * @brief Checks that runs of uneven sizes merge back in KernelRecordLess order
 */
void TestMergesRunsInStartOrder() {
  std::mt19937_64 random(2026);
  const std::vector<KernelRecord> kernels = RandomKernels(&random, 20'000);
  const std::filesystem::path path = TemporaryPath("kernel-journal-merge");

  KernelJournal journal;
  EFIMON_VERIFY(journal.Open(path, 1234));
  EFIMON_VERIFY(journal.AppendName(0, "kernel_a"));
  /* Runs of uneven sizes, including runs smaller than the merge buffer. */
  std::vector<KernelRecord> run;
  size_t next = 0;
  for (size_t run_size = 1; next < kernels.size();
       run_size = run_size * 3 % 997 + 1) {
    run.assign(kernels.begin() + next,
               kernels.begin() + std::min(kernels.size(), next + run_size));
    next += run.size();
    EFIMON_VERIFY(journal.AppendRun(&run));
    EFIMON_VERIFY(run.empty());
  }
  EFIMON_VERIFY(journal.record_count() == kernels.size());

  std::vector<KernelRecord> merged;
  EFIMON_VERIFY(journal.ForEachSorted(
      [&merged](const KernelRecord& kernel) { merged.push_back(kernel); }));

  std::vector<KernelRecord> expected = kernels;
  std::sort(expected.begin(), expected.end(), KernelRecordLess);
  EFIMON_VERIFY(merged.size() == expected.size());
  bool sorted = true;
  for (size_t index = 1; index < merged.size(); ++index) {
    sorted = sorted && merged[index - 1].start_ns <= merged[index].start_ns;
  }
  EFIMON_VERIFY(sorted);
  /* Correlation ids are unique here, so the merged order is fully determined,
     ties in start time included. */
  bool same = merged.size() == expected.size();
  for (size_t index = 0; same && index < merged.size(); ++index) {
    same = SameRecord(merged[index], expected[index]);
  }
  EFIMON_VERIFY(same);

  EFIMON_VERIFY(journal.Remove());
  EFIMON_VERIFY(!std::filesystem::exists(path));
}

/**
 * @brief Checks the bytes of the header, a name chunk and a run chunk
 */
void TestWritesDocumentedLayout() {
  const std::filesystem::path path = TemporaryPath("kernel-journal-layout");
  KernelJournal journal;
  EFIMON_VERIFY(journal.Open(path, 77));
  EFIMON_VERIFY(journal.AppendName(5, "gemm"));
  std::vector<KernelRecord> run(1);
  run[0].start_ns = 10;
  run[0].end_ns = 25;
  run[0].stream_id = 3;
  run[0].name_id = 5;
  EFIMON_VERIFY(journal.AppendRun(&run));
  EFIMON_VERIFY(journal.Flush());

  std::ifstream input(path, std::ios::binary);
  const std::vector<unsigned char> bytes(
      (std::istreambuf_iterator<char>(input)),
      std::istreambuf_iterator<char>());
  /* Header (16) + name chunk (1 + 4 + 4 + 4) + run chunk (1 + 4 + 36). */
  EFIMON_VERIFY(bytes.size() == 16 + 13 + 41);
  EFIMON_VERIFY(std::memcmp(bytes.data(), "NVPMKJ01", 8) == 0);
  EFIMON_VERIFY(bytes[16] == 'N');
  EFIMON_VERIFY(bytes[29] == 'R');
  uint64_t start = 0;
  std::memcpy(&start, bytes.data() + 34, sizeof(start));
  EFIMON_VERIFY(start == 10);

  EFIMON_VERIFY(journal.Remove());
}

/**
 * @brief Checks that a write error keeps the runs written before it
 */
void TestKeepsRunsWrittenBeforeAWriteError() {
  const std::filesystem::path path = TemporaryPath("kernel-journal-failure");
  constexpr size_t kRunRecords = 100;
  constexpr size_t kRunBytes =
      1 + 4 + kRunRecords * KernelJournal::kRecordBytes;

  KernelJournal journal;
  EFIMON_VERIFY(journal.Open(path, 1));

  /* Let the file grow by exactly two runs, then fail with EFBIG. */
  rlimit original{};
  EFIMON_VERIFY(getrlimit(RLIMIT_FSIZE, &original) == 0);
  rlimit limited = original;
  limited.rlim_cur = 16 + 2 * kRunBytes;
  void (*previous_handler)(int) = std::signal(SIGXFSZ, SIG_IGN);
  EFIMON_VERIFY(setrlimit(RLIMIT_FSIZE, &limited) == 0);

  std::vector<KernelRecord> run(kRunRecords);
  for (size_t index = 0; index < run.size(); ++index) {
    run[index].start_ns = 1000 - index;
    run[index].correlation_id = static_cast<uint32_t>(index);
  }
  std::vector<KernelRecord> copy = run;
  EFIMON_VERIFY(journal.AppendRun(&copy));
  copy = run;
  EFIMON_VERIFY(journal.AppendRun(&copy));
  copy = run;
  const bool third_appended = journal.AppendRun(&copy);
  EFIMON_VERIFY(!third_appended);
  /* The caller keeps the records of the failed run. */
  EFIMON_VERIFY(copy.size() == kRunRecords);
  copy = run;
  EFIMON_VERIFY(!journal.AppendRun(&copy));
  EFIMON_VERIFY(journal.failed());

  EFIMON_VERIFY(setrlimit(RLIMIT_FSIZE, &original) == 0);
  std::signal(SIGXFSZ, previous_handler);

  EFIMON_VERIFY(journal.record_count() == 2 * kRunRecords);
  size_t visited = 0;
  uint64_t previous_start = 0;
  bool sorted = true;
  EFIMON_VERIFY(journal.ForEachSorted([&](const KernelRecord& record) {
    sorted = sorted && previous_start <= record.start_ns;
    previous_start = record.start_ns;
    ++visited;
  }));
  EFIMON_VERIFY(visited == 2 * kRunRecords);
  EFIMON_VERIFY(sorted);
  EFIMON_VERIFY(journal.Remove());
}

/**
 * @brief Checks the merge of more runs than the merge buffer holds
 */
void TestMergesManySmallRuns() {
  /* More runs than the merge buffer holds at the largest read size. */
  constexpr size_t kRuns = 9'000;
  std::mt19937_64 random(5);
  const std::vector<KernelRecord> kernels = RandomKernels(&random, kRuns * 3);
  const std::filesystem::path path = TemporaryPath("kernel-journal-many");
  KernelJournal journal;
  EFIMON_VERIFY(journal.Open(path, 1));
  std::vector<KernelRecord> run;
  for (size_t next = 0; next < kernels.size(); next += 3) {
    run.assign(kernels.begin() + next, kernels.begin() + next + 3);
    EFIMON_VERIFY(journal.AppendRun(&run));
  }
  std::vector<KernelRecord> expected = kernels;
  std::sort(expected.begin(), expected.end(), KernelRecordLess);
  std::vector<KernelRecord> merged;
  EFIMON_VERIFY(journal.ForEachSorted(
      [&merged](const KernelRecord& record) { merged.push_back(record); }));
  EFIMON_VERIFY(merged.size() == expected.size());
  EFIMON_VERIFY(
      std::equal(merged.begin(), merged.end(), expected.begin(),
                 [](const KernelRecord& left, const KernelRecord& right) {
                   return left.start_ns == right.start_ns &&
                          left.correlation_id == right.correlation_id;
                 }));
  EFIMON_VERIFY(journal.Remove());
}

/**
 * @brief Checks that the CSV rows match the std::ostream output
 */
void TestCsvRowMatchesStreamFormatting() {
  std::mt19937_64 random(11);
  std::vector<KernelRecord> records = RandomKernels(&random, 2'000);
  KernelRecord extreme;
  extreme.start_ns = std::numeric_limits<uint64_t>::max();
  extreme.end_ns = 0; /* Unsigned duration wraps, as before. */
  extreme.device_id = std::numeric_limits<uint32_t>::max();
  records.push_back(extreme);
  records.push_back(KernelRecord{});

  const std::string quoted = "\"gemm \"\"tile\"\"\"";
  std::string fast;
  std::ostringstream reference;
  for (size_t index = 0; index < records.size(); ++index) {
    const KernelRecord& record = records[index];
    const int pid = static_cast<int>(index % 3 == 0 ? 1 : 4194304);
    AppendKernelCsvRow(index, pid, record, quoted, &fast);
    /* The stream-based row writer this replaces. */
    reference << index << ',' << pid << ',' << record.device_id << ','
              << record.context_id << ',' << record.stream_id << ','
              << record.correlation_id << ',' << record.start_ns << ','
              << record.end_ns << ',' << (record.end_ns - record.start_ns)
              << ',' << quoted << '\n';
  }
  EFIMON_VERIFY(fast == reference.str());
}

/**
 * @brief Checks that an empty journal visits no record
 */
void TestEmptyJournalVisitsNothing() {
  const std::filesystem::path path = TemporaryPath("kernel-journal-empty");
  KernelJournal journal;
  EFIMON_VERIFY(journal.Open(path, 1));
  size_t visited = 0;
  EFIMON_VERIFY(
      journal.ForEachSorted([&visited](const KernelRecord&) { ++visited; }));
  EFIMON_VERIFY(visited == 0);
  EFIMON_VERIFY(journal.Remove());
}

/**
 * @brief Checks the windows against the original computation
 */
void TestWindowsMatchOriginalComputation() {
  std::mt19937_64 random(7);
  for (const uint64_t window_ns : {1'000ULL, 50'000ULL, 200'000ULL}) {
    std::vector<KernelRecord> kernels = RandomKernels(&random, 3'000);
    const std::vector<uint64_t> expected = ReferenceWindows(kernels, window_ns);

    std::stable_sort(kernels.begin(), kernels.end(),
                     [](const KernelRecord& left, const KernelRecord& right) {
                       return left.start_ns < right.start_ns;
                     });
    ActiveTimeWindows windows(window_ns);
    for (const KernelRecord& kernel : kernels) {
      windows.Add(kernel.start_ns, kernel.end_ns);
    }
    windows.Finish();
    EFIMON_VERIFY(windows.ActiveNs() == expected);
  }
}

/**
 * @brief Checks that invalid timestamps do not move the first window
 */
void TestWindowsIgnoreInvalidTimestamps() {
  /* An incomplete record from a forced flush, then a valid kernel late on. */
  ActiveTimeWindows windows(200'000'000);
  windows.Add(0, 0);
  windows.Add(1'000'000'000'000, 1'000'000'100'000);
  windows.Add(1'000'000'200'000, 5);
  windows.Finish();
  EFIMON_VERIFY(windows.first_start_ns() == 1'000'000'000'000);
  EFIMON_VERIFY(windows.ActiveNs() == std::vector<uint64_t>{100'000});
}

/**
 * @brief Checks the windows when no kernel was added
 */
void TestWindowsWithoutKernels() {
  ActiveTimeWindows windows(200);
  windows.Finish();
  EFIMON_VERIFY(windows.empty());
  EFIMON_VERIFY(windows.ActiveNs().empty());
}

} /* namespace */
} /* namespace nvidia */
} /* namespace efimon */

int main() {
  efimon::nvidia::TestMergesRunsInStartOrder();
  efimon::nvidia::TestWritesDocumentedLayout();
  efimon::nvidia::TestKeepsRunsWrittenBeforeAWriteError();
  efimon::nvidia::TestMergesManySmallRuns();
  efimon::nvidia::TestCsvRowMatchesStreamFormatting();
  efimon::nvidia::TestEmptyJournalVisitsNothing();
  efimon::nvidia::TestWindowsMatchOriginalComputation();
  efimon::nvidia::TestWindowsIgnoreInvalidTimestamps();
  efimon::nvidia::TestWindowsWithoutKernels();
  if (efimon::nvidia::failures != 0) {
    std::fprintf(stderr, "%d checks failed\n", efimon::nvidia::failures);
    return 1;
  }
  std::printf("kernel-journal-testing: all checks passed\n");
  return 0;
}
