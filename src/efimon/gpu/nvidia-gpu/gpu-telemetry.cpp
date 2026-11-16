/**
 * @file gpu-telemetry.cpp
 * @brief NVML telemetry of the measured GPU: power, energy, temperature,
 * utilization and clock, sampled every 50 ms in a dedicated thread
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#include <cupti.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>  // NOLINT
#include <cstdint>
#include <cstdio>
#include <efimon/gpu/nvidia-gpu/gpu-telemetry.hpp>
#include <efimon/gpu/nvidia-gpu/profiler-utils.hpp>
#include <exception>
#include <filesystem>  // NOLINT
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

namespace efimon {
namespace nvidia {
namespace {

/** Period of the telemetry rows */
constexpr auto kSampleInterval = std::chrono::milliseconds(50);
/** Period of the flushes that keep the rows of an abnormal exit */
constexpr auto kFlushInterval = std::chrono::seconds(1);

/**
 * @brief Reads an unsigned NVML sample value
 *
 * @param type value type reported by NVML
 * @param value sample value
 * @param result receives the value; must not be nullptr
 * @return false if the type is not unsigned
 */
bool SampleValueAsUnsigned(nvmlValueType_t type, const nvmlValue_t& value,
                           uint64_t* result) {
  switch (type) {
    case NVML_VALUE_TYPE_UNSIGNED_INT:
      *result = value.uiVal;
      return true;
    case NVML_VALUE_TYPE_UNSIGNED_LONG:
      *result = value.ulVal;
      return true;
    case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
      *result = value.ullVal;
      return true;
    default:
      return false;
  }
}

} /* namespace */

void GpuTelemetry::Configure(const ProfilerOptions& options) {
  options_ = options;
}

bool GpuTelemetry::OpenTelemetryFile(const std::filesystem::path& directory) {
  telemetry_path_ = directory / "gpu_telemetry.csv";
  power_samples_path_ = directory / "gpu_power_samples.csv";
  telemetry_file_.open(telemetry_path_);
  if (!telemetry_file_.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 telemetry_path_.c_str());
    return false;
  }
  telemetry_file_
      << "timestamp_ns,query_start_ns,query_end_ns,power_mw,temperature_c,"
         "gpu_utilization_pct,"
         "memory_utilization_pct,graphics_clock_mhz,"
         "energy_mj\n";
  return true;
}

void GpuTelemetry::OpenPowerSamplesFile() {
  if (!options_.power_samples) return;
  power_samples_file_.open(power_samples_path_);
  if (!power_samples_file_.is_open()) {
    std::fprintf(stderr, "[Profiler] Cannot open CSV file %s\n",
                 power_samples_path_.c_str());
    options_.power_samples = false;
  } else {
    power_samples_file_ << "nvml_timestamp_us,timestamp_ns,power_mw\n";
  }
}

bool GpuTelemetry::InitializeDevice(int device_index,
                                    const std::string& pci_bus_id) {
  nvmlReturn_t res = nvmlInit_v2();
  if (NVML_SUCCESS != res) {
    std::fprintf(stderr, "[NVML] Cannot initialize NVML: %s\n",
                 nvmlErrorString(res));
    return false;
  }

  res = nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id.c_str(), &device_);
  if (NVML_SUCCESS != res) {
    std::fprintf(stderr,
                 "[NVML] Cannot match CUDA device %d at PCI bus %s: %s\n",
                 device_index, pci_bus_id.c_str(), nvmlErrorString(res));
    nvmlShutdown();
    return false;
  }

  initialized_ = true;
  ResolveTemperatureV();
  std::printf("CUDA device: %d; NVML PCI device: %s\n", device_index,
              pci_bus_id.c_str());
  return true;
}

void GpuTelemetry::Start() {
  if (!initialized_ || !telemetry_file_.is_open() || thread_ != nullptr) {
    return;
  }

  running_.store(true, std::memory_order_release);

  /* std::thread reports a failure to start only with an exception; catching
     it keeps the profiled process alive without telemetry. */
  try {
    thread_ = new std::thread(&GpuTelemetry::SampleLoop, this);
  } catch (const std::exception& error) {
    running_.store(false, std::memory_order_release);

    std::fprintf(stderr, "[NVML] Failed to start local sampling: %s\n",
                 error.what());
  }
}

void GpuTelemetry::Stop() {
  running_.store(false, std::memory_order_release);

  if (thread_ == nullptr) return;

  if (thread_->joinable()) thread_->join();

  delete thread_;
  thread_ = nullptr;
}

bool GpuTelemetry::FinishFiles() {
  bool success = true;
  if (telemetry_file_.is_open() &&
      !FinishCsvFile(&telemetry_file_, telemetry_path_)) {
    success = false;
  }
  if (power_samples_file_.is_open() &&
      !FinishCsvFile(&power_samples_file_, power_samples_path_)) {
    success = false;
  }
  return success;
}

uint64_t GpuTelemetry::ProcessMemory(pid_t pid) const {
  if (!initialized_) return 0;

  unsigned int count = 0;

  nvmlReturn_t res =
      nvmlDeviceGetComputeRunningProcesses_v3(device_, &count, nullptr);

  if (NVML_ERROR_INSUFFICIENT_SIZE != res && NVML_SUCCESS != res) {
    return 0;
  }

  if (count == 0) return 0;

  std::vector<nvmlProcessInfo_t> processes(count);

  res = nvmlDeviceGetComputeRunningProcesses_v3(device_, &count,
                                                processes.data());

  if (NVML_SUCCESS != res) return 0;

  for (const auto& process : processes) {
    if (process.pid == static_cast<unsigned int>(pid)) {
      return process.usedGpuMemory;
    }
  }

  return 0;
}

void GpuTelemetry::Shutdown() {
  if (initialized_) nvmlShutdown();
}

/* Reads the energy counter in millijoules since the driver loaded. A device
   that does not support it is disabled so that later samples skip the call. */
std::optional<uint64_t> GpuTelemetry::ReadEnergyCounter(
    bool* enabled, bool* error_reported) const {
  if (!*enabled) return std::nullopt;

  unsigned long long energy_mj = 0;  // NOLINT(runtime/int): NVML API type.
  const nvmlReturn_t result =
      nvmlDeviceGetTotalEnergyConsumption(device_, &energy_mj);
  if (result == NVML_SUCCESS) return static_cast<uint64_t>(energy_mj);

  if (!*error_reported) {
    std::fprintf(stderr, "[NVML] Cannot sample total energy: %s\n",
                 nvmlErrorString(result));
    *error_reported = true;
  }
  if (result == NVML_ERROR_NOT_SUPPORTED) *enabled = false;
  return std::nullopt;
}

/* nvmlDeviceGetTemperature is deprecated in NVML 13. Its replacement,
   nvmlDeviceGetTemperatureV, is declared from the NVML 12.8 headers on, but
   the driver provides the symbol, so it is resolved at run time and an older
   driver keeps using the original call. */
void GpuTelemetry::ResolveTemperatureV() {
#ifdef nvmlTemperature_v1
  get_temperature_v_ =
      reinterpret_cast<nvmlReturn_t (*)(nvmlDevice_t, nvmlTemperature_t*)>(
          dlsym(RTLD_DEFAULT, "nvmlDeviceGetTemperatureV"));
#endif
}

nvmlReturn_t GpuTelemetry::ReadTemperature(unsigned int* temperature_c) const {
#ifdef nvmlTemperature_v1
  if (get_temperature_v_ != nullptr) {
    nvmlTemperature_t temperature{};
    temperature.version = nvmlTemperature_v1;
    temperature.sensorType = NVML_TEMPERATURE_GPU;
    const nvmlReturn_t result = get_temperature_v_(device_, &temperature);
    if (result != NVML_SUCCESS) return result;
    /* The CSV column is unsigned, as with the original call. */
    if (temperature.temperature < 0) return NVML_ERROR_UNKNOWN;
    *temperature_c = static_cast<unsigned int>(temperature.temperature);
    return NVML_SUCCESS;
  }
#endif
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  return nvmlDeviceGetTemperature(device_, NVML_TEMPERATURE_GPU, temperature_c);
#pragma GCC diagnostic pop
}

/* Sizes the sample buffer and skips the samples taken before the profiler
   started. A device without power samples disables the reader. */
void GpuTelemetry::InitializePowerSamples(PowerSampleReader* reader) const {
  reader->enabled = options_.power_samples;
  if (!reader->enabled) return;

  nvmlValueType_t type{};
  unsigned int count = 0;
  const nvmlReturn_t result = nvmlDeviceGetSamples(
      device_, NVML_TOTAL_POWER_SAMPLES, 0, &type, &count, nullptr);
  if (result != NVML_SUCCESS || count == 0) {
    std::fprintf(stderr, "[NVML] Cannot read buffered power samples: %s\n",
                 result != NVML_SUCCESS ? nvmlErrorString(result)
                                        : "empty sample buffer");
    reader->enabled = false;
    return;
  }
  reader->samples.resize(count);
  reader->last_seen_us = static_cast<unsigned long long>(  // NOLINT
      std::chrono::duration_cast<std::chrono::microseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
}

/* Appends the power samples that the driver buffered since the last call. */
void GpuTelemetry::ReadPowerSamples(PowerSampleReader* reader) {
  if (!reader->enabled) return;

  nvmlValueType_t type{};
  auto count = static_cast<unsigned int>(reader->samples.size());
  const nvmlReturn_t result = nvmlDeviceGetSamples(
      device_, NVML_TOTAL_POWER_SAMPLES, reader->last_seen_us, &type, &count,
      reader->samples.data());
  /* NOT_FOUND means that the driver has no newer sample yet. */
  if (result == NVML_ERROR_NOT_FOUND) return;
  if (result != NVML_SUCCESS) {
    if (!reader->error_reported) {
      std::fprintf(stderr, "[NVML] Cannot read buffered power samples: %s\n",
                   nvmlErrorString(result));
      reader->error_reported = true;
    }
    return;
  }

  for (unsigned int index = 0; index < count; ++index) {
    const nvmlSample_t& sample = reader->samples[index];
    uint64_t power_mw = 0;
    if (sample.timeStamp <= reader->last_seen_us ||
        !SampleValueAsUnsigned(type, sample.sampleValue, &power_mw)) {
      continue;
    }
    ++power_sample_rows_;
    power_samples_file_ << sample.timeStamp << ',' << sample.timeStamp * 1000
                        << ',' << power_mw << '\n';
  }
  for (unsigned int index = 0; index < count; ++index) {
    reader->last_seen_us =
        std::max(reader->last_seen_us, reader->samples[index].timeStamp);
  }
}

void GpuTelemetry::WriteRow(const Sample& sample) {
  telemetry_file_ << sample.timestamp_ns << ',' << sample.query_start_ns << ','
                  << sample.query_end_ns << ',';
  if (sample.power_mw) telemetry_file_ << *sample.power_mw;
  telemetry_file_ << ',';
  if (sample.temperature_c) telemetry_file_ << *sample.temperature_c;
  telemetry_file_ << ',';
  if (sample.gpu_utilization) telemetry_file_ << *sample.gpu_utilization;
  telemetry_file_ << ',';
  if (sample.memory_utilization) telemetry_file_ << *sample.memory_utilization;
  telemetry_file_ << ',';
  if (sample.graphics_clock_mhz) telemetry_file_ << *sample.graphics_clock_mhz;
  telemetry_file_ << ',';
  if (sample.energy_mj) telemetry_file_ << *sample.energy_mj;
  telemetry_file_ << '\n';
}

void GpuTelemetry::SampleLoop() {
  NameProfilerThread("nvpm-telemetry");
  bool timestamp_error_reported = false;
  bool power_error_reported = false;
  bool temperature_error_reported = false;
  bool utilization_error_reported = false;
  bool clock_error_reported = false;
  bool energy_error_reported = false;
  bool energy_enabled = options_.nvml_fields;
  bool write_error_reported = false;
  PowerSampleReader power_samples;
  InitializePowerSamples(&power_samples);

  auto last_flush = std::chrono::steady_clock::now();
  auto next_sample = last_flush;

  /* On a V100 the energy counter costs as much CPU as the power read, and the
     other slow reads half as much again. Temperature and utilization, which
     the driver averages anyway, repeat their last value in the rows between
     two reads, so every row stays complete; energy is left empty there so
     that no difference is computed from a stale counter. */
  const auto slow_read_period =
      std::chrono::milliseconds(options_.slow_read_ms);
  auto next_slow_read = last_flush;
  unsigned int temperature_c = 0;
  nvmlUtilization_t utilization{};
  nvmlReturn_t temperature_result = NVML_ERROR_UNINITIALIZED;
  nvmlReturn_t utilization_result = NVML_ERROR_UNINITIALIZED;

  while (true) {
    unsigned int power_mw = 0;
    unsigned int graphics_clock_mhz = 0;

    /* The last row after the stop request always reads everything. */
    const auto poll_time = std::chrono::steady_clock::now();
    const bool slow_read = options_.slow_read_ms == 0 ||
                           poll_time >= next_slow_read ||
                           !running_.load(std::memory_order_acquire);
    if (slow_read) next_slow_read = poll_time + slow_read_period;

    uint64_t query_start_ns = 0;
    uint64_t query_end_ns = 0;
    CUptiResult timestamp_end_result = CUPTI_SUCCESS;

    const CUptiResult timestamp_start_result =
        cuptiGetTimestamp(&query_start_ns);

    const nvmlReturn_t power_result =
        nvmlDeviceGetPowerUsage(device_, &power_mw);

    const std::optional<uint64_t> energy_mj =
        slow_read ? ReadEnergyCounter(&energy_enabled, &energy_error_reported)
                  : std::nullopt;

    /* The energy model integrates power, so keep the slower driver-averaged
       counters outside the interval that dates the power sample. */
    if (options_.power_bracket) {
      timestamp_end_result = cuptiGetTimestamp(&query_end_ns);
    }

    if (slow_read) {
      temperature_result = ReadTemperature(&temperature_c);
      utilization_result = nvmlDeviceGetUtilizationRates(device_, &utilization);
    }

    const nvmlReturn_t clock_result = nvmlDeviceGetClockInfo(
        device_, NVML_CLOCK_GRAPHICS, &graphics_clock_mhz);

    if (!options_.power_bracket) {
      timestamp_end_result = cuptiGetTimestamp(&query_end_ns);
    }

    /* The driver keeps over two seconds of samples, so reading them at the
       slow period loses none. */
    if (slow_read) ReadPowerSamples(&power_samples);

    if (power_result != NVML_SUCCESS && !power_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample power: %s\n",
                   nvmlErrorString(power_result));
      power_error_reported = true;
    }
    if (temperature_result != NVML_SUCCESS && !temperature_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample temperature: %s\n",
                   nvmlErrorString(temperature_result));
      temperature_error_reported = true;
    }
    if (utilization_result != NVML_SUCCESS && !utilization_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample utilization: %s\n",
                   nvmlErrorString(utilization_result));
      utilization_error_reported = true;
    }
    if (clock_result != NVML_SUCCESS && !clock_error_reported) {
      std::fprintf(stderr, "[NVML] Cannot sample graphics clock: %s\n",
                   nvmlErrorString(clock_result));
      clock_error_reported = true;
    }

    if (timestamp_start_result == CUPTI_SUCCESS &&
        timestamp_end_result == CUPTI_SUCCESS) {
      const uint64_t timestamp_ns =
          query_start_ns + (query_end_ns - query_start_ns) / 2;

      ++rows_;
      WriteRow({timestamp_ns, query_start_ns, query_end_ns,
                power_result == NVML_SUCCESS
                    ? std::optional<unsigned int>(power_mw)
                    : std::nullopt,
                temperature_result == NVML_SUCCESS
                    ? std::optional<unsigned int>(temperature_c)
                    : std::nullopt,
                utilization_result == NVML_SUCCESS
                    ? std::optional<unsigned int>(utilization.gpu)
                    : std::nullopt,
                utilization_result == NVML_SUCCESS
                    ? std::optional<unsigned int>(utilization.memory)
                    : std::nullopt,
                clock_result == NVML_SUCCESS
                    ? std::optional<unsigned int>(graphics_clock_mhz)
                    : std::nullopt,
                energy_mj});

    } else if (!timestamp_error_reported) {
      if (timestamp_start_result != CUPTI_SUCCESS) {
        PrintCuptiError("Cannot timestamp the start of an NVML query",
                        timestamp_start_result);
      }
      if (timestamp_end_result != CUPTI_SUCCESS) {
        PrintCuptiError("Cannot timestamp the end of an NVML query",
                        timestamp_end_result);
      }
      timestamp_error_reported = true;
    }

    /* Flush periodically so that an abnormal exit keeps the samples so far. */
    const auto now = std::chrono::steady_clock::now();
    if (now - last_flush >= kFlushInterval) {
      telemetry_file_.flush();
      if (power_samples.enabled) power_samples_file_.flush();
      last_flush = now;
    }
    if (!telemetry_file_ && !write_error_reported) {
      std::fprintf(stderr, "[Profiler] Cannot write CSV file %s\n",
                   telemetry_path_.c_str());
      failed_ = true;
      write_error_reported = true;
    }

    /* Take one final sample after the stop request to bracket the tail. */
    if (!running_.load(std::memory_order_acquire)) break;

    if (options_.fixed_period) {
      /* Skip the slots missed by a slow read instead of sampling in a
         burst. */
      const auto now = std::chrono::steady_clock::now();
      do {
        next_sample += kSampleInterval;
      } while (next_sample <= now);
      std::this_thread::sleep_until(next_sample);
    } else {
      std::this_thread::sleep_for(kSampleInterval);
    }
  }
  thread_cpu_ms_ = ThreadCpuMilliseconds();
}

} /* namespace nvidia */
} /* namespace efimon */
