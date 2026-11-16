/**
 * @file gpu-telemetry.hpp
 * @brief NVML telemetry of the measured GPU: power, energy, temperature,
 * utilization and clock, sampled every 50 ms in a dedicated thread
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_GPU_TELEMETRY_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_GPU_TELEMETRY_HPP_

#include <nvml.h>
#include <sys/types.h>

#include <atomic>
#include <cstdint>
#include <efimon/gpu/nvidia-gpu/profiler-options.hpp>
#include <filesystem>  // NOLINT
#include <fstream>
#include <optional>
#include <string>
#include <thread>  // NOLINT
#include <vector>

namespace efimon {
namespace nvidia {

/**
 * @brief NVML telemetry of the measured GPU
 *
 * It writes gpu_telemetry.csv, one row every 50 ms dated with CUPTI
 * timestamps, and gpu_power_samples.csv, the power samples that the driver
 * buffers. The sampling thread is the only writer of both files and of the
 * counters until Stop() joins it; the other methods are called from the
 * profiler's start and stop sequence only.
 */
class GpuTelemetry {
 public:
  /**
   * @brief Construct a new GPU Telemetry object, without device or files
   */
  GpuTelemetry() = default;

  GpuTelemetry(const GpuTelemetry&) = delete;
  GpuTelemetry& operator=(const GpuTelemetry&) = delete;

  /**
   * @brief Takes the telemetry switches of the profiler
   *
   * @param options profiler options; nvml_fields, power_bracket,
   * power_samples, fixed_period and slow_read_ms are used
   */
  void Configure(const ProfilerOptions& options);

  /**
   * @brief Opens gpu_telemetry.csv and writes its header
   *
   * @param directory output directory, which must exist
   * @return false if the file cannot be opened
   */
  bool OpenTelemetryFile(const std::filesystem::path& directory);

  /**
   * @brief Opens gpu_power_samples.csv and writes its header
   *
   * It does nothing when power samples are disabled, and disables them if
   * the file cannot be opened. Call it after OpenTelemetryFile().
   */
  void OpenPowerSamplesFile();

  /**
   * @brief Opens the NVML device that matches a CUDA device
   *
   * @param device_index CUDA device, used in the messages
   * @param pci_bus_id PCI bus id of the CUDA device
   * @return false if NVML cannot be initialized or has no such device
   */
  bool InitializeDevice(int device_index, const std::string& pci_bus_id);

  /**
   * @brief Starts the sampling thread
   *
   * It does nothing without an NVML device, without gpu_telemetry.csv or if
   * the thread already runs.
   */
  void Start();

  /**
   * @brief Takes one last sample and stops the sampling thread
   */
  void Stop();

  /**
   * @brief Flushes and closes the telemetry files
   *
   * @return false if an open file cannot be written or closed
   */
  bool FinishFiles();

  /**
   * @brief Gets the GPU memory that a process uses on the measured device
   *
   * @param pid process
   * @return used memory in bytes, or 0 if it cannot be read
   */
  uint64_t ProcessMemory(pid_t pid) const;

  /**
   * @brief Shuts NVML down if InitializeDevice() succeeded
   */
  void Shutdown();

  /**
   * @brief Checks if the sampling thread was started and not stopped yet
   *
   * @return true while the thread exists
   */
  bool sampling() const { return thread_ != nullptr; }

  /**
   * @brief Checks if InitializeDevice() succeeded
   *
   * @return true with an NVML device
   */
  bool initialized() const { return initialized_; }

  /**
   * @brief Checks if gpu_telemetry.csv could not be written
   *
   * Read it after Stop().
   *
   * @return true after a write error
   */
  bool failed() const { return failed_; }

  /**
   * @brief Gets the path of gpu_telemetry.csv
   *
   * @return path set by OpenTelemetryFile()
   */
  const std::filesystem::path& telemetry_path() const {
    return telemetry_path_;
  }

  /**
   * @brief Gets the rows written to gpu_telemetry.csv
   *
   * @return number of rows; read it after Stop()
   */
  uint64_t rows() const { return rows_; }

  /**
   * @brief Gets the rows written to gpu_power_samples.csv
   *
   * @return number of rows; read it after Stop()
   */
  uint64_t power_sample_rows() const { return power_sample_rows_; }

  /**
   * @brief Gets the CPU time of the sampling thread
   *
   * @return milliseconds; read it after Stop()
   */
  double thread_cpu_ms() const { return thread_cpu_ms_; }

 private:
  /** Values of one gpu_telemetry.csv row; empty fields failed to read */
  struct Sample {
    /** Middle of the query interval */
    uint64_t timestamp_ns;
    /** CUPTI timestamp before the reads */
    uint64_t query_start_ns;
    /** CUPTI timestamp after the bracketed reads */
    uint64_t query_end_ns;
    /** Board power */
    std::optional<unsigned int> power_mw;
    /** GPU temperature */
    std::optional<unsigned int> temperature_c;
    /** GPU utilization */
    std::optional<unsigned int> gpu_utilization;
    /** Memory utilization */
    std::optional<unsigned int> memory_utilization;
    /** Graphics clock */
    std::optional<unsigned int> graphics_clock_mhz;
    /** Energy counter since the driver loaded */
    std::optional<uint64_t> energy_mj;
  };

  /** State of the reads of the power samples that the driver buffers */
  struct PowerSampleReader {
    /** False when power samples are disabled or not supported */
    bool enabled = false;
    /** Set once a read error was reported */
    bool error_reported = false;
    /** Newest sample seen; NVML timestamps are CPU microseconds since the
        epoch */
    unsigned long long last_seen_us = 0;  // NOLINT(runtime/int): NVML type.
    /** Buffer for one read */
    std::vector<nvmlSample_t> samples;
  };

  void SampleLoop();
  std::optional<uint64_t> ReadEnergyCounter(bool* enabled,
                                            bool* error_reported) const;
  nvmlReturn_t ReadTemperature(unsigned int* temperature_c) const;
  void ResolveTemperatureV();
  void InitializePowerSamples(PowerSampleReader* reader) const;
  void ReadPowerSamples(PowerSampleReader* reader);
  void WriteRow(const Sample& sample);

  /** Copy of the profiler options; power_samples is cleared when its file
      cannot be opened */
  ProfilerOptions options_;
  /** NVML device of the measured GPU; valid when initialized_ */
  nvmlDevice_t device_{};
  /** True once InitializeDevice() succeeded */
  bool initialized_ = false;
  /** Path of gpu_telemetry.csv */
  std::filesystem::path telemetry_path_;
  /** Path of gpu_power_samples.csv */
  std::filesystem::path power_samples_path_;
  /** gpu_telemetry.csv */
  std::ofstream telemetry_file_;
  /** gpu_power_samples.csv */
  std::ofstream power_samples_file_;
  /** Cleared to request the last sample */
  std::atomic<bool> running_{false};
  /** Sampling thread; nullptr when not running */
  std::thread* thread_ = nullptr;
  /** Rows written to gpu_telemetry.csv */
  uint64_t rows_ = 0;
  /** Rows written to gpu_power_samples.csv */
  uint64_t power_sample_rows_ = 0;
  /** CPU time of the sampling thread */
  double thread_cpu_ms_ = 0.0;
  /** Set when gpu_telemetry.csv cannot be written */
  bool failed_ = false;
#ifdef nvmlTemperature_v1
  /** nvmlDeviceGetTemperatureV, resolved at run time; nullptr when the
      driver lacks it */
  nvmlReturn_t (*get_temperature_v_)(nvmlDevice_t,
                                     nvmlTemperature_t*) = nullptr;
#endif
};

} /* namespace nvidia */
} /* namespace efimon */

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_GPU_TELEMETRY_HPP_
