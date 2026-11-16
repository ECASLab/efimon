/**
 * @file profiler-utils.hpp
 * @brief Helpers shared by the modules of the NVIDIA process metrics library
 *
 * @copyright Copyright (c) 2026. See License for Licensing
 */

#ifndef INCLUDE_EFIMON_GPU_NVIDIA_GPU_PROFILER_UTILS_HPP_
#define INCLUDE_EFIMON_GPU_NVIDIA_GPU_PROFILER_UTILS_HPP_

#include <cuda.h>
#include <cupti.h>

#include <filesystem>  // NOLINT
#include <fstream>
#include <string>

namespace efimon {
namespace nvidia {

/**
 * @brief Quotes a value for a CSV field
 *
 * @param value text to quote
 * @return value between double quotes, with its double quotes doubled
 */
std::string EscapeCsv(const std::string& value);

/**
 * @brief Prints a CUDA driver error to stderr
 *
 * @param operation operation that failed
 * @param result CUDA driver result
 */
void PrintCudaError(const char* operation, CUresult result);

/**
 * @brief Prints a CUPTI error to stderr
 *
 * @param operation operation that failed
 * @param result CUPTI result
 */
void PrintCuptiError(const char* operation, CUptiResult result);

/**
 * @brief Flushes and closes a CSV file, reporting any write error
 *
 * @param file open file; must not be nullptr
 * @param path path of the file, used in the error messages
 * @return false if the file cannot be written or closed
 */
bool FinishCsvFile(std::ofstream* file, const std::filesystem::path& path);

/**
 * @brief Gets the CPU time used so far by the calling thread
 *
 * @return CPU time in milliseconds, or 0 if it cannot be read
 */
double ThreadCpuMilliseconds();

/**
 * @brief Names the calling thread
 *
 * Per-thread CPU tools can then tell the profiler's threads from the
 * target's.
 *
 * @param name thread name, at most 15 characters
 */
void NameProfilerThread(const char* name);

} /* namespace nvidia */
} /* namespace efimon */

#endif  // INCLUDE_EFIMON_GPU_NVIDIA_GPU_PROFILER_UTILS_HPP_
