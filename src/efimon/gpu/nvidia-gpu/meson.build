#
# See LICENSE for more information about licensing
#  Copyright 2026
#

kernel_journal_sources = files('kernel-journal.cpp')

# Preloaded collector: it runs inside the profiled process, so it does not
# link libefimon and it is always optimized, as in the standalone profiler.
libnvidia_process_metrics = shared_library('nvidia-process-metrics',
  [
    files('activity-collector.cpp'),
    files('capture-writer.cpp'),
    files('gpu-telemetry.cpp'),
    files('process-metrics.cpp'),
    files('profiler-options.cpp'),
    files('profiler-utils.cpp'),
    kernel_journal_sources,
  ],
  cpp_args : cpp_args,
  include_directories : [project_inc, nvidia_gpu_inc],
  dependencies : [nvidia_gpu_deps, dependency('threads')],
  override_options : ['optimization=2'],
  install_rpath : cupti_lib_dir,
  install : true,
  install_dir : join_paths(get_option('libdir'), 'nvidia-process-metrics'),
)
