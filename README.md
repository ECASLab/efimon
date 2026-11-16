# EfiMon: An agnostic efficiency monitor library

Easy-To-Use Linux Power and Resource Consumption Monitoring Library and Daemon. Monitors CPU, GPU, RAM, I/O, Network and the overall system.

## Installation

This library uses `meson` as the building system. To install meson, Python >= 3.8 is required. Please, check the following instructions according to your needs:

* Python and PIP

```bash
# Ubuntu
sudo apt install python3 python3-pip
```

* Ninja Build

```bash
# Ubuntu
sudo apt install ninja-build
```

* Meson

```bash
sudo pip3 install meson
```

**Dependencies**

First, it is important to recall that this is a Linux library. It requires the following dependencies:

* libprocps
* sqlite3 >= 3.31.1
* Linux Perf
* Intel RAPL
* Free IPMI

On Fedora 40, you can install some of these dependencies using:

```bash
# ProcPS
yum install procps-ng-devel.x86_64 procps-ng.x86_64
# Linux Perf
yum install perf
# SQLite
yum install libsqlite3x.x86_64 libsqlite3x-devel.x86_64
# ZeroMQ
yum install cppzmq-devel.x86_64
# JsonCPP
yum install jsoncpp-devel.x86_64
# Capstone
yum install capstone capstone-devel
```

On Ubuntu 20.04:

```bash
# ProcPS
apt install libprocps-dev libprocps8
# Linux Perf
apt install linux-tools-common linux-tools-generic
# SQLite
apt install libsqlite3-dev libsqlite3-0
# ZeroMQ
apt install libzmqpp4 libzmqpp-dev
# JsonCPP
apt install libjsoncpp-dev
# Capstone
apt install libcapstone-dev
```

On Ubuntu 24.04:

```bash
# ProcPS
apt install libcapstone-dev 
# Linux Perf
apt install linux-tools-common linux-tools-generic
# SQLite
apt install libsqlite3-dev libsqlite3-0
# ZeroMQ
apt install cppzmq-dev
# JsonCPP
apt install libjsoncpp-dev
```

**Optional**

For the documentation, you will require other dependencies like graphviz, java, texlive and doxygen. Please, install them by using:

```bash
# Ubuntu
sudo apt install doxygen graphviz openjdk-17-jre texlive-font-utils
```

You can switch the "-Dbuild-docs" or "-Dbuild-docs-only" to true if you want to compile the documentation.

### Compiling EfiMon

Compiling EfiMon follows the same process as any meson project. Please, follow the steps to have EfiMon on your system:

```bash
meson builddir $OPTIONS
ninja -C builddinr
sudo ninja -C builddir
```

EfiMon will be installed under the `/usr/local/` prefix.

For the `$OPTIONS`, you can select the following:

| Option                 | Default Value    | Values        | Description                      |
|------------------------|------------------|---------------|----------------------------------|
| build-docs             | false            | [true, false] | Enable docs compilation          |
| build-docs-only        | false            | [true, false] | Enable docs-only compilation     |
| build-examples         | true             | [true, false] | Enable examples compilation      |
| developer-mode         | true             | [true, false] | Enable developer mode            |
| enable-pcm             | true             | [true, false] | Enable the Intel PCM             |
| enable-perf            | true             | [true, false] | Enable the Linux Perf Tool       |
| enable-rapl            | true             | [true, false] | Enable the RAPL Interface        |
| enable-sql             | true             | [true, false] | Enable the SQL Logger            |
| enable-ipmi            | true             | [true, false] | Enable the IPMI Logger           |
| enable-nvidia-gpu      | false            | [true, false] | Enable the NVIDIA GPU process metrics (CUPTI and NVML) |
| nvidia-toolkit-root    | /usr/local/cuda  | path          | CUDA toolkit prefix              |
| nvidia-toolkit-target  | (empty)          | path          | CUDA target prefix, for example `/usr/local/cuda/targets/sbsa-linux` on Jetson Thor; empty uses `extras/CUPTI` of the toolkit |

### Testing the installation

T.B.D


## Usage

### EfiMon Power Analyser

The EfiMon Power Analyser wraps a process and perform the analysis. It is self-contained and requires root.

Currently, we have been focused on an analyser for measuring the power consumption based on the histograms.

```bash
PID=2000 # Process ID
STIME=10 # Metering during 10 seconds
sudo efimon-power-analyser -p ${PID} -s ${STIME} > consumption.csv
```

Another way to execute it:

```bash
STIME=10 # Metering during 10 seconds
# the command is time sleep 1
sudo efimon-power-analyser -s ${STIME} -c time sleep 1
```

### EfiMon Daemon

The EfiMon Daemon is a server that performs observations of PID. It receives the information about the processes to analyse over IPC (TCP). It does require root.

Example of usage:

```bash
efimon-daemon
```

It has options to:

* Change the output folder for the logs (default folder: /tmp)
* Adjust the frequency of perf
* Adjust the delay between samples
* Change the port of the IPC

### EfiMon Launcher

The EfiMon Launcher wraps an application, launching its execution or intercepting a PID. It connects to the EfiMon Daemon over IPC and extracts the analysis.

It does not require root.

* Example of usage (wrapper):

```bash
APP="../NanoBenchmark/cpu 11"

efimon-launcher -s 10 -c ${APP}
```

It launches a command.

* Example of usage (interceptor):

```bash
efimon-launcher -s 3 --pid 60603
```

It monitors an already running process.

It has options to:

* Save the log file in a different location
* Adjust the frequency of perf
* Adjust the delay between samples
* Change the port of the IPC
* Enable of disable perf

> The launcher requires a running instance of the EfiMon Daemon

### EfiMon NVIDIA Process Metrics

The NVIDIA process metrics library measures one CUDA process from inside it. It is loaded with `LD_PRELOAD` and records:

* The GPU telemetry from NVML every 50 ms: power, energy, temperature, utilization and graphics clock.
* Every kernel and memory copy of the process from CUPTI, with CUPTI timestamps, so they can be aligned with the telemetry.

It requires the CUDA toolkit (CUDA driver, CUPTI and NVML) and does not require root. Build it with:

```bash
meson setup builddir -Denable-nvidia-gpu=true -Dnvidia-toolkit-root=/usr/local/cuda
ninja -C builddir
```

On Jetson Thor, add `-Dnvidia-toolkit-target=/usr/local/cuda/targets/sbsa-linux`. The library is installed as `<prefix>/<libdir>/nvidia-process-metrics/libnvidia-process-metrics.so`.

* Example of usage:

```bash
LIB=builddir/src/efimon/gpu/nvidia-gpu/libnvidia-process-metrics.so
NVIDIA_METRICS_OUTPUT_DIR=session LD_PRELOAD=${LIB} ./my-cuda-app
```

When the process exits, the session directory holds:

| File | Content |
|------|---------|
| `gpu_telemetry.csv` | One NVML sample every 50 ms, dated with CUPTI timestamps |
| `gpu_power_samples.csv` | Power samples that the driver buffers, when the GPU provides them |
| `kernel_activity.csv` | Every kernel in start order: ids, start, end, duration and name |
| `memcpy_activity.csv` | Every memory copy: start, end, bytes and kind |
| `capture_summary.txt` | Counters of the capture; `capture_status` is `complete`, `partial` or `failed` |

While the process runs, kernels go to `kernel_activity.journal`, which is deleted once `kernel_activity.csv` is written. If the process is killed, the journal stays and the journal recovery of efimon-analyser turns it into the CSV.

The library reads these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_METRICS_OUTPUT_DIR` | `.` | Session directory |
| `NVIDIA_METRICS_DEVICE` | 0 | CUDA device whose telemetry and activity are recorded |
| `NVIDIA_METRICS_WINDOW_MS` | 200 | Kernel window of the console summary |
| `NVIDIA_METRICS_TAIL_MS` | window | Telemetry kept after the last kernel, up to 60000 |
| `NVIDIA_METRICS_FOLLOW_CHILDREN` | 0 | Also profile child processes, under `children/<pid>` |
| `NVIDIA_METRICS_ALL_DEVICES` | 0 | Keep the activity of every device, not only the measured one |
| `NVIDIA_METRICS_SLOW_READ_MS` | 200 | Period of the energy, temperature and utilization reads; 0 reads them in every row |
| `NVIDIA_METRICS_MEMCPY_CSV` | 1 | Write `memcpy_activity.csv` |
| `NVIDIA_METRICS_KERNEL_JOURNAL` | 1 | Write kernels to the journal instead of keeping them in memory |
| `NVIDIA_METRICS_ACTIVITY_WORKER` | 1 | Parse CUPTI buffers in a dedicated thread |
| `NVIDIA_METRICS_ACTIVITY_POOL_BUFFERS` | 20 | Preallocated 1 MiB CUPTI buffers, up to 256 |
| `NVIDIA_METRICS_ACTIVITY_FLUSH_MS` | 1000 | CUPTI periodic flush; 0 keeps the CUPTI heuristics |
| `NVIDIA_METRICS_NVML_FIELDS` | 1 | Read the NVML energy counter |
| `NVIDIA_METRICS_POWER_SAMPLES` | 1 | Write `gpu_power_samples.csv` |
| `NVIDIA_METRICS_POWER_BRACKET` | 1 | Date only the power and energy reads with the CUPTI timestamps |
| `NVIDIA_METRICS_FIXED_PERIOD` | 1 | Sample every 50 ms from a fixed start, without drift |

> Only one CUPTI client can run in a process: the library cannot be combined with Nsight Systems, Nsight Compute or compute-sanitizer.

The `kernel-journal-testing` example checks the kernel journal; it runs with `meson test -C builddir` when `enable-nvidia-gpu` is set.

## Platforms

EfiMon has been tested in the following platforms:

* Ubuntu 20.04 with Intel Core i5 6000
* Fedora 40 with AMD Epyc Zen2

## Service Installation

You can launch the daemon to run as root from the system start:

```bash
cp misc/efimon.service /etc/systemd/system/efimon.service
sudo systemctl enable efimon
sudo systemctl start efimon
```

## Packaging as RPM file

Prepare the environment:

```bash
mkdir ~/rpmbuild/SOURCES
tar czf efimon-0.2.0.tar.gz .
cp efimon-0.2.0.tar.gz ~/rpmbuild/SOURCES/efimon-0.2.0.tar.gz
```

Then, compile:

```bash
cd misc
rpmbuild -bb efimon.spec
```

The RPM file will be available in `~/rpmbuild/RPMS`

Install:

```bash
sudo dnf localinstall efimon-0*.rpm
```

## Additional Information

This project is given to you under the LGPL v2.1 Licence. Dynamic linkage from commercial applications to EfiMon is allowed. Derived work and modifications must be released under the same license as EfiMon (LGPL v2.1).

### Contributors

* Luis G. Leon Vega <l.leon@tec.ac.cr>
* Niccolo Tosato <niccolo.tosato@areasciencepark.it>

### Official Repository

* Github: https://github.com/lleon95/efimon
