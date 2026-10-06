# Whisper on ANE under Asahi Linux

Asahi uses a Linux kernel driver and `libane`, independently of macOS ANEForge.
The requested [PR #1021](https://github.com/ggml-org/whisper.cpp/pull/1021)
is a 2023 proof of concept and remains unmerged as of 2026-10-06. There is no
current upstream `WHISPER_ASAHI` CMake switch. Installing ANEForge or enabling
Core ML in a Linux build does not provide Linux ANE support.

The [Asahi M1 support page](https://asahilinux.org/docs/platform/feature-support/m1/#ane-driver)
lists the Neural Engine driver as out of tree. The experiment's documented
machine was an M1 MacBook Pro running `6.3.0-asahi-8-1-ARCH`. Its results do not
establish compatibility with today's Fedora Asahi kernel or every M-series chip.
**This Linux path was inspected, not executed here: this host runs macOS.**

## Required stack

| Component | Purpose |
| --- | --- |
| Native Asahi installation | Provides Linux access to Apple Silicon hardware; a regular Linux VM is insufficient |
| [eiln/ane kernel module](https://github.com/eiln/ane/tree/main/ane) | Exposes an ANE accelerator node and custom DRM ioctls |
| [libane](https://github.com/eiln/ane/tree/main/libane) | Sends tiled FP16 inputs and receives outputs through that driver |
| [anecc](https://github.com/eiln/anecc) | Converts Apple's hardware executable into the Linux `.anec` format |
| PR #1021 checkout | Replaces Whisper's encoder with libane calls using the historical Core ML interface names |
| Trained tiny encoder artifact | Must match the normal `ggml-tiny.bin` checkpoint and the target ANE hardware |

The inspected driver has device-tree matches for `apple,t8103-ane` and
`apple,t6000-ane`. Verify support for your SoC and the driver/kernel pair before
attempting this recipe. Modern kernel API changes may require a driver port.
The named macOS dylib, `.mlmodelc` directories and ANEForge `model.mil` bundles
are not substitutes for a Linux `.anec` model.

## 1. Prepare the driver on native Linux

Obtain matching running-kernel development headers, a C/C++ toolchain, make,
Git, Python and libdrm development headers using your distribution's packages.
The header tree must match the actual Asahi/16K kernel, rather than an unrelated
generic aarch64 kernel.

```sh
uname -r
uname -m
test -f "/lib/modules/$(uname -r)/build/Makefile"

git clone https://github.com/eiln/ane.git ane
git -C ane checkout 0dcea9976fae0b500a236a62fca69cd4d39f0809
make -C ane/ane
```

The driver's Makefile copies its custom UAPI header into the kernel header tree
using sudo. If compilation fails against your kernel, stop and resolve that
compatibility issue; successful compilation of whisper.cpp alone cannot fix it.
On a compatible development installation, install the built module explicitly:

```sh
sudo install -D -m 644 ane/ane/ane.ko "/lib/modules/$(uname -r)/extra/ane.ko"
sudo depmod -a
sudo modprobe ane
lsmod | rg '^ane '
ls -l /dev/accel/
sudo dmesg | rg -i 'ane|neural'
```

`libane` scans `/dev/accel/accel0` through `accel63` and verifies the DRM driver
name. A GPU render node is not the ANE node. Grant the intended user access to
the actual ANE accelerator node with a group/udev rule. For a temporary test
where that node is `accel0`:

```sh
sudo chgrp "$(id -gn)" /dev/accel/accel0
sudo chmod g+rw /dev/accel/accel0
```

The source Makefile's install target uses `chmod 666`; a user/group rule avoids
granting all users access. Build and install the historical userspace library:

```sh
make -C ane/libane
sudo make -C ane/libane install
```

This implementation installs the static library into `/usr/lib/libane.a` and
headers into `/usr/include/libane`, which are the paths the Whisper proof of
concept expects. If your layout differs, update its include/link paths.

## 2. Produce a compatible trained model

Model compilation starts on macOS. `anecc` expects an ANE hardware executable
(`.hwx`), then emits `.anec`. Its `tohwx` tool uses private macOS frameworks to
extract the executable from a neural-network `.mlmodel`.

```sh
# On the Apple Silicon Mac used to prepare the matching tiny encoder:
git clone https://github.com/eiln/anecc.git anecc
git -C anecc checkout aa7b14f455e43db53b949c2151c430cc0c3e1d1b
make -C anecc/tohwx

# Prerequisite: a trained, ANE-compatible tiny encoder .mlmodel.
./anecc/tohwx/tohwx coreml_encoder_tiny.mlmodel

python3 -m venv .venv-anecc
.venv-anecc/bin/python -m pip install ./anecc/anecc
.venv-anecc/bin/anecc coreml_encoder_tiny.hwx \
  -o coreml_encoder_tiny.anec
```

`-o` writes the output; alternatively use `-w`. Copy the resulting `.anec` to
native Linux. A preconverted compatible model can skip macOS preparation.

This prerequisite needs care: the old Whisper converter defaults to MLProgram
and saves `.mlpackage`, while
[anecc's instructions](https://github.com/eiln/anecc#coreml-conversion)
require a neural-network `.mlmodel`. Merely renaming a `.mlpackage` does not
convert it. Use that converter's neural-network guidance and verify that the
trained encoder compiles entirely to a compatible ANE executable with the
expected FP16 input/output shapes. Current macOS compiler output may differ from
what the historical extractor/parser accepts.

There is **no turnkey encoder artifact in the PR's committed tree**. Its
`.gitignore` excludes `.anec`/`.hwx` data, and the C wrapper references a generated
header/object that are absent. These steps explain the model pipeline; they do
not assert that today's macOS tools can regenerate the missing artifact without
conversion work.

## 3. Check out and adapt the historical Whisper proof of concept

Use a separate checkout; the branch targets the old Makefile build and `./main`,
not today's CMake `whisper-cli` layout.

```sh
git clone https://github.com/ggml-org/whisper.cpp.git whisper-asahi
cd whisper-asahi
git fetch origin pull/1021/head
git checkout --detach e5346aeb5803d47f4b717a4fe82621d21cbc7b01

mkdir -p asahi/data
# Place coreml_encoder_tiny.anec in asahi/data/.
hf download ggerganov/whisper.cpp ggml-tiny.bin --local-dir models
```

At that revision, [the C wrapper](https://github.com/eiln/whisper.cpp/blob/e5346aeb5803d47f4b717a4fe82621d21cbc7b01/asahi/whisper-encoder.c)
includes `data/anec_coreml_encoder_tiny.h` and calls
`ane_init_coreml_encoder_tiny()`. The
[Makefile](https://github.com/eiln/whisper.cpp/blob/e5346aeb5803d47f4b717a4fe82621d21cbc7b01/Makefile)
links an absent object at `/home/eileen/whisper.cpp/asahi/data/coreml_encoder_tiny.anec.o`.
These are experiment-specific dependencies, not paths you should recreate.

Either obtain the author's generated embedding files, or adapt the isolated
checkout to file loading with libane's public API:

1. Remove `#include "data/anec_coreml_encoder_tiny.h"` from
   `asahi/whisper-encoder.c`.
2. Replace `ane_init_coreml_encoder_tiny()` with
   `ane_init("asahi/data/coreml_encoder_tiny.anec")`.
3. Remove only the hard-coded `.anec.o` path from `WHISPER_OBJ` in the Makefile,
   retaining `whisper.o whisper-encoder.o`.

This adaptation is inferred from the inspected libane interface and has not
been validated on Linux here. Run from the checkout root because the model path
is relative. The built-in wrapper ignores the normal model-path argument, so
you must manually keep the ANE tiny encoder paired with `ggml-tiny.bin`.

On a compatible driver/library/model setup:

```sh
make clean
WHISPER_ASAHI=1 make -j4 main
./main -m models/ggml-tiny.bin -f samples/jfk.wav -t 4 -l en
```

Verify a sensible JFK transcript, driver execution and absence of libane ioctl,
model-loading or execution errors. The `COREML = 1` label in the historical
example reflects reused interface names; it does not mean Apple's Core ML
framework runs on Linux. Compare against a clean CPU build of the same revision:

```sh
make clean
make -j4 main
./main -m models/ggml-tiny.bin -f samples/jfk.wav -t 4 -l en
```

The PR reports faster encoding on its M1/2023 setup. Re-measure on your own
machine; decoder and preprocessing remain separate costs. Its additional
`asahi/decoder.py` is a partial Python decoder demonstration, not a complete
ANE decoder integrated into `./main`.

## If the historical stack is unavailable

Run current whisper.cpp on CPU while the driver/model work is unresolved:

```sh
git clone https://github.com/ggml-org/whisper.cpp.git whisper-cpu
cmake -S whisper-cpu -B whisper-cpu/build -DCMAKE_BUILD_TYPE=Release
cmake --build whisper-cpu/build --target whisper-cli -j4
hf download ggerganov/whisper.cpp ggml-tiny.bin --local-dir whisper-cpu/models
./whisper-cpu/build/bin/whisper-cli \
  -m whisper-cpu/models/ggml-tiny.bin -f whisper-cpu/samples/jfk.wav -ng
```

This last command is a CPU route. Linux GPU acceleration, when configured
separately, also does not establish ANE usage.
