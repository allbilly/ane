# ANE runtime device tree overlay and loadable KMD

The KMD has always been a loadable module. Stock Asahi device trees omit its
ANE, DART and six power controller nodes. This package supplies those resources
with a runtime overlay; the kernel image and boot DTB do not need to be rebuilt.

This overlay supports **base M1 (T8103)**. M1 Pro/Max and M2 or later need their
own resource definitions. It requires `CONFIG_OF_OVERLAY=y`,
`CONFIG_OF_DYNAMIC=y`, `CONFIG_DRM_ACCEL=y`, and the stock Apple AIC, DART and
PMGR drivers. The Fedora Asahi 6.17.12 and 6.19.14 development packages inspected
on this machine enable these options.

## Build and install

Install `gcc`, `make`, `dtc`, `python3`, and the **kernel development files
matching `uname -r`**. On Fedora Asahi, the matching package is normally
`kernel-16k-devel`. A custom kernel can use its already built source tree via
`KDIR=/path/to/linux`. This build uses its own ANE UAPI header and does not write
into the kernel source or headers.

From the repository root:

```bash
make -C kmod -j"$(nproc)"
sudo make -C kmod install
sudo modprobe ane
ls -l /dev/accel/
```

`modprobe ane` loads the following dependencies:

1. The stock `apple-dart` driver.
2. `ane_dt.ko`, which resolves the existing AIC and ANE power controller
   phandles from the running tree and calls `of_overlay_fdt_apply()` with the
   embedded overlay.
3. `ane.ko`, which binds to the new accelerator node.

The overlay adds `ane_base`, `ane_set1` through `ane_set5`, DART at
`0x26b800000`, and the ANE engine register range at `0x26bc04000`. It uses AIC
interrupts 416 and 417 and DART stream 0. Those resources come from
[Eileen Yoon's T8103 device tree patch](https://github.com/eiln/linux/commit/bf6651bb55212f2cfab573bd0d49bf5c601b4703).
Existing boot ANE nodes are used as-is; conflicting partial trees are refused.

The installed udev rule gives the `render` group and an active local user
session access to the ANE device. For SSH access, join the `render` group and
start a new login session, or grant a temporary ACL:

```bash
sudo setfacl -m "u:$USER:rw" /dev/accel/accel0
```

## Test on this machine

With NumPy available in `.venv`, run:

```bash
sudo ./kmod/load-test.sh
```

The script installs the two already built modules for the running kernel,
loads them with `modprobe`, grants the invoking user a device ACL, and compares
ADD, MUL, MAX, MIN, SUMSQ, RELU, CONV, GEMM, CONCAT and SIGMOID against expected
results. It checks submission after autosuspend, then unloads and reloads
the KMD and repeats the numerical checks. Logs are in `kmod/test-output/`.
The script also fails on new kernel warnings, translation faults or unhandled
interrupts, even if the numerical checks pass.
Use `ANE_TEST_PYTHON=/path/to/python` when NumPy is in another environment.

After loading the driver, GPT-2 can be checked as the normal user:

```bash
gpt2/.venv/bin/python gpt2/gpt2.py verify --backend ane --all-kernels
gpt2/.venv/bin/python gpt2/gpt2.py generate --backend ane \
    --prompt 'The Apple Neural Engine' --max-tokens 32
OPENBLAS_NUM_THREADS=1 gpt2/.venv/bin/python gpt2/training/asahi.py train \
    --steps 10 --checkpoint
```

On this machine all 49 inference kernels and generation parity passed, all
54 training fixtures from ANEForge and Orion matched exactly, and ten full
GPT-2 Adam updates reduced fixed-batch loss from 4.42788 to 0.000207.
[Training instructions and timing scope](../gpt2/training/README-asahi.md)
describe the ANE/CPU split and captured-reference checks.

## Rebuild only the driver

```bash
sudo modprobe -r -i ane
make -C kmod -j"$(nproc)"
sudo make -C kmod install
sudo modprobe ane
```

`ane_dt` and its overlay intentionally remain until reboot. The stock
`apple-pmgr-pwrstate` driver has no removal callback and its registered power
domains retain references to the added nodes. Removing that overlay would
leave stale kernel pointers. Reboot to change the resource overlay; driver
code changes can be reloaded independently. Stop ANE clients before unloading
the driver. `-i` skips removal of its persistent overlay soft dependency.
Rebuild the modules against matching development files after a
kernel update; a `.ko` is specific to its kernel build.

`ane-overlay.dts` contains external phandle markers that the module resolves
at load time. **Do not pass the generated `ane.dtbo` directly to a boot loader
or a generic overlay loader.** The embedded overlay has no dependency on
`/__symbols__`, so it works with stock Asahi DTBs that lack symbol tables.

## Source and validation

The KMD is derived from the local `allbilly/libane` checkout at
`1e0afd832cf171be543d18069cef726aae2b9634`, including its `linux/device.h` include
fix. This copy adds the module dependency, GEM object lifetime cleanup,
persistent DART IRQ masking, and runtime PM error handling needed for reloads.
Removal disables task manager interrupts and acknowledges pending errors in
the secondary DART banks.

The KMD retains the original driver's polling design. Its combined ANE/DART
fault IRQ stays masked until reboot because the stock IOMMU handler covers
only the primary DART bank. The added `apple,ane-polled-fault-irq` device tree
property records that the mask was taken, so repeated KMD loads do not keep
incrementing the IRQ disable count. As in the original driver, hardware faults
on this masked line are not reported asynchronously by the stock handler;
the numerical tests and submission timeouts remain necessary checks.
It retains the original MIT/GPL licensing and Eileen Yoon's copyright.

Builds passed against the running `7.1.13+` tree and the installed stock Fedora
Asahi `6.17.12-400.asahi.fc42.aarch64+16k` and
`6.19.14-400.asahi.fc42.aarch64+16k` development packages. An offline
`fdtoverlay` merge against this machine's live tree passed the register,
interrupt, IOMMU and power controller checks. Hardware validation requires
running the privileged load script; compilation and offline merging alone do
not establish hardware operation.

**Hardware tested on 2026-10-04:** base M1 MacBook Air (J313), running
`7.1.13+` with no ANE nodes in the boot tree. The runtime overlay created
`/dev/accel/accel0`; all ten operation checks passed, submission after runtime
autosuspend passed, and unloading/reloading `ane.ko` and repeating all ten
checks passed without a reboot or kernel rebuild. Final kernel logs had no
new faults or unhandled IRQs. Stock Fedora versions were compile checked;
they were not booted for hardware testing in this session.

The first runtime attempt exposed a difference from `fdtoverlay`: the kernel
rejects an overlay exporting `__symbols__` when the live tree has no symbol
table. The corrected build omits `dtc -@`, retains `__local_fixups__`, and the
embedding script rejects exported symbols or external symbol fixups.

DeepWiki was consulted for the nine repositories listed in `AGENTS.md`.
[The query result](https://deepwiki.com/search/we-are-implementing-an-outoftr_678e1254-d4e7-47f6-97db-8be9024bc35c)
points to `eiln/ane` for the Linux KMD; those reference projects did not supply
a Linux runtime overlay implementation.
