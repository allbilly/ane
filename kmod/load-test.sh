#!/usr/bin/env bash
# Install already built modules and test the runtime overlay on the running M1.
set -euo pipefail
task_dir=$(cd -- "$(dirname -- "$0")" && pwd)
task_root=$(dirname -- "$task_dir")
task_kernel=$(uname -r)
task_user=${SUDO_USER:-$(id -un)}
task_started=$(date '+%Y-%m-%d %H:%M:%S')
task_python=${ANE_TEST_PYTHON:-$task_root/.venv/bin/python}

if [[ $EUID != 0 ]]; then
    echo "Run with sudo: $0" >&2
    exit 1
fi
for module in ane ane_dt; do
    task_vermagic=$(modinfo -F vermagic "$task_dir/$module.ko")
    [[ ${task_vermagic%% *} == "$task_kernel" ]] || {
        echo "$module.ko does not match running kernel $task_kernel" >&2
        exit 1
    }
done
[[ -x $task_python ]] || task_python=$(command -v python3)
mkdir -p "$task_dir/test-output"
exec > >(tee "$task_dir/test-output/load-test.log") 2>&1
finish_test() {
    task_status=$?
    if ! dmesg --since "$task_started" > "$task_dir/test-output/dmesg.log"; then
        echo "FAIL: could not capture kernel logs" >&2
        task_status=1
    elif awk '/nobody cared|Disabling IRQ|BUG:|Oops:|WARNING:|translation fault|Internal error:/ {bad=1} END {exit !bad}' "$task_dir/test-output/dmesg.log"; then
        echo "FAIL: kernel reported an error; see test-output/dmesg.log" >&2
        task_status=1
    elif [[ $task_status == 0 ]]; then
        echo "PASS: kernel log has no new faults or unhandled IRQs"
    fi
    exit "$task_status"
}
trap finish_test EXIT

echo "Testing kernel $task_kernel; user $task_user"
if lsmod | awk '$1 == "ane" {found=1} END {exit !found}'; then
    modprobe -r -i ane
fi
install -d "/lib/modules/$task_kernel/extra/ane"
install -m 644 "$task_dir/ane.ko" "$task_dir/ane_dt.ko" "/lib/modules/$task_kernel/extra/ane/"
depmod -a "$task_kernel"

load_driver() {
    modprobe ane_dt
    modprobe ane
    udevadm settle --timeout=15
    task_device=
    for device in /sys/class/accel/accel*; do
        [[ -e $device/device/driver ]] || continue
        if [[ $(basename -- "$(readlink -f "$device/device/driver")") == ane ]]; then
            task_device="/dev/accel/$(basename -- "$device")"
            break
        fi
    done
    [[ -n $task_device && -c $task_device ]] || {
        echo "ANE did not probe; see test-output/dmesg.log" >&2
        return 1
    }
    # Existing examples use accel0; avoid running them on another accelerator.
    [[ $task_device == /dev/accel/accel0 ]] || {
        echo "ANE is $task_device; examples need their device path adjusted" >&2
        return 1
    }
    setfacl -m "u:$task_user:rw" "$task_device"
    echo "PASS: modprobe ane created $task_device"
}

load_driver
runuser -u "$task_user" -- "$task_python" "$task_dir/smoke.py"
sleep 2
runuser -u "$task_user" -- "$task_python" "$task_dir/smoke.py" elementwise add
echo "PASS: submit after runtime autosuspend"
modprobe -r -i ane
sleep 1
echo "PASS: unloaded ane; overlay resource module remains loaded"
load_driver
runuser -u "$task_user" -- "$task_python" "$task_dir/smoke.py"
echo "PASS: driver reload and hardware checks; no reboot or kernel rebuild"
