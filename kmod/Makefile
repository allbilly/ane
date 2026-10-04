ifneq ($(KERNELRELEASE),)
ccflags-y += -I$(src)
obj-m := ane.o ane_dt.o
ane-y := ane_drv.o ane_tm.o
$(obj)/ane_dt.o: $(src)/overlay_blob.h
else
KVER ?= $(shell uname -r)
KDIR ?= /lib/modules/$(KVER)/build
DTC ?= dtc
PYTHON ?= python3
MODULE_DIR := /lib/modules/$(KVER)/extra/ane

.PHONY: all install uninstall clean
all: overlay_blob.h
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

ane.dtbo: ane-overlay.dts Makefile
	$(DTC) -Wno-interrupts_property -Wno-interrupt_provider -Wno-power_domains_property -I dts -O dtb -o $@ $<

overlay_blob.h: ane.dtbo embed-overlay.py
	$(PYTHON) embed-overlay.py $< $@

install: all
	install -d $(MODULE_DIR)
	install -m 644 ane.ko ane_dt.ko $(MODULE_DIR)/
	install -D -m 644 99-ane.rules /etc/udev/rules.d/99-ane.rules
	depmod -a $(KVER)
	udevadm control --reload-rules

uninstall:
	modprobe -r -i ane
	rm -f $(MODULE_DIR)/ane.ko $(MODULE_DIR)/ane_dt.ko
	rm -f /etc/udev/rules.d/99-ane.rules
	depmod -a $(KVER)
	udevadm control --reload-rules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
	rm -f ane.dtbo overlay_blob.h
endif
