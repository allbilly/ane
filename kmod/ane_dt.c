// SPDX-License-Identifier: GPL-2.0-only OR MIT
/* Apply the T8103 ANE overlay to a stock Asahi live device tree. */
#define pr_fmt(fmt) "ane_dt: " fmt

#include <linux/module.h>
#include <linux/of.h>
#include <linux/of_platform.h>
#include <linux/platform_device.h>
#include <linux/slab.h>

#include "overlay_blob.h"

#define PMGR_PATH "/soc/power-management@23b700000"
static int overlay_id;

static bool provider_ready(struct device_node *np)
{
	struct platform_device *pdev = of_find_device_by_node(np);
	bool ready = pdev && pdev->dev.driver;

	platform_device_put(pdev);
	return ready;
}

/* Stock DTBs may omit the CPU domain's phandle because it has no consumers.
 * It already has a registered genpd; give that existing node a handle rather
 * than attempting to create a second controller for the same register.
 */
static int ensure_phandle(struct device_node *np)
{
	struct device_node *node;
	struct property *prop;
	__be32 *value;
	u32 max = 0;
	int ret;

	if (np->phandle)
		return 0;
	if (of_find_property(np, "phandle", NULL))
		return -EINVAL;
	for (node = of_find_all_nodes(NULL); node; node = of_find_all_nodes(node))
		if (node->phandle != U32_MAX)
			max = max_t(u32, max, node->phandle);
	if (max >= U32_MAX - 1)
		return -EOVERFLOW;

	prop = kzalloc(sizeof(*prop), GFP_KERNEL);
	if (!prop)
		return -ENOMEM;
	prop->name = kstrdup("phandle", GFP_KERNEL);
	value = kmalloc(sizeof(*value), GFP_KERNEL);
	if (!prop->name || !value) {
		kfree(prop->name);
		kfree(value);
		kfree(prop);
		return -ENOMEM;
	}
	*value = cpu_to_be32(max + 1);
	prop->value = value;
	prop->length = sizeof(*value);
	of_property_set_flag(prop, OF_DYNAMIC);
	ret = of_add_property(np, prop);
	if (ret) {
		kfree(prop->name);
		kfree(value);
		kfree(prop);
		return ret;
	}
	WRITE_ONCE(np->phandle, max + 1);
	return 0;
}

static void patch_phandles(unsigned char *blob, const unsigned int *offsets,
			   size_t count, u32 phandle)
{
	__be32 value = cpu_to_be32(phandle);
	size_t i;

	for (i = 0; i < count; i++)
		memcpy(blob + offsets[i], &value, sizeof(value));
}

static int __init ane_dt_init(void)
{
	static const char * const additions[] = {
		PMGR_PATH "/power-controller@c008",
		PMGR_PATH "/power-controller@c010",
		PMGR_PATH "/power-controller@c018",
		PMGR_PATH "/power-controller@c020",
		PMGR_PATH "/power-controller@c028",
		PMGR_PATH "/power-controller@c030",
		"/soc/iommu@26b800000",
	};
	struct device_node *aic = NULL, *sys = NULL, *cpu = NULL;
	struct device_node *soc = NULL, *pmgr = NULL, *node;
	unsigned char *blob = NULL;
	u32 cells;
	int i, ret = -ENODEV;

	if (!IS_ENABLED(CONFIG_OF_OVERLAY)) {
		pr_err("kernel needs CONFIG_OF_OVERLAY=y\n");
		return -EOPNOTSUPP;
	}
	if (!of_machine_is_compatible("apple,t8103")) {
		pr_err("this overlay supports base M1 (T8103) only\n");
		return -ENODEV;
	}
	node = of_find_compatible_node(NULL, NULL, "apple,t8103-ane");
	if (node) {
		pr_info("ANE node already exists at %pOF; using boot device tree\n", node);
		of_node_put(node);
		return 0;
	}
	for (i = 0; i < ARRAY_SIZE(additions); i++) {
		node = of_find_node_by_path(additions[i]);
		if (node) {
			pr_err("refusing partial/conflicting ANE tree at %pOF\n", node);
			of_node_put(node);
			return -EEXIST;
		}
	}

	soc = of_find_node_by_path("/soc");
	pmgr = of_find_node_by_path(PMGR_PATH);
	aic = of_find_node_by_path("/soc/interrupt-controller@23b100000");
	sys = of_find_node_by_path(PMGR_PATH "/power-controller@470");
	cpu = of_find_node_by_path(PMGR_PATH "/power-controller@c000");
	if (!soc || !pmgr || !aic || !sys || !cpu || !aic->phandle ||
	    !sys->phandle || !of_device_is_available(cpu) ||
	    !of_device_is_compatible(aic, "apple,t8103-aic") ||
	    !of_node_check_flag(soc, OF_POPULATED_BUS) ||
	    !of_node_check_flag(pmgr, OF_POPULATED_BUS)) {
		pr_err("required T8103 buses, AIC or power controllers are missing\n");
		goto out;
	}
	if (of_property_read_u32(aic, "#interrupt-cells", &cells) || cells != 3 ||
	    of_property_read_u32(cpu, "#power-domain-cells", &cells) || cells != 0 ||
	    of_property_read_u32(sys, "#power-domain-cells", &cells) || cells != 0) {
		pr_err("unexpected interrupt or power domain cell counts\n");
		goto out;
	}
	if (!provider_ready(sys) || !provider_ready(cpu)) {
		pr_err("stock ANE power providers have not probed; retry later\n");
		ret = -EAGAIN;
		goto out;
	}
	blob = kmemdup(ane_overlay, sizeof(ane_overlay), GFP_KERNEL);
	if (!blob) {
		ret = -ENOMEM;
		goto out;
	}
	ret = ensure_phandle(cpu);
	if (ret)
		goto out;
	patch_phandles(blob, aic_offsets, ARRAY_SIZE(aic_offsets), aic->phandle);
	patch_phandles(blob, sys_offsets, ARRAY_SIZE(sys_offsets), sys->phandle);
	patch_phandles(blob, cpu_offsets, ARRAY_SIZE(cpu_offsets), cpu->phandle);
	ret = of_overlay_fdt_apply(blob, sizeof(ane_overlay), &overlay_id, NULL);
	if (ret)
		pr_err("overlay apply failed (%d); inspect kernel logs before retrying\n", ret);
	else
		pr_info("applied T8103 ANE overlay %d; resources persist until reboot\n",
			overlay_id);
out:
	kfree(blob);
	of_node_put(cpu);
	of_node_put(sys);
	of_node_put(aic);
	of_node_put(pmgr);
	of_node_put(soc);
	return ret;
}

module_init(ane_dt_init);
/* Intentionally no module_exit: apple-pmgr-pwrstate has no remove callback.
 * Removing its overlay would free memory still used by registered genpds.
 * The independently loadable ane.ko can be unloaded and replaced normally.
 */
MODULE_SOFTDEP("pre: apple-dart");
MODULE_DESCRIPTION("Runtime T8103 ANE device tree overlay (persistent until reboot)");
MODULE_LICENSE("Dual MIT/GPL");
