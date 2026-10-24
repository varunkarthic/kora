# kora-info

Reports the KORA version, the Vulkan loader and instance versions, and every physical device's
capabilities. It does not create a runtime, so it is safe to run anywhere.

```sh
kora-info
kora-info --json | python3 -m json.tool
```

Expected result: one block per device with its API version, driver, portability-subset and MoltenVK
flags, memory topology, features (timeline semaphores, synchronization2, buffer device address,
descriptor indexing), subgroup support, queue families, and memory heaps and types, with heap budget
and usage when `VK_EXT_memory_budget` is available. Without a Vulkan loader it prints
`NoVulkanImplementation` and exits with status 1.

Reports omit device UUIDs, vendor IDs, device IDs, and other serial identifiers. The pipeline cache
is reported as `not queried`, because no runtime is created.

## JSON

The root object has `schema_version` (`2`), `kora_version`, `loader_api_version`,
`instance_api_version`, `pipeline_cache_state` (always `"not_queried"`), and `devices`. Each device
has `index`, `name`, `api_version`, `driver_name`, `portability_subset`, `moltenvk`,
`memory_topology` (`discrete`, `uma`, `discrete_with_bar`, `unknown`),
`memory_topology_needs_benchmark`, `features`, `subgroup`, `queue_families[]`, and
`memory.{budget_available, heaps[], types[]}`.

Fields may be added within a schema version; incompatible changes increment it.
`tests/integration/check_kora_info_json.py` checks the contract and runs in CI.
