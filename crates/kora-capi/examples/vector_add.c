/* Adds two vectors on the first KORA device through the C ABI.
 *
 *   cc -I<prefix>/include vector_add.c -L<prefix>/lib -lkora -o vector_add
 *   ./vector_add path/to/vector_add.spv
 *
 * Exit status: 0 on success, 77 when no Vulkan device is available, 1 on any other failure.
 */
#include <stdio.h>
#include <stdlib.h>

#include "kora.h"

#define COUNT 4099u

static const char METADATA[] =
    "{\"schema_version\":1,\"kernel_id\":\"c.vector_add\",\"local_size\":[64,1,1],"
    "\"push_constant_size\":4,\"push_constants\":[{\"name\":\"count\",\"type\":\"u32\",\"offset\":0,"
    "\"size\":4}],\"bindings\":["
    "{\"set\":0,\"binding\":0,\"name\":\"input_a\",\"kind\":\"storage_buffer\",\"access\":\"read\"},"
    "{\"set\":0,\"binding\":1,\"name\":\"input_b\",\"kind\":\"storage_buffer\",\"access\":\"read\"},"
    "{\"set\":0,\"binding\":2,\"name\":\"output_data\",\"kind\":\"storage_buffer\",\"access\":\"write\"}],"
    "\"specialization_constants\":[]}";

static int fail(const char* what, KoraStatus status) {
    fprintf(stderr, "%s failed (%s): %s\n", what, kora_status_name(status), kora_last_error_message());
    return 1;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s vector_add.spv\n", argv[0]);
        return 1;
    }
    if (kora_abi_version() != KORA_ABI_VERSION) {
        fprintf(stderr, "header and library ABI versions differ\n");
        return 1;
    }

    KoraRuntimeOptions options;
    kora_runtime_options_default(&options);
    options.enable_startup_calibration = 0;
    KoraRuntime* runtime = NULL;
    KoraStatus status = kora_runtime_create(&options, &runtime);
    if (status == KORA_STATUS_NO_VULKAN_IMPLEMENTATION || status == KORA_STATUS_NO_ELIGIBLE_DEVICE) {
        printf("skipped: %s\n", kora_last_error_message());
        return 77;
    }
    if (status != KORA_STATUS_OK) return fail("kora_runtime_create", status);

    char name[256];
    if (kora_runtime_device_name(runtime, 0, name, sizeof name, NULL) == KORA_STATUS_OK)
        printf("KORA %s on %s\n", kora_version_string(), name);

    KoraKernel* kernel = NULL;
    status = kora_kernel_create_from_file(runtime, argv[1], "main", METADATA, &kernel);
    if (status != KORA_STATUS_OK) return fail("kora_kernel_create_from_file", status);

    uint32_t* a = malloc(COUNT * sizeof(uint32_t));
    uint32_t* b = malloc(COUNT * sizeof(uint32_t));
    uint32_t* out = calloc(COUNT, sizeof(uint32_t));
    for (uint32_t i = 0; i < COUNT; ++i) {
        a[i] = i;
        b[i] = 2 * i;
    }
    KoraBuffer* buffers[3] = {NULL, NULL, NULL};
    for (int i = 0; i < 3; ++i) {
        status = kora_buffer_create(runtime, COUNT * sizeof(uint32_t), &buffers[i]);
        if (status != KORA_STATUS_OK) return fail("kora_buffer_create", status);
    }
    if ((status = kora_buffer_write(buffers[0], (const uint8_t*)a, COUNT * sizeof(uint32_t))) != KORA_STATUS_OK ||
        (status = kora_buffer_write(buffers[1], (const uint8_t*)b, COUNT * sizeof(uint32_t))) != KORA_STATUS_OK)
        return fail("kora_buffer_write", status);

    KoraDispatch* dispatch = NULL;
    KoraEvent* event = NULL;
    const uint32_t count = COUNT;
    if ((status = kora_dispatch_create(runtime, kernel, &dispatch)) != KORA_STATUS_OK ||
        (status = kora_dispatch_set_global(dispatch, COUNT, 1, 1)) != KORA_STATUS_OK ||
        (status = kora_dispatch_bind(dispatch, buffers[0], KORA_ACCESS_READ)) != KORA_STATUS_OK ||
        (status = kora_dispatch_bind(dispatch, buffers[1], KORA_ACCESS_READ)) != KORA_STATUS_OK ||
        (status = kora_dispatch_bind(dispatch, buffers[2], KORA_ACCESS_WRITE)) != KORA_STATUS_OK ||
        (status = kora_dispatch_push(dispatch, (const uint8_t*)&count, sizeof count)) != KORA_STATUS_OK ||
        (status = kora_dispatch_submit(dispatch, &event)) != KORA_STATUS_OK ||
        (status = kora_event_wait(event)) != KORA_STATUS_OK)
        return fail("dispatch", status);
    if ((status = kora_buffer_read(buffers[2], (uint8_t*)out, COUNT * sizeof(uint32_t))) != KORA_STATUS_OK)
        return fail("kora_buffer_read", status);

    int mismatches = 0;
    for (uint32_t i = 0; i < COUNT; ++i) mismatches += out[i] != 3 * i;
    printf("vector add of %u elements: %d mismatches\n", COUNT, mismatches);

    kora_event_destroy(event);
    kora_dispatch_destroy(dispatch);
    for (int i = 0; i < 3; ++i) kora_buffer_destroy(buffers[i]);
    kora_kernel_destroy(kernel);
    kora_runtime_destroy(runtime);
    free(a);
    free(b);
    free(out);
    return mismatches == 0 ? 0 : 1;
}
