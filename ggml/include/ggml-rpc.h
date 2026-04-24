#pragma once

#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#define RPC_PROTO_MAJOR_VERSION    4
#define RPC_PROTO_MINOR_VERSION    1
#define RPC_PROTO_PATCH_VERSION    0

#ifdef  __cplusplus
static_assert(GGML_OP_COUNT == 96, "GGML_OP_COUNT has changed - update RPC_PROTO_PATCH_VERSION");
#endif

#define GGML_RPC_MAX_SERVERS       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device);
GGML_BACKEND_API bool ggml_backend_is_rpc(ggml_backend_t backend);

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device);

GGML_BACKEND_API void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total);

GGML_BACKEND_API void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                                    size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_reg(void);
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint);

// Load-time profiling (parallel-rpc-loading branch).
// Reset clears per-endpoint counters and starts the wall-clock timer.
// Log emits a per-worker summary of set_tensor/init_tensor activity.
// Both are also retrievable via ggml_backend_reg_get_proc_address by name.
GGML_BACKEND_API void ggml_backend_rpc_load_stats_reset(void);
GGML_BACKEND_API void ggml_backend_rpc_log_load_stats(void);

// Worker-side file-read variant of set_tensor (parallel-rpc-loading branch).
// The worker opens `file_path`, reads `size` bytes from `file_offset`, and writes
// them into the destination tensor at `tensor_offset`. Bytes do not flow through
// the head — only the request envelope. Returns false if the buffer is not RPC
// or the worker can't satisfy the read; caller should fall back to set_tensor.
GGML_BACKEND_API bool ggml_backend_rpc_buffer_set_tensor_from_file(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const char * file_path,
        uint64_t file_offset,
        uint64_t tensor_offset,
        uint64_t size);

// Fire-and-forget variant: dispatches without waiting for the response; a
// per-endpoint receiver thread drains replies in FIFO order. Caller MUST call
// ggml_backend_rpc_flush_pending_reads() before observing tensor data to
// ensure all reads have completed and to surface errors.
GGML_BACKEND_API bool ggml_backend_rpc_buffer_set_tensor_from_file_async(
        ggml_backend_buffer_t buffer,
        struct ggml_tensor * tensor,
        const char * file_path,
        uint64_t file_offset,
        uint64_t tensor_offset,
        uint64_t size);

// Waits for all in-flight async set_tensor_from_file reads across all
// endpoints, then tears down per-endpoint receiver threads. Returns false if
// any pipeline observed an error. No-op if no pipelines were started.
GGML_BACKEND_API bool ggml_backend_rpc_flush_pending_reads(void);

#ifdef  __cplusplus
}
#endif
