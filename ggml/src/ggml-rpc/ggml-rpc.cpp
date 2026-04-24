#include "ggml-rpc.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "ggml-cpp.h"
#include "transport.h"

#include <array>
#include <atomic>
#include <cinttypes>
#include <condition_variable>
#include <optional>
#include <queue>
#include <string>
#include <thread>
#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <algorithm>


static const char * RPC_DEBUG = std::getenv("GGML_RPC_DEBUG");

#define LOG_DBG(...) \
    do { if (RPC_DEBUG) GGML_LOG_DEBUG(__VA_ARGS__); } while (0)


namespace fs = std::filesystem;

// macro for nicer error messages on server crash
#define RPC_STATUS_ASSERT(x) if (!(x)) GGML_ABORT("Remote RPC server crashed or returned malformed response")

// all RPC structures must be packed
#pragma pack(push, 1)
// ggml_tensor is serialized into rpc_tensor
struct rpc_tensor {
    uint64_t id;
    uint32_t type;
    uint64_t buffer;
    uint32_t ne[GGML_MAX_DIMS];
    uint32_t nb[GGML_MAX_DIMS];
    uint32_t op;
    int32_t  op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
    int32_t  flags;
    uint64_t src[GGML_MAX_SRC];
    uint64_t view_src;
    uint64_t view_offs;
    uint64_t data;
    char name[GGML_MAX_NAME];

    char padding[4];
};

static_assert(sizeof(rpc_tensor) % 8 == 0, "rpc_tensor size must be multiple of 8");

// RPC commands
enum rpc_cmd {
    RPC_CMD_ALLOC_BUFFER = 0,
    RPC_CMD_GET_ALIGNMENT,
    RPC_CMD_GET_MAX_SIZE,
    RPC_CMD_BUFFER_GET_BASE,
    RPC_CMD_FREE_BUFFER,
    RPC_CMD_BUFFER_CLEAR,
    RPC_CMD_SET_TENSOR,
    RPC_CMD_SET_TENSOR_HASH,
    RPC_CMD_GET_TENSOR,
    RPC_CMD_COPY_TENSOR,
    RPC_CMD_GRAPH_COMPUTE,
    RPC_CMD_GET_DEVICE_MEMORY,
    RPC_CMD_INIT_TENSOR,
    RPC_CMD_GET_ALLOC_SIZE,
    RPC_CMD_HELLO,
    RPC_CMD_DEVICE_COUNT,
    RPC_CMD_GRAPH_RECOMPUTE,
    // parallel-rpc-loading branch: worker reads tensor bytes from a shared-fs path
    RPC_CMD_SET_TENSOR_FROM_FILE,
    RPC_CMD_COUNT,
};

static_assert(RPC_CMD_HELLO == 14, "RPC_CMD_HELLO must be always 14");

// Try RPC_CMD_SET_TENSOR_HASH first when data size is larger than this threshold
const size_t HASH_THRESHOLD = 10 * 1024 * 1024;

struct rpc_msg_hello_req {
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_hello_rsp {
    uint8_t major;
    uint8_t minor;
    uint8_t patch;
    uint8_t padding;
    uint8_t conn_caps[RPC_CONN_CAPS_SIZE];
};

struct rpc_msg_device_count_rsp {
    uint32_t device_count;
};

struct rpc_msg_get_alloc_size_req {
    uint32_t   device;
    rpc_tensor tensor;
    rpc_tensor srcs[GGML_MAX_SRC];
};

struct rpc_msg_get_alloc_size_rsp {
    uint64_t alloc_size;
};

struct rpc_msg_init_tensor_req {
    rpc_tensor tensor;
};

struct rpc_msg_alloc_buffer_req {
    uint32_t device;
    uint64_t size;
};

struct rpc_msg_alloc_buffer_rsp {
    uint64_t remote_ptr;
    uint64_t remote_size;
};

struct rpc_msg_get_alignment_req {
    uint32_t device;
};

struct rpc_msg_get_alignment_rsp {
    uint64_t alignment;
};

struct rpc_msg_get_max_size_req {
    uint32_t device;
};

struct rpc_msg_get_max_size_rsp {
    uint64_t max_size;
};

struct rpc_msg_buffer_get_base_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_get_base_rsp {
    uint64_t base_ptr;
};

struct rpc_msg_free_buffer_req {
    uint64_t remote_ptr;
};

struct rpc_msg_buffer_clear_req {
    uint64_t remote_ptr;
    uint8_t value;
};

struct rpc_msg_set_tensor_hash_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t hash;
};

struct rpc_msg_set_tensor_hash_rsp {
    uint8_t result;
};

struct rpc_msg_get_tensor_req {
    rpc_tensor tensor;
    uint64_t offset;
    uint64_t size;
};

struct rpc_msg_copy_tensor_req {
    rpc_tensor src;
    rpc_tensor dst;
};

struct rpc_msg_copy_tensor_rsp {
    uint8_t result;
};

struct rpc_msg_get_device_memory_req {
    uint32_t device;
};

struct rpc_msg_get_device_memory_rsp {
    uint64_t free_mem;
    uint64_t total_mem;
};

struct rpc_msg_graph_recompute_req {
    uint32_t device;
};

// SET_TENSOR_FROM_FILE request body:
//   | rpc_tensor | tensor_offset (8) | file_offset (8) | size (8) | path_len (4) | path[path_len] |
// The variable-length tail makes this a vector<uint8_t> on the wire (like SET_TENSOR).
struct rpc_msg_set_tensor_from_file_rsp {
    uint8_t  result;       // 1 = success, 0 = failure (worker fell back; caller should push)
    uint64_t bytes_read;   // bytes the worker actually read from the file (0 on failure)
};

#pragma pack(pop)

// RPC data structures

static ggml_guid_t ggml_backend_rpc_guid() {
    static ggml_guid guid = {0x99, 0x68, 0x5b, 0x6c, 0xd2, 0x83, 0x3d, 0x24, 0x25, 0x36, 0x72, 0xe1, 0x5b, 0x0e, 0x14, 0x03};
    return &guid;
}

struct ggml_backend_rpc_buffer_type_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    size_t      alignment;
    size_t      max_size;
};

struct graph_cache {

    bool is_cached(const ggml_cgraph * cgraph) {
        if ((int)last_graph.size() != cgraph->n_nodes) {
            return false;
        }
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (memcmp(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor)) != 0) {
                return false;
            }
        }
        return true;
    }

    void add(const ggml_cgraph * cgraph) {
        last_graph.resize(cgraph->n_nodes);
        for (int i = 0; i < cgraph->n_nodes; i++) {
            memcpy(&last_graph[i], cgraph->nodes[i], sizeof(ggml_tensor));
        }
    }

    std::vector<ggml_tensor> last_graph;
};

struct ggml_backend_rpc_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    graph_cache gc;
};

struct ggml_backend_rpc_buffer_context {
    std::shared_ptr<socket_t> sock;
    void * base_ptr;
    uint64_t remote_ptr;
    std::string endpoint;
};

// --- Load-time profiling (parallel-rpc-loading branch) -----------------------
// Per-endpoint accumulator for what set_tensor / init_tensor cost during model
// load. Always-on (counters are tiny); the dump is gated by the
// LLAMA_RPC_LOAD_PROFILE env var inside the loader.

struct rpc_load_stats {
    uint64_t n_set_tensor       = 0;  // total set_tensor calls
    uint64_t n_init_tensor      = 0;  // total init_tensor RPCs
    uint64_t n_hash_check       = 0;  // SET_TENSOR_HASH attempts
    uint64_t n_hash_hit         = 0;  // hash cache hits (no data sent)
    uint64_t n_worker_read      = 0;  // SET_TENSOR_FROM_FILE successes (worker read from shared fs)
    uint64_t n_worker_read_miss = 0;  // SET_TENSOR_FROM_FILE failures that fell back to push
    uint64_t bytes_data         = 0;  // tensor bytes set (logical)
    uint64_t bytes_sent         = 0;  // bytes actually sent over the wire (head→worker)
    uint64_t bytes_skipped      = 0;  // bytes avoided via hash hit
    uint64_t bytes_worker_read  = 0;  // bytes the worker pulled directly from shared fs
    int64_t  t_set_us           = 0;  // wall time inside SET_TENSOR sends
    int64_t  t_hash_us          = 0;  // wall time inside SET_TENSOR_HASH round-trips
    int64_t  t_init_us          = 0;  // wall time inside INIT_TENSOR round-trips
    int64_t  t_worker_read_us   = 0;  // wall time inside SET_TENSOR_FROM_FILE round-trips
};

static std::mutex                                 g_rpc_stats_mutex;
static std::unordered_map<std::string, rpc_load_stats> g_rpc_stats;
static int64_t                                    g_rpc_stats_t_start_us = 0;  // wall-clock start of current load

static rpc_load_stats & rpc_stats_for(const std::string & endpoint) {
    // caller holds g_rpc_stats_mutex
    return g_rpc_stats[endpoint];
}

// --- Pipelined worker-read (B1: non-blocking SET_TENSOR_FROM_FILE) ----------
// Per-endpoint background receiver thread drains command responses in FIFO
// order while the head dispatches more. Each worker has its own socket, so
// firing async dispatches at all three lets them read concurrently.
// Auto-started on first _async dispatch for an endpoint; torn down by
// ggml_backend_rpc_flush_pending_reads().
// Assumes no concurrent sync RPC calls on the same endpoint while active.
struct rpc_load_pipeline {
    std::shared_ptr<socket_t> sock;
    std::thread               recv_thread;
    std::atomic<bool>         stop{false};
    std::atomic<int64_t>      pending{0};
    std::mutex                mu;
    std::condition_variable   cv;
    std::atomic<bool>         had_error{false};
    uint64_t                  n_ok   = 0;
    uint64_t                  n_fail = 0;
};

static std::mutex                                                           g_rpc_pipe_mu;
static std::unordered_map<std::string, std::unique_ptr<rpc_load_pipeline>>  g_rpc_pipes;

static void rpc_load_recv_loop(rpc_load_pipeline * p) {
    while (true) {
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->cv.wait(lk, [p]{ return p->pending.load() > 0 || p->stop.load(); });
            if (p->stop.load() && p->pending.load() == 0) {
                return;
            }
        }
        uint64_t out_size = 0;
        if (!p->sock->recv_data(&out_size, sizeof(out_size))) {
            p->had_error.store(true);
            std::lock_guard<std::mutex> lk(p->mu);
            p->pending.store(0);
            p->cv.notify_all();
            return;
        }
        rpc_msg_set_tensor_from_file_rsp response = {};
        if (out_size != sizeof(response) ||
            !p->sock->recv_data(&response, sizeof(response))) {
            p->had_error.store(true);
            std::lock_guard<std::mutex> lk(p->mu);
            p->pending.store(0);
            p->cv.notify_all();
            return;
        }
        {
            std::lock_guard<std::mutex> lk(p->mu);
            if (response.result) {
                ++p->n_ok;
            } else {
                ++p->n_fail;
                p->had_error.store(true);
            }
            --p->pending;
            p->cv.notify_all();
        }
    }
}

static rpc_load_pipeline * get_or_start_pipeline(const std::string & endpoint, std::shared_ptr<socket_t> sock) {
    std::lock_guard<std::mutex> lk(g_rpc_pipe_mu);
    auto it = g_rpc_pipes.find(endpoint);
    if (it != g_rpc_pipes.end()) {
        return it->second.get();
    }
    auto p = std::make_unique<rpc_load_pipeline>();
    p->sock = std::move(sock);
    rpc_load_pipeline * ptr = p.get();
    p->recv_thread = std::thread(rpc_load_recv_loop, ptr);
    g_rpc_pipes.emplace(endpoint, std::move(p));
    return ptr;
}

// RPC helper functions

// Computes FNV-1a hash of the data
static uint64_t fnv_hash(const uint8_t * data, size_t len) {
    const uint64_t fnv_prime = 0x100000001b3ULL;
    uint64_t hash = 0xcbf29ce484222325ULL;

    for (size_t i = 0; i < len; ++i) {
        hash ^= data[i];
        hash *= fnv_prime;
    }
    return hash;
}

static bool send_msg(socket_ptr sock, const void * msg, size_t msg_size) {
    if (!sock->send_data(&msg_size, sizeof(msg_size))) {
        return false;
    }
    return sock->send_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, void * msg, size_t msg_size) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    if (size != msg_size) {
        return false;
    }
    return sock->recv_data(msg, msg_size);
}

static bool recv_msg(socket_ptr sock, std::vector<uint8_t> & input) {
    uint64_t size;
    if (!sock->recv_data(&size, sizeof(size))) {
        return false;
    }
    try {
        input.resize(size);
    } catch (const std::bad_alloc & e) {
        GGML_LOG_ERROR("Failed to allocate input buffer of size %" PRIu64 "\n", size);
        return false;
    }
    return sock->recv_data(input.data(), size);
}

static bool parse_endpoint(const std::string & endpoint, std::string & host, int & port) {
    size_t pos = endpoint.find(':');
    if (pos == std::string::npos) {
        return false;
    }
    host = endpoint.substr(0, pos);
    try {
        port = std::stoi(endpoint.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// No response
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size) {
    uint8_t cmd_byte = cmd;
    if (!sock->send_data(&cmd_byte, sizeof(cmd_byte))) {
        return false;
    }
    if (!sock->send_data(&input_size, sizeof(input_size))) {
        return false;
    }
    if (!sock->send_data(input, input_size)) {
        return false;
    }
    return true;
}

// RPC request : | rpc_cmd (1 byte) | request_size (8 bytes) | request_data (request_size bytes) |
// RPC response: | response_size (8 bytes) | response_data (response_size bytes) |
static bool send_rpc_cmd(socket_ptr sock, enum rpc_cmd cmd, const void * input, size_t input_size, void * output, size_t output_size) {
    if (!send_rpc_cmd(sock, cmd, input, input_size)) {
        return false;
    }
    uint64_t out_size;
    if (!sock->recv_data(&out_size, sizeof(out_size))) {
        return false;
    }
    if (out_size != output_size) {
        return false;
    }
    if (!sock->recv_data(output, output_size)) {
        return false;
    }
    return true;
}

// RPC client-side implementation

// Performs HELLO handshake with transport auto-negotiation.
// Advertises local capabilities via conn_caps; if the server responds with
// matching capabilities, the socket is upgraded transparently.
static bool negotiate_hello(const std::shared_ptr<socket_t> & sock) {
    rpc_msg_hello_req request = {};
    rpc_msg_hello_rsp response = {};

    sock->get_caps(request.conn_caps);

    bool status = send_rpc_cmd(sock, RPC_CMD_HELLO, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);

    if (response.major != RPC_PROTO_MAJOR_VERSION || response.minor > RPC_PROTO_MINOR_VERSION) {
        GGML_LOG_ERROR("RPC server version mismatch: %d.%d.%d\n",
                       response.major, response.minor, response.patch);
        return false;
    }

    sock->update_caps(response.conn_caps);
    return true;
}

static std::shared_ptr<socket_t> get_socket(const std::string & endpoint) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    static std::unordered_map<std::string, std::weak_ptr<socket_t>> sockets;

    auto it = sockets.find(endpoint);
    if (it != sockets.end()) {
        if (auto sock = it->second.lock()) {
            return sock;
        }
    }
    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        GGML_LOG_ERROR("Failed to parse endpoint: %s\n", endpoint.c_str());
        return nullptr;
    }

    if (!rpc_transport_init()) {
        return nullptr;
    }
    auto sock = socket_t::connect(host.c_str(), port);
    if (sock == nullptr) {
        return nullptr;
    }
    if (!negotiate_hello(sock)) {
        return nullptr;
    }
    LOG_DBG("[%s] connected to %s\n", __func__, endpoint.c_str());
    sockets[endpoint] = sock;
    return sock;
}

static void ggml_backend_rpc_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_free_buffer_req request = {ctx->remote_ptr};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_FREE_BUFFER, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
    delete ctx;
}

static void * ggml_backend_rpc_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    if (ctx->base_ptr != nullptr) {
        return ctx->base_ptr;
    }
    rpc_msg_buffer_get_base_req request = {ctx->remote_ptr};
    rpc_msg_buffer_get_base_rsp response;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_GET_BASE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    ctx->base_ptr = reinterpret_cast<void *>(response.base_ptr);
    return ctx->base_ptr;
}

static bool ggml_backend_buffer_is_rpc(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_rpc_buffer_free_buffer;
}

static rpc_tensor serialize_tensor(const ggml_tensor * tensor) {
    rpc_tensor result;
    if (!tensor) {
        memset(&result, 0, sizeof(result));
        return result;
    }

    result.id = reinterpret_cast<uint64_t>(tensor);
    result.type = tensor->type;
    if (tensor->buffer && ggml_backend_buffer_is_rpc(tensor->buffer)) {
        ggml_backend_buffer_t buffer = tensor->buffer;
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        result.buffer = ctx != nullptr ? ctx->remote_ptr : 0;
        result.data = reinterpret_cast<uint64_t>(tensor->data);
    } else {
        result.buffer = 0;
        result.data   = 0;
    }
    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result.ne[i] = tensor->ne[i];
        result.nb[i] = tensor->nb[i];
    }
    result.op = tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result.op_params[i] = tensor->op_params[i];
    }
    result.flags = tensor->flags;
    for (uint32_t i = 0; i < GGML_MAX_SRC; i++) {
        result.src[i] = reinterpret_cast<uint64_t>(tensor->src[i]);
    }
    result.view_src = reinterpret_cast<uint64_t>(tensor->view_src);
    result.view_offs = tensor->view_offs;

    // Avoid sending uninitialized data over the wire
    memset(result.name, 0, sizeof(result.name));
    memset(result.padding, 0, sizeof(result.padding));

    snprintf(result.name, GGML_MAX_NAME, "%s", tensor->name);
    return result;
}

static enum ggml_status ggml_backend_rpc_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;

    // CUDA backend on the server pads everything to 512 due to CUDA limitations.
    // Due to bandwidth constraints, we only call the server init tensor functions if necessary.
    // In particular, only quantized tensors need padding
    if (ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr)) {
        rpc_msg_init_tensor_req request;

        request.tensor = serialize_tensor(tensor);

        const int64_t t0 = ggml_time_us();
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_INIT_TENSOR, &request, sizeof(request), nullptr, 0);
        const int64_t dt = ggml_time_us() - t0;
        {
            std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
            auto & s = rpc_stats_for(ctx->endpoint);
            s.n_init_tensor++;
            s.t_init_us += dt;
        }
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

// Worker-side file-read variant. Returns true on success (worker pulled the bytes
// from the shared filesystem itself). Returns false on any failure, including:
//   - buffer is not an RPC buffer
//   - worker can't open the file or read fails
//   - protocol mismatch with an older server
// On false, the caller is expected to fall back to ggml_backend_rpc_buffer_set_tensor.
extern "C" bool ggml_backend_rpc_buffer_set_tensor_from_file(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const char * file_path,
        uint64_t file_offset,
        uint64_t tensor_offset,
        uint64_t size) {
    if (buffer == nullptr || tensor == nullptr || file_path == nullptr) {
        return false;
    }
    if (!ggml_backend_buffer_is_rpc(buffer)) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rt = serialize_tensor(tensor);

    // Build the variable-length request body.
    const uint32_t path_len = (uint32_t) std::strlen(file_path);
    const size_t input_size = sizeof(rpc_tensor) + 3 * sizeof(uint64_t) + sizeof(uint32_t) + path_len;
    std::vector<uint8_t> input(input_size);
    size_t off = 0;
    std::memcpy(input.data() + off, &rt, sizeof(rt));            off += sizeof(rt);
    std::memcpy(input.data() + off, &tensor_offset, sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(input.data() + off, &file_offset,   sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(input.data() + off, &size,          sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(input.data() + off, &path_len,      sizeof(uint32_t)); off += sizeof(uint32_t);
    std::memcpy(input.data() + off, file_path, path_len);

    rpc_msg_set_tensor_from_file_rsp response = {};
    const int64_t t0 = ggml_time_us();
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_FROM_FILE,
                               input.data(), input.size(),
                               &response, sizeof(response));
    const int64_t dt = ggml_time_us() - t0;
    if (!status) {
        // Old worker that doesn't know the command will close the socket on cmd>=COUNT.
        // We can't recover gracefully from that here — abort so the user notices.
        RPC_STATUS_ASSERT(status);
    }
    {
        std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
        auto & s = rpc_stats_for(ctx->endpoint);
        s.t_worker_read_us += dt;
        if (response.result) {
            s.n_set_tensor++;
            s.n_worker_read++;
            s.bytes_data        += size;
            s.bytes_worker_read += response.bytes_read;
        } else {
            s.n_worker_read_miss++;
        }
    }
    return response.result != 0;
}

// Fire-and-forget variant. Sends RPC_CMD_SET_TENSOR_FROM_FILE without blocking
// for the response — a per-endpoint receiver thread drains responses in FIFO
// order. Returns true if the send succeeded; actual read success is observed
// later via ggml_backend_rpc_flush_pending_reads().
extern "C" bool ggml_backend_rpc_buffer_set_tensor_from_file_async(
        ggml_backend_buffer_t buffer,
        ggml_tensor * tensor,
        const char * file_path,
        uint64_t file_offset,
        uint64_t tensor_offset,
        uint64_t size) {
    if (buffer == nullptr || tensor == nullptr || file_path == nullptr) {
        return false;
    }
    if (!ggml_backend_buffer_is_rpc(buffer)) {
        return false;
    }
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rt = serialize_tensor(tensor);

    const uint32_t path_len = (uint32_t) std::strlen(file_path);
    const size_t input_size = sizeof(rpc_tensor) + 3 * sizeof(uint64_t) + sizeof(uint32_t) + path_len;
    std::vector<uint8_t> input(input_size);
    size_t off = 0;
    std::memcpy(input.data() + off, &rt, sizeof(rt));                  off += sizeof(rt);
    std::memcpy(input.data() + off, &tensor_offset, sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(input.data() + off, &file_offset,   sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(input.data() + off, &size,          sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(input.data() + off, &path_len,      sizeof(uint32_t)); off += sizeof(uint32_t);
    std::memcpy(input.data() + off, file_path, path_len);

    rpc_load_pipeline * pipe = get_or_start_pipeline(ctx->endpoint, ctx->sock);
    {
        std::lock_guard<std::mutex> lk(pipe->mu);
        ++pipe->pending;
        pipe->cv.notify_all();
    }
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_FROM_FILE,
                               input.data(), input.size());
    if (!status) {
        std::lock_guard<std::mutex> lk(pipe->mu);
        --pipe->pending;
        pipe->had_error.store(true);
        pipe->cv.notify_all();
        RPC_STATUS_ASSERT(status);
    }
    {
        std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
        auto & s = rpc_stats_for(ctx->endpoint);
        s.n_set_tensor++;
        s.n_worker_read++;
        s.bytes_data        += size;
        s.bytes_worker_read += size;
    }
    return true;
}

// Waits for all in-flight async set_tensor_from_file reads across all
// endpoints, then tears down per-endpoint receiver threads. Returns false if
// any pipeline observed an error (failed read, socket error, or non-success
// response).
extern "C" bool ggml_backend_rpc_flush_pending_reads(void) {
    std::unordered_map<std::string, std::unique_ptr<rpc_load_pipeline>> pipes;
    {
        std::lock_guard<std::mutex> lk(g_rpc_pipe_mu);
        pipes.swap(g_rpc_pipes);
    }
    bool ok = true;
    for (auto & kv : pipes) {
        auto & p = kv.second;
        {
            std::unique_lock<std::mutex> lk(p->mu);
            p->cv.wait(lk, [&]{ return p->pending.load() == 0 || p->had_error.load(); });
            p->stop.store(true);
            p->cv.notify_all();
        }
        if (p->recv_thread.joinable()) {
            p->recv_thread.join();
        }
        if (p->had_error.load()) {
            ok = false;
        }
    }
    return ok;
}

static void ggml_backend_rpc_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_tensor rpc_tensor = serialize_tensor(tensor);
    bool hash_hit = false;
    int64_t t_hash_us = 0;
    if (size > HASH_THRESHOLD) {
        rpc_msg_set_tensor_hash_req request;
        request.tensor = rpc_tensor;
        request.offset = offset;
        request.hash = fnv_hash((const uint8_t*)data, size);
        rpc_msg_set_tensor_hash_rsp response;
        const int64_t t0 = ggml_time_us();
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR_HASH, &request, sizeof(request), &response, sizeof(response));
        t_hash_us = ggml_time_us() - t0;
        RPC_STATUS_ASSERT(status);
        if (response.result) {
            // the server has the same data, no need to send it
            hash_hit = true;
        }
    }
    if (hash_hit) {
        std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
        auto & s = rpc_stats_for(ctx->endpoint);
        s.n_set_tensor++;
        s.n_hash_check++;
        s.n_hash_hit++;
        s.bytes_data    += size;
        s.bytes_skipped += size;
        s.t_hash_us     += t_hash_us;
        return;
    }
    // input serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes)
    size_t input_size = sizeof(rpc_tensor) + sizeof(uint64_t) + size;
    std::vector<uint8_t> input(input_size, 0);
    memcpy(input.data(), &rpc_tensor, sizeof(rpc_tensor));
    memcpy(input.data() + sizeof(rpc_tensor), &offset, sizeof(offset));
    memcpy(input.data() + sizeof(rpc_tensor) + sizeof(offset), data, size);
    const int64_t t0 = ggml_time_us();
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_SET_TENSOR, input.data(), input.size());
    const int64_t dt = ggml_time_us() - t0;
    {
        std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
        auto & s = rpc_stats_for(ctx->endpoint);
        s.n_set_tensor++;
        if (size > HASH_THRESHOLD) {
            s.n_hash_check++;       // attempted but missed
            s.t_hash_us += t_hash_us;
        }
        s.bytes_data += size;
        s.bytes_sent += input_size;
        s.t_set_us   += dt;
    }
    RPC_STATUS_ASSERT(status);
}

static void ggml_backend_rpc_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_get_tensor_req request;
    request.tensor = serialize_tensor(tensor);
    request.offset = offset;
    request.size = size;
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_GET_TENSOR, &request, sizeof(request), data, size);
    RPC_STATUS_ASSERT(status);
}

static bool ggml_backend_rpc_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_rpc(src->buffer)) {
        // check if src and dst are on the same server
        ggml_backend_buffer_t src_buffer = src->buffer;
        ggml_backend_rpc_buffer_context * src_ctx = (ggml_backend_rpc_buffer_context *)src_buffer->context;
        ggml_backend_buffer_t dst_buffer = dst->buffer;
        ggml_backend_rpc_buffer_context * dst_ctx = (ggml_backend_rpc_buffer_context *)dst_buffer->context;
        if (src_ctx->sock != dst_ctx->sock) {
            return false;
        }
        ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
        rpc_msg_copy_tensor_req request;
        request.src = serialize_tensor(src);
        request.dst = serialize_tensor(dst);
        rpc_msg_copy_tensor_rsp response;
        bool status = send_rpc_cmd(ctx->sock, RPC_CMD_COPY_TENSOR, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);
        return response.result;
    }
    return false;
}

static void ggml_backend_rpc_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_rpc_buffer_context * ctx = (ggml_backend_rpc_buffer_context *)buffer->context;
    rpc_msg_buffer_clear_req request = {ctx->remote_ptr, value};
    bool status = send_rpc_cmd(ctx->sock, RPC_CMD_BUFFER_CLEAR, &request, sizeof(request), nullptr, 0);
    RPC_STATUS_ASSERT(status);
}

static ggml_backend_buffer_i ggml_backend_rpc_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_rpc_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_rpc_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_rpc_buffer_init_tensor,
    /* .memset_tensor   = */ NULL,
    /* .set_tensor      = */ ggml_backend_rpc_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_rpc_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_rpc_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_rpc_buffer_clear,
    /* .reset           = */ NULL,
};

static const char * ggml_backend_rpc_buffer_type_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_rpc_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    rpc_msg_alloc_buffer_req request = {buft_ctx->device, size};
    rpc_msg_alloc_buffer_rsp response;
    auto sock = get_socket(buft_ctx->endpoint);
    bool status = send_rpc_cmd(sock, RPC_CMD_ALLOC_BUFFER, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    if (response.remote_ptr != 0) {
        ggml_backend_buffer_t buffer = ggml_backend_buffer_init(buft,
            ggml_backend_rpc_buffer_interface,
            new ggml_backend_rpc_buffer_context{sock, nullptr, response.remote_ptr, buft_ctx->endpoint},
            response.remote_size);
        return buffer;
    } else {
        return nullptr;
    }
}

static size_t get_alignment(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_alignment_req request = {device};
    rpc_msg_get_alignment_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALIGNMENT, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.alignment;
}

static size_t ggml_backend_rpc_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->alignment;
}

static size_t get_max_size(const std::shared_ptr<socket_t> & sock, uint32_t device) {
    rpc_msg_get_max_size_req request = {device};
    rpc_msg_get_max_size_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_MAX_SIZE, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.max_size;
}

static size_t ggml_backend_rpc_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    return buft_ctx->max_size;
}

static size_t ggml_backend_rpc_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    // should we query the remote server for the actual size
    bool rpc_get = false;

    // See comments in init_tensor.
    rpc_get |= ggml_is_quantized(tensor->type) && (tensor->ne[0] % 512 != 0) && (tensor->view_src == nullptr);

    // ops that require additional memory for fleeting data on certain backends
    // ref: https://github.com/ggml-org/llama.cpp/pull/15966
    rpc_get |= tensor->op == GGML_OP_FLASH_ATTN_EXT;
    rpc_get |= tensor->op == GGML_OP_MUL_MAT_ID;

    if (rpc_get) {
        ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
        auto sock = get_socket(buft_ctx->endpoint);

        rpc_msg_get_alloc_size_req request = {
            /*.device =*/ buft_ctx->device,
            /*.tensor =*/ serialize_tensor(tensor),
            /*.srcs   =*/ {},
        };

        // .get_alloc_size could be a function of the tensor's srcs, so we must serialize them as well
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            request.srcs[i] = serialize_tensor(tensor->src[i]);
        }

        // TODO: cache the alloc responses to avoid extra RPC calls?
        rpc_msg_get_alloc_size_rsp response;
        bool status = send_rpc_cmd(sock, RPC_CMD_GET_ALLOC_SIZE, &request, sizeof(request), &response, sizeof(response));
        RPC_STATUS_ASSERT(status);

        return response.alloc_size;
    }

    return ggml_nbytes(tensor);
}

static ggml_backend_buffer_type_i ggml_backend_rpc_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_rpc_buffer_type_name,
    /* .alloc_buffer     = */ ggml_backend_rpc_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_rpc_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_rpc_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_rpc_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

static const char * ggml_backend_rpc_name(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    return rpc_ctx->name.c_str();
}

static void ggml_backend_rpc_free(ggml_backend_t backend) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;
    delete rpc_ctx;
    delete backend;
}

static void ggml_backend_rpc_synchronize(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    // this is no-op because we don't have any async operations
}

static void add_tensor(ggml_tensor * tensor, std::vector<rpc_tensor> & tensors, std::unordered_set<ggml_tensor*> & visited) {
    if (tensor == nullptr) {
        return;
    }
    if (visited.find(tensor) != visited.end()) {
        return;
    }
    visited.insert(tensor);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        add_tensor(tensor->src[i], tensors, visited);
    }
    add_tensor(tensor->view_src, tensors, visited);
    tensors.push_back(serialize_tensor(tensor));
}

static void serialize_graph(uint32_t device, const ggml_cgraph * cgraph, std::vector<uint8_t> & output) {
    uint32_t n_nodes = cgraph->n_nodes;
    std::vector<rpc_tensor> tensors;
    std::unordered_set<ggml_tensor*> visited;
    for (uint32_t i = 0; i < n_nodes; i++) {
        add_tensor(cgraph->nodes[i], tensors, visited);
    }
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    uint32_t n_tensors = tensors.size();
    int output_size = 2*sizeof(uint32_t) + n_nodes * sizeof(uint64_t) + sizeof(uint32_t) + n_tensors * sizeof(rpc_tensor);
    output.resize(output_size, 0);
    uint8_t * dest = output.data();
    memcpy(dest, &device, sizeof(device));
    dest += sizeof(device);
    memcpy(dest, &n_nodes, sizeof(n_nodes));
    dest += sizeof(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        memcpy(dest + i * sizeof(uint64_t), &cgraph->nodes[i], sizeof(uint64_t));
    }
    dest += n_nodes * sizeof(uint64_t);
    memcpy(dest, &n_tensors, sizeof(n_tensors));
    dest += sizeof(n_tensors);
    rpc_tensor * out_tensors = (rpc_tensor *)dest;
    memcpy(out_tensors, tensors.data(), n_tensors * sizeof(rpc_tensor));
}

static enum ggml_status ggml_backend_rpc_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_rpc_context * rpc_ctx = (ggml_backend_rpc_context *)backend->context;

    GGML_ASSERT(cgraph->n_nodes > 0);
    bool reuse = rpc_ctx->gc.is_cached(cgraph);
    if (reuse) {
        rpc_msg_graph_recompute_req request;
        request.device = rpc_ctx->device;
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_RECOMPUTE, &request, sizeof(request));
        RPC_STATUS_ASSERT(status);
    } else {
        rpc_ctx->gc.add(cgraph);
        std::vector<uint8_t> input;
        serialize_graph(rpc_ctx->device, cgraph, input);
        auto sock = get_socket(rpc_ctx->endpoint);
        bool status = send_rpc_cmd(sock, RPC_CMD_GRAPH_COMPUTE, input.data(), input.size());
        RPC_STATUS_ASSERT(status);
    }
    return GGML_STATUS_SUCCESS;
}

static ggml_backend_i ggml_backend_rpc_interface = {
    /* .get_name                = */ ggml_backend_rpc_name,
    /* .free                    = */ ggml_backend_rpc_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .synchronize             = */ ggml_backend_rpc_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_rpc_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_rpc_buffer_type(const char * endpoint, uint32_t device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::string buft_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    // NOTE: buffer types are allocated and never freed; this is by design
    static std::unordered_map<std::string, ggml_backend_buffer_type_t> buft_map;
    auto it = buft_map.find(buft_name);
    if (it != buft_map.end()) {
        return it->second;
    }
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return nullptr;
    }
    size_t alignment = get_alignment(sock, device);
    size_t max_size = get_max_size(sock, device);
    ggml_backend_rpc_buffer_type_context * buft_ctx = new ggml_backend_rpc_buffer_type_context {
        /* .endpoint  = */ endpoint,
        /* .device    = */ device,
        /* .name      = */ buft_name,
        /* .alignment = */ alignment,
        /* .max_size  = */ max_size
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_buffer_type_t buft = new ggml_backend_buffer_type {
        /* .iface   = */ ggml_backend_rpc_buffer_type_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ buft_ctx
    };
    buft_map[buft_name] = buft;
    return buft;
}

ggml_backend_t ggml_backend_rpc_init(const char * endpoint, uint32_t device) {
    std::string dev_name = "RPC" + std::to_string(device) + "[" + std::string(endpoint) + "]";
    ggml_backend_rpc_context * ctx = new ggml_backend_rpc_context {
        /* .endpoint = */ endpoint,
        /* .device   = */ device,
        /* .name     = */ dev_name,
        /* .gc       = */ {},
    };
    auto reg = ggml_backend_rpc_add_server(endpoint);
    ggml_backend_t backend = new ggml_backend {
        /* .guid    = */ ggml_backend_rpc_guid(),
        /* .iface   = */ ggml_backend_rpc_interface,
        /* .device  = */ ggml_backend_reg_dev_get(reg, device),
        /* .context = */ ctx
    };
    return backend;
}

bool ggml_backend_is_rpc(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_rpc_guid());
}

static void get_device_memory(const std::shared_ptr<socket_t> & sock, uint32_t device, size_t * free, size_t * total) {
    rpc_msg_get_device_memory_req request;
    request.device = device;
    rpc_msg_get_device_memory_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_GET_DEVICE_MEMORY, &request, sizeof(request), &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    *free = response.free_mem;
    *total = response.total_mem;
}

void ggml_backend_rpc_get_device_memory(const char * endpoint, uint32_t device, size_t * free, size_t * total) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        *free = 0;
        *total = 0;
        return;
    }
    get_device_memory(sock, device, free, total);
}

// RPC server-side implementation

class rpc_server {
public:
    rpc_server(std::vector<ggml_backend_t> all_backends, const char * cache_dir)
        : backends(std::move(all_backends)), cache_dir(cache_dir) {
        stored_graphs.resize(backends.size());
        // staging_buffers[backend_idx][thread_idx]. Thread dimension sized to
        // the set_tensor_from_file pool width; +1 for the main-thread
        // fallback slot used when the pool is disabled (pool_size==0).
        staging_buffers.resize(backends.size());
        const size_t slots = pool_size() + 1;
        for (auto & row : staging_buffers) {
            row.resize(slots);
        }
    }
    ~rpc_server();

    // Read the SET_TENSOR_FROM_FILE worker pool size once per process. 0
    // disables the pool (main-thread path). Defaults to 8, overridable via
    // LLAMA_RPC_STFF_POOL env var.
    static size_t pool_size() {
        static size_t cached = [](){
            const char * s = std::getenv("LLAMA_RPC_STFF_POOL");
            if (s != nullptr) {
                long v = std::atol(s);
                if (v >= 0 && v <= 64) {
                    return (size_t) v;
                }
            }
            return (size_t) 8;
        }();
        return cached;
    }

    void hello(rpc_msg_hello_rsp & response);
    bool alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response);
    bool get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response);
    bool get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response);
    bool buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response);
    bool free_buffer(const rpc_msg_free_buffer_req & request);
    bool buffer_clear(const rpc_msg_buffer_clear_req & request);
    bool set_tensor(const std::vector<uint8_t> & input);
    bool set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response);
    bool set_tensor_from_file(const std::vector<uint8_t> & input, rpc_msg_set_tensor_from_file_rsp & response, size_t thread_idx = 0);
    bool get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response);
    bool copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response);
    bool graph_compute(const std::vector<uint8_t> & input);
    bool graph_recompute(const rpc_msg_graph_recompute_req & request);
    bool init_tensor(const rpc_msg_init_tensor_req & request);
    bool get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response);
    bool get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response);

    struct stored_graph {
        std::vector<uint8_t>   buffer;
        ggml_cgraph          * graph;
    };

    // Per-backend pinned staging region for set_tensor_from_file. Allocated
    // lazily on first use via the backend's host buffer type (pinned memory
    // on CUDA/HIP; plain malloc on CPU). Grows as needed to fit the largest
    // tensor seen. See _enhancements/findings.md "Warm-cache rerun" section —
    // HtoD from pageable std::vector was ~340 ms per 294 MiB tensor; from
    // pinned it should be ~15 ms.
    struct staging_buffer {
        ggml_backend_buffer_t buf      = nullptr;
        uint8_t *             ptr      = nullptr;
        size_t                capacity = 0;
    };

private:
    bool get_cached_file(uint64_t hash, std::vector<uint8_t> & data);
    ggml_tensor * deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor);
    ggml_tensor * create_node(uint64_t id,
                              struct ggml_context * ctx,
                              const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                              std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map);

    // Returns a host pointer to a staging region of at least `needed` bytes
    // for `(backend_idx, thread_idx)`. Grows the region on demand. Returns
    // nullptr if the backend has no host buffer type or allocation fails —
    // caller falls back to a pageable std::vector.
    uint8_t * get_staging(size_t backend_idx, size_t thread_idx, size_t needed);

    // Returns the index into `backends` that owns `tensor->buffer`, or -1.
    int find_backend_for_tensor(const ggml_tensor * tensor) const;


    std::vector<ggml_backend_t> backends;
    const char * cache_dir;
    std::mutex buffers_mu;                                 // guards `buffers` (pool threads never mutate it; kept for future-proofing)
    std::unordered_set<ggml_backend_buffer_t> buffers;
    // store the last computed graph for each backend
    std::vector<stored_graph> stored_graphs;
    // staging_buffers[backend_idx][thread_idx]. thread_idx 0..pool_size-1 are
    // pool workers; the last slot is the main-thread fallback.
    std::vector<std::vector<staging_buffer>> staging_buffers;

    // Serialises ggml_backend_tensor_set across pool threads. Single-stream
    // tset runs at ~5 ms/tensor; with N pool threads contending on the CUDA
    // context concurrently the same call climbs to ~125–160 ms/tensor.
    // Holding this mutex across the HtoD keeps the context single-tenant
    // without giving up pool read parallelism. The dedicated-uploader
    // variant (Refactor F attempt, not shipped) dropped per-tset further to
    // ~52 ms but regressed setup cost by 3× — net loss.
    std::mutex tset_mu;

    // Diagnostic probe — aggregated across all threads on this connection.
    // Destructor prints the totals. Atomic because pool workers update
    // concurrently.
    struct stff_probe {
        std::atomic<uint64_t> count   {0};
        std::atomic<uint64_t> bytes   {0};
        std::atomic<uint64_t> ns_setup{0};
        std::atomic<uint64_t> ns_read {0};
        std::atomic<uint64_t> ns_tset {0};
    } stff;
};

void rpc_server::hello(rpc_msg_hello_rsp & response) {
    response.major = RPC_PROTO_MAJOR_VERSION;
    response.minor = RPC_PROTO_MINOR_VERSION;
    response.patch = RPC_PROTO_PATCH_VERSION;
    LOG_DBG("[%s] version: %d.%d.%d\n", __func__, response.major, response.minor, response.patch);
}

bool rpc_server::get_alloc_size(const rpc_msg_get_alloc_size_req & request, rpc_msg_get_alloc_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft;
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead()*(1 + GGML_MAX_SRC),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };

    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server get_alloc_size function.\n");
        return false;
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (request.srcs[i].id != 0) {
            tensor->src[i] = deserialize_tensor(ctx, &request.srcs[i]);
        }
    }

    LOG_DBG("[%s] device: %d, buffer: %p, data: %p\n", __func__, dev_id, (void*)tensor->buffer, tensor->data);
    if (tensor->buffer == nullptr) {
        //No buffer allocated.
        buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    } else {
        buft = tensor->buffer->buft;
    }

    response.alloc_size = ggml_backend_buft_get_alloc_size(buft, tensor);

    return true;
}

bool rpc_server::alloc_buffer(const rpc_msg_alloc_buffer_req & request, rpc_msg_alloc_buffer_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    ggml_backend_buffer_t buffer = ggml_backend_buft_alloc_buffer(buft, request.size);
    response.remote_ptr = 0;
    response.remote_size = 0;
    if (buffer != nullptr) {
        response.remote_ptr = reinterpret_cast<uint64_t>(buffer);
        response.remote_size = buffer->size;
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> remote_ptr: %" PRIx64 ", remote_size: %" PRIu64 "\n",
            __func__, dev_id, request.size, response.remote_ptr, response.remote_size);
        buffers.insert(buffer);
    } else {
        LOG_DBG("[%s] device: %d, size: %" PRIu64 " -> failed\n", __func__, dev_id, request.size);
    }
    return true;
}

bool rpc_server::get_alignment(const rpc_msg_get_alignment_req & request, rpc_msg_get_alignment_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t alignment = ggml_backend_buft_get_alignment(buft);
    LOG_DBG("[%s] device: %d, alignment: %lu\n", __func__, dev_id, alignment);
    response.alignment = alignment;
    return true;
}

bool rpc_server::get_max_size(const rpc_msg_get_max_size_req & request, rpc_msg_get_max_size_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    ggml_backend_buffer_type_t buft = ggml_backend_get_default_buffer_type(backends[dev_id]);
    size_t max_size = ggml_backend_buft_get_max_size(buft);
    LOG_DBG("[%s] device: %d, max_size: %lu\n", __func__, dev_id, max_size);
    response.max_size = max_size;
    return true;
}

bool rpc_server::buffer_get_base(const rpc_msg_buffer_get_base_req & request, rpc_msg_buffer_get_base_rsp & response) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    void * base = ggml_backend_buffer_get_base(buffer);
    response.base_ptr = reinterpret_cast<uint64_t>(base);
    return true;
}

bool rpc_server::free_buffer(const rpc_msg_free_buffer_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 "\n", __func__, request.remote_ptr);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_free(buffer);
    buffers.erase(buffer);
    return true;
}

bool rpc_server::buffer_clear(const rpc_msg_buffer_clear_req & request) {
    LOG_DBG("[%s] remote_ptr: %" PRIx64 ", value: %u\n", __func__, request.remote_ptr, request.value);
    ggml_backend_buffer_t buffer = reinterpret_cast<ggml_backend_buffer_t>(request.remote_ptr);
    if (buffers.find(buffer) == buffers.end()) {
        GGML_LOG_ERROR("[%s] buffer not found\n", __func__);
        return false;
    }
    ggml_backend_buffer_clear(buffer, request.value);
    return true;
}

ggml_tensor * rpc_server::deserialize_tensor(struct ggml_context * ctx, const rpc_tensor * tensor) {
    // Validate tensor type before using it
    if (tensor->type >= GGML_TYPE_COUNT) {
        GGML_LOG_ERROR("[%s] invalid tensor type received: %u\n", __func__, tensor->type);
        return nullptr;
    }

    // Fix: Prevent division by zero if blck_size is 0 (e.g., deprecated types)
    if (ggml_blck_size((enum ggml_type)tensor->type) == 0) {
        GGML_LOG_ERROR("[%s] invalid tensor type received (blck_size is 0): %u\n", __func__, tensor->type);
        return nullptr;
    }

    ggml_tensor * result = ggml_new_tensor_4d(ctx, (ggml_type) tensor->type,
        tensor->ne[0], tensor->ne[1], tensor->ne[2], tensor->ne[3]);

    // ggml_new_tensor_4d might fail if dimensions are invalid, although less likely to crash than invalid type
    if (result == nullptr) {
        GGML_LOG_ERROR("[%s] ggml_new_tensor_4d failed for type %u\n", __func__, tensor->type);
        return nullptr;
    }

    for (uint32_t i = 0; i < GGML_MAX_DIMS; i++) {
        result->nb[i] = tensor->nb[i];
    }
    result->buffer = reinterpret_cast<ggml_backend_buffer_t>(tensor->buffer);
    if (result->buffer && buffers.find(result->buffer) == buffers.end()) {
        result->buffer = nullptr;
    }

    if (result->buffer) {
        // require that the tensor data does not go beyond the buffer end
        uint64_t tensor_size = (uint64_t) ggml_nbytes(result);
        uint64_t buffer_start = (uint64_t) ggml_backend_buffer_get_base(result->buffer);
        uint64_t buffer_size = (uint64_t) ggml_backend_buffer_get_size(result->buffer);
        GGML_ASSERT(tensor->data + tensor_size >= tensor->data); // check for overflow
        GGML_ASSERT(tensor->data >= buffer_start && tensor->data + tensor_size <= buffer_start + buffer_size);
    }

    result->op = (ggml_op) tensor->op;
    for (uint32_t i = 0; i < GGML_MAX_OP_PARAMS / sizeof(int32_t); i++) {
        result->op_params[i] = tensor->op_params[i];
    }
    result->flags = tensor->flags;
    result->data = reinterpret_cast<void *>(tensor->data);
    ggml_set_name(result, tensor->name);
    return result;
}


bool rpc_server::set_tensor(const std::vector<uint8_t> & input) {
    // serialization format: | rpc_tensor | offset (8 bytes) | data (size bytes) |
    if (input.size() < sizeof(rpc_tensor) + sizeof(uint64_t)) {
        return false;
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *)input.data();
    uint64_t offset;
    memcpy(&offset, input.data() + sizeof(rpc_tensor), sizeof(offset));
    const size_t size = input.size() - sizeof(rpc_tensor) - sizeof(offset);

    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu\n", __func__, (void*)tensor->buffer, tensor->data, offset, size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (in_tensor->data + offset < p0 || in_tensor->data + offset >= p1 || size > (p1 - in_tensor->data - offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu) out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, in_tensor->data, offset, size, p0, p1);
            return false;
        }
    }

    const void * data = input.data() + sizeof(rpc_tensor) + sizeof(offset);
    if (cache_dir && size > HASH_THRESHOLD) {
        uint64_t hash = fnv_hash((const uint8_t*)data, size);
        char hash_str[17];
        snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
        // save to cache_dir/hash_str
        fs::path cache_file = fs::path(cache_dir) / hash_str;
        std::ofstream ofs(cache_file, std::ios::binary);
        ofs.write((const char *)data, size);
        GGML_LOG_INFO("[%s] saved to '%s'\n", __func__, cache_file.c_str());
    }
    ggml_backend_tensor_set(tensor, data, offset, size);
    return true;
}

// Worker-side handler for SET_TENSOR_FROM_FILE.
// Parses request, opens the file, reads bytes, writes them into the tensor's
// destination buffer. Returns true on success and sets response.result accordingly.
// On any failure (path validation, open, read, bounds), response.result=0 — the
// client will fall back to RPC_CMD_SET_TENSOR for that tensor.
int rpc_server::find_backend_for_tensor(const ggml_tensor * tensor) const {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return -1;
    }
    ggml_backend_dev_t want = tensor->buffer->buft->device;
    for (size_t i = 0; i < backends.size(); ++i) {
        if (ggml_backend_get_device(backends[i]) == want) {
            return (int) i;
        }
    }
    return -1;
}

uint8_t * rpc_server::get_staging(size_t backend_idx, size_t thread_idx, size_t needed) {
    if (backend_idx >= staging_buffers.size()) {
        return nullptr;
    }
    auto & row = staging_buffers[backend_idx];
    if (thread_idx >= row.size()) {
        return nullptr;
    }
    staging_buffer & sb = row[thread_idx];
    if (sb.capacity >= needed && sb.ptr != nullptr) {
        return sb.ptr;
    }
    // Need to (re)allocate. Free any existing region first.
    if (sb.buf != nullptr) {
        ggml_backend_buffer_free(sb.buf);
        sb.buf = nullptr;
        sb.ptr = nullptr;
        sb.capacity = 0;
    }
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[backend_idx]);
    if (dev == nullptr) {
        return nullptr;
    }
    ggml_backend_buffer_type_t host_buft = ggml_backend_dev_host_buffer_type(dev);
    if (host_buft == nullptr) {
        // Backend doesn't expose a host buffer type (e.g. pure-CPU build) —
        // caller will fall back to std::vector, which is fine there.
        return nullptr;
    }
    ggml_backend_buffer_t buf = ggml_backend_buft_alloc_buffer(host_buft, needed);
    if (buf == nullptr) {
        GGML_LOG_DEBUG("[%s] failed to allocate %zu byte pinned staging buffer for backend %zu thread %zu\n",
                       __func__, needed, backend_idx, thread_idx);
        return nullptr;
    }
    sb.buf      = buf;
    sb.ptr      = (uint8_t *) ggml_backend_buffer_get_base(buf);
    sb.capacity = needed;
    return sb.ptr;
}

bool rpc_server::set_tensor_from_file(const std::vector<uint8_t> & input,
                                      rpc_msg_set_tensor_from_file_rsp & response,
                                      size_t thread_idx) {
    const auto probe_t0 = std::chrono::steady_clock::now();
    response.result     = 0;
    response.bytes_read = 0;

    // Layout: | rpc_tensor | tensor_offset (8) | file_offset (8) | size (8) | path_len (4) | path[path_len] |
    constexpr size_t fixed_size = sizeof(rpc_tensor) + 3 * sizeof(uint64_t) + sizeof(uint32_t);
    if (input.size() < fixed_size) {
        GGML_LOG_ERROR("[%s] request too short (%zu)\n", __func__, input.size());
        return true; // ok=true means we sent a response; result=0 signals failure
    }
    const rpc_tensor * in_tensor = (const rpc_tensor *) input.data();
    size_t off = sizeof(rpc_tensor);
    uint64_t tensor_offset = 0, file_offset = 0, size = 0;
    uint32_t path_len = 0;
    std::memcpy(&tensor_offset, input.data() + off, sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(&file_offset,   input.data() + off, sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(&size,          input.data() + off, sizeof(uint64_t)); off += sizeof(uint64_t);
    std::memcpy(&path_len,      input.data() + off, sizeof(uint32_t)); off += sizeof(uint32_t);
    if (input.size() < fixed_size + path_len) {
        GGML_LOG_ERROR("[%s] truncated path (have %zu, need %zu)\n",
                       __func__, input.size(), fixed_size + path_len);
        return true;
    }
    std::string path((const char *) input.data() + off, path_len);

    // Path safety: must be absolute, no embedded NUL, no parent traversal.
    if (path.empty() || path[0] != '/' || path.find('\0') != std::string::npos
        || path.find("/../") != std::string::npos
        || path.size() > 4096) {
        GGML_LOG_ERROR("[%s] rejected path '%s'\n", __func__, path.c_str());
        return true;
    }

    // Deserialize the destination tensor (mirrors set_tensor).
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, in_tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return true;
    }

    // Bounds check — same form as set_tensor at line ~1153.
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);
        if (in_tensor->data + tensor_offset < p0
         || in_tensor->data + tensor_offset >= p1
         || size > (p1 - in_tensor->data - tensor_offset)) {
            GGML_LOG_ERROR("[%s] tensor data region out of buffer bounds\n", __func__);
            return true;
        }
    }

    // Read into a pinned staging buffer (via the backend's host buffer type)
    // so the subsequent HtoD inside ggml_backend_tensor_set runs as a genuine
    // pinned DMA (~10–25 GiB/s) rather than a pageable staging copy (~0.8
    // GiB/s). The pageable std::vector path remains the fallback for any
    // backend that doesn't expose a host buffer type, or if pinned alloc
    // fails (e.g. out of wired memory). See _enhancements/findings.md.
    //
    // N.B. mmap fast-path was tried here — it caused a 3× regression because
    // cudaMemcpyAsync on unpinned mmap'd memory demand-faults pages one-at-a-
    // time. Pinning from a pread'd buffer is the right shape.
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs.is_open()) {
        GGML_LOG_ERROR("[%s] cannot open '%s'\n", __func__, path.c_str());
        return true; // signal fallback
    }
    ifs.seekg((std::streamoff) file_offset, std::ios::beg);
    if (!ifs.good()) {
        GGML_LOG_ERROR("[%s] seek to %" PRIu64 " in '%s' failed\n",
                       __func__, file_offset, path.c_str());
        return true;
    }

    uint8_t * dst = nullptr;
    std::vector<uint8_t> fallback_buf;
    const int backend_idx = find_backend_for_tensor(tensor);
    if (backend_idx >= 0) {
        dst = get_staging((size_t) backend_idx, thread_idx, size);
    }
    if (dst == nullptr) {
        // Fall back to pageable host memory. Correct but slower; path runs
        // for CPU-only builds or if pinned alloc failed.
        fallback_buf.resize(size);
        dst = fallback_buf.data();
    }

    const auto probe_t1 = std::chrono::steady_clock::now();

    ifs.read((char *) dst, (std::streamsize) size);
    const std::streamsize got = ifs.gcount();
    if ((uint64_t) got != size) {
        GGML_LOG_ERROR("[%s] short read at %" PRIu64 " in '%s': got %lld of %" PRIu64 "\n",
                       __func__, file_offset, path.c_str(),
                       (long long) got, size);
        return true;
    }

    const auto probe_t2 = std::chrono::steady_clock::now();

    // HtoD, serialised across pool threads. See tset_mu comment for rationale.
    {
        std::lock_guard<std::mutex> lk(tset_mu);
        ggml_backend_tensor_set(tensor, dst, tensor_offset, size);
    }

    const auto probe_t3 = std::chrono::steady_clock::now();

    const uint64_t ns_setup = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(probe_t1 - probe_t0).count();
    const uint64_t ns_read  = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(probe_t2 - probe_t1).count();
    const uint64_t ns_tset  = (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(probe_t3 - probe_t2).count();
    stff.count.fetch_add(1,           std::memory_order_relaxed);
    stff.bytes.fetch_add(size,        std::memory_order_relaxed);
    stff.ns_setup.fetch_add(ns_setup, std::memory_order_relaxed);
    stff.ns_read.fetch_add(ns_read,   std::memory_order_relaxed);
    stff.ns_tset.fetch_add(ns_tset,   std::memory_order_relaxed);

    response.result     = 1;
    response.bytes_read = size;
    return true;
}

bool rpc_server::get_cached_file(uint64_t hash, std::vector<uint8_t> & data) {
    if (!cache_dir) {
        return false;
    }
    char hash_str[17];
    snprintf(hash_str, sizeof(hash_str), "%016" PRIx64, hash);
    fs::path cache_file = fs::path(cache_dir) / hash_str;
    std::error_code ec;
    if (!fs::exists(cache_file, ec)) {
        return false;
    }
    std::ifstream ifs(cache_file, std::ios::binary);
    ifs.seekg(0, std::ios::end);
    size_t size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);
    data.resize(size);
    ifs.read((char *)data.data(), size);
    return true;
}

bool rpc_server::set_tensor_hash(const rpc_msg_set_tensor_hash_req & request, rpc_msg_set_tensor_hash_rsp & response)
{
    std::vector<uint8_t> cached_file;
    if (!get_cached_file(request.hash, cached_file)) {
        response.result = 0;
        return true;
    }
    size_t size = cached_file.size();
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %zu, hash: %" PRIx64 "\n",
            __func__, (void*)tensor->buffer, tensor->data, request.offset, size, request.hash);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0
         || request.tensor.data + request.offset >= p1
         || size > (p1 - request.tensor.data - request.offset)) {
            GGML_LOG_ERROR("[%s] tensor data region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%zu, hash=0x%" PRIx64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                           __func__, request.tensor.data, request.offset, size, request.hash, p0, p1);
            return false;
        }
    }
    ggml_backend_tensor_set(tensor, cached_file.data(), request.offset, size);
    response.result = 1;
    return true;
}

bool rpc_server::init_tensor(const rpc_msg_init_tensor_req & request) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr) {
        GGML_LOG_ERROR("Null tensor pointer passed to server init_tensor function.\n");
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p\n", __func__, (void*)tensor->buffer, tensor->data);
    // Call the backend's buffer_init_tensor function
    ggml_backend_buffer_t buffer = tensor->buffer;
    if (buffer && buffer->iface.init_tensor) {
        buffer->iface.init_tensor(buffer, tensor);
    } else {
        if (!buffer) {
            GGML_LOG_ERROR("Tensor with null buffer passed to init_tensor function\n");
        }
    }

    if (tensor->extra != nullptr) {
        // This pointer can either be passed around client/server, or probably better stored server-side and kept track of.
        // Currently unimplemented.
        GGML_LOG_ERROR("tensor->extra populated by the backend, this is currently unsupported.\n");
        return false;
    }

    return true;
}

bool rpc_server::get_tensor(const rpc_msg_get_tensor_req & request, std::vector<uint8_t> & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    ggml_tensor * tensor = deserialize_tensor(ctx, &request.tensor);
    if (tensor == nullptr || tensor->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensor\n", __func__);
        return false;
    }
    LOG_DBG("[%s] buffer: %p, data: %p, offset: %" PRIu64 ", size: %" PRIu64 "\n", __func__, (void*)tensor->buffer, tensor->data, request.offset, request.size);

    // sanitize tensor->data
    {
        const size_t p0 = (size_t) ggml_backend_buffer_get_base(tensor->buffer);
        const size_t p1 = p0 + ggml_backend_buffer_get_size(tensor->buffer);

        if (request.tensor.data + request.offset < p0 ||
            request.tensor.data + request.offset >= p1 ||
            request.size > (p1 - request.tensor.data - request.offset)) {
                GGML_LOG_ERROR("[%s] requested tensor region (data=0x%" PRIx64 ", offset=%" PRIu64 ", size=%" PRIu64 ") out of buffer bounds [0x%zx, 0x%zx)\n",
                               __func__, request.tensor.data, request.offset, request.size, p0, p1);
                return false;
        }
    }

    response.resize(request.size, 0);
    ggml_backend_tensor_get(tensor, response.data(), request.offset, request.size);
    return true;
}

bool rpc_server::copy_tensor(const rpc_msg_copy_tensor_req & request, rpc_msg_copy_tensor_rsp & response) {
    struct ggml_init_params params {
        /*.mem_size   =*/ 2*ggml_tensor_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();

    ggml_tensor * src = deserialize_tensor(ctx, &request.src);
    ggml_tensor * dst = deserialize_tensor(ctx, &request.dst);
    if (src == nullptr || dst == nullptr || src->buffer == nullptr || dst->buffer == nullptr) {
        GGML_LOG_ERROR("[%s] error deserializing tensors\n", __func__);
        return false;
    }

    uint64_t src_size   = (uint64_t) ggml_nbytes(src);
    uint64_t dst_data   = (uint64_t) dst->data;
    uint64_t dst_base   = (uint64_t) ggml_backend_buffer_get_base(dst->buffer);
    uint64_t dst_buf_sz = (uint64_t) ggml_backend_buffer_get_size(dst->buffer);

    if (dst_data + src_size > dst_base + dst_buf_sz) {
        GGML_LOG_ERROR("[%s] out-of-bounds write in rpc_server::copy_tensor:\n"
                         "    write range : [0x%" PRIx64 ", 0x%" PRIx64 "]\n"
                         "    buffer base: [0x%" PRIx64 ", 0x%" PRIx64 "]\n",
                         __func__,
                         dst_data,
                         dst_data + src_size,
                         dst_base,
                         dst_base + dst_buf_sz);
        return false;
    }

    LOG_DBG("[%s] src->buffer: %p, dst->buffer: %p\n",
            __func__, (void*) src->buffer, (void*) dst->buffer);

    response.result = ggml_backend_buffer_copy_tensor(src, dst);
    return true;
}

ggml_tensor * rpc_server::create_node(uint64_t id,
                                      struct ggml_context * ctx,
                                      const std::unordered_map<uint64_t, const rpc_tensor*> & tensor_ptrs,
                                      std::unordered_map<uint64_t, struct ggml_tensor*> & tensor_map) {
    if (tensor_map.find(id) != tensor_map.end()) {
        return tensor_map[id];
    }
    // Safely find the tensor pointer
    auto it_ptr = tensor_ptrs.find(id);
    if (it_ptr == tensor_ptrs.end()) {
        return nullptr;
    }
    const rpc_tensor * tensor = it_ptr->second;

    struct ggml_tensor * result = deserialize_tensor(ctx, tensor);
    if (result == nullptr) {
        return nullptr;
    }
    if (result->buffer == nullptr && result->data != nullptr) {
        GGML_LOG_ERROR("[%s] invalid data ptr", __func__);
        return nullptr;
    }
    tensor_map[id] = result;
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        // Check if the source ID is 0 before calling create_node recursively
        if (tensor->src[i] == 0) {
            result->src[i] = nullptr;
        } else {
            result->src[i] = create_node(tensor->src[i], ctx, tensor_ptrs, tensor_map);
            // If the recursive call failed for a non-zero ID, propagate the error
            if (result->src[i] == nullptr) {
                GGML_LOG_ERROR("[%s] failed to create source node %d (src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                               __func__, i, tensor->src[i], id);
                // Must return nullptr to signal failure up the call stack
                return nullptr;
            }
        }
    }

    // Handle view_src similarly
    if (tensor->view_src == 0) {
        result->view_src = nullptr;
    } else {
        result->view_src = create_node(tensor->view_src, ctx, tensor_ptrs, tensor_map);
        // If the recursive call failed for a non-zero ID, propagate the error
        if (result->view_src == nullptr) {
            GGML_LOG_ERROR("[%s] failed to create view_src node (view_src_id=%" PRIu64 ") for node id %" PRIu64 "\n",
                           __func__, tensor->view_src, id);
            // Must return nullptr to signal failure up the call stack
            return nullptr;
        }
    }
    result->view_offs = tensor->view_offs;
    return result;
}

bool rpc_server::graph_compute(const std::vector<uint8_t> & input) {
    // serialization format:
    // | device (4 bytes) | n_nodes (4 bytes) | nodes (n_nodes * sizeof(uint64_t) | n_tensors (4 bytes) | tensors (n_tensors * sizeof(rpc_tensor)) |
    if (input.size() < 2*sizeof(uint32_t)) {
        return false;
    }
    const uint8_t * src = input.data();
    uint32_t device;
    memcpy(&device, src, sizeof(device));
    src += sizeof(device);
    if (device >= backends.size()) {
        return false;
    }
    uint32_t n_nodes;
    memcpy(&n_nodes, src, sizeof(n_nodes));
    src += sizeof(n_nodes);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t)) {
        return false;
    }
    const uint64_t * nodes = (const uint64_t *)src;
    src += n_nodes*sizeof(uint64_t);
    uint32_t n_tensors;
    memcpy(&n_tensors, src, sizeof(n_tensors));
    src += sizeof(n_tensors);
    if (input.size() < 2*sizeof(uint32_t) + n_nodes*sizeof(uint64_t) + sizeof(uint32_t) + n_tensors*sizeof(rpc_tensor)) {
        return false;
    }
    const rpc_tensor * tensors = (const rpc_tensor *)src;
    LOG_DBG("[%s] device: %u, n_nodes: %u, n_tensors: %u\n", __func__, device, n_nodes, n_tensors);

    size_t buf_size = ggml_tensor_overhead()*(n_nodes + n_tensors) + ggml_graph_overhead_custom(n_nodes, false);
    if (stored_graphs[device].buffer.size() < buf_size) {
        stored_graphs[device].buffer.resize(buf_size);
    }
    struct ggml_init_params params = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ stored_graphs[device].buffer.data(),
        /*.no_alloc   =*/ true,
    };
    ggml_context_ptr ctx_ptr { ggml_init(params) };
    GGML_ASSERT(ctx_ptr != nullptr);
    ggml_context * ctx = ctx_ptr.get();
    struct ggml_cgraph * graph = ggml_new_graph_custom(ctx, n_nodes, false);
    graph->n_nodes = n_nodes;
    std::unordered_map<uint64_t, const rpc_tensor*> tensor_ptrs;
    tensor_ptrs.reserve(n_tensors);
    for (uint32_t i = 0; i < n_tensors; i++) {
        tensor_ptrs.emplace(tensors[i].id, &tensors[i]);
    }
    std::unordered_map<uint64_t, ggml_tensor*> tensor_map;
    tensor_map.reserve(n_nodes);
    for (uint32_t i = 0; i < n_nodes; i++) {
        int64_t id;
        memcpy(&id, &nodes[i], sizeof(id));
        graph->nodes[i] = create_node(id, ctx, tensor_ptrs, tensor_map);

        // Check if create_node failed for a *non-zero* ID.
        // If id was 0, create_node returning nullptr is expected.
        // If id was non-zero and create_node returned nullptr, it indicates a deserialization error.
        if (graph->nodes[i] == nullptr && id != 0) {
            GGML_LOG_ERROR("[%s] failed to create graph node %d (id=%" PRId64 ")\n", __func__, i, id);
            return false;
        }
    }
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    stored_graphs[device].graph = graph;
    return true;
}

bool rpc_server::graph_recompute(const rpc_msg_graph_recompute_req & request) {
    uint32_t device = request.device;
    if (device >= backends.size()) {
        return false;
    }
    if (stored_graphs[device].graph == nullptr) {
        return false;
    }
    ggml_cgraph * graph = stored_graphs[device].graph;
    LOG_DBG("[%s] device: %u\n", __func__, device);
    ggml_status status = ggml_backend_graph_compute(backends[device], graph);
    GGML_ASSERT(status == GGML_STATUS_SUCCESS && "Unsuccessful graph computations are not supported with RPC");
    return true;
}

bool rpc_server::get_device_memory(const rpc_msg_get_device_memory_req & request, rpc_msg_get_device_memory_rsp & response) {
    uint32_t dev_id = request.device;
    if (dev_id >= backends.size()) {
        return false;
    }
    size_t free, total;
    ggml_backend_dev_t dev = ggml_backend_get_device(backends[dev_id]);
    ggml_backend_dev_memory(dev, &free, &total);
    response.free_mem = free;
    response.total_mem = total;
    LOG_DBG("[%s] device: %u, free_mem: %" PRIu64 ", total_mem: %" PRIu64 "\n", __func__, dev_id, response.free_mem, response.total_mem);
    return true;
}

rpc_server::~rpc_server() {
    const uint64_t count    = stff.count.load();
    if (count > 0) {
        const uint64_t bytes    = stff.bytes.load();
        const uint64_t ns_setup = stff.ns_setup.load();
        const uint64_t ns_read  = stff.ns_read.load();
        const uint64_t ns_tset  = stff.ns_tset.load();
        const double total_ms = (ns_setup + ns_read + ns_tset) / 1e6;
        const double gib = (double) bytes / (1024.0 * 1024.0 * 1024.0);
        // Sum of per-call times; wall time is lower when pool threads overlap.
        GGML_LOG_INFO("[stff_probe_summary] tensors=%" PRIu64 " bytes=%.2f GiB pool=%zu\n",
                      count, gib, pool_size());
        GGML_LOG_INFO("[stff_probe_summary]   setup  = %.3f s  (avg %.3f ms/tensor)\n",
                      ns_setup / 1e9, (ns_setup / 1e6) / count);
        GGML_LOG_INFO("[stff_probe_summary]   read   = %.3f s  (avg %.3f ms/tensor, %.2f GiB/s summed)\n",
                      ns_read / 1e9, (ns_read / 1e6) / count,
                      gib / (ns_read / 1e9));
        GGML_LOG_INFO("[stff_probe_summary]   tset   = %.3f s  (avg %.3f ms/tensor, %.2f GiB/s summed)\n",
                      ns_tset / 1e9, (ns_tset / 1e6) / count,
                      gib / (ns_tset / 1e9));
        GGML_LOG_INFO("[stff_probe_summary]   total  = %.3f s  (avg %.3f ms/tensor)\n",
                      total_ms / 1000.0, total_ms / count);
    }
    for (auto buffer : buffers) {
        ggml_backend_buffer_free(buffer);
    }
    for (auto & row : staging_buffers) {
        for (auto & sb : row) {
            if (sb.buf != nullptr) {
                ggml_backend_buffer_free(sb.buf);
            }
        }
    }
}

// --- SET_TENSOR_FROM_FILE worker pool (Refactor D — QD>1 pread) --------------
//
// The probe in Refactor C's diagnostic section showed 83% of the worker-side
// load wall sits in `ifs.read()` at ~1.3 GiB/s — single-stream NVMe QD=1.
// Pool lets N `pread`s be in flight concurrently so the drive sees QD=N,
// lifting aggregate read throughput toward its spec (~3+ GiB/s at QD=8).
//
// Invariants:
//   - Acks are emitted in request-submission order. The head's response
//     receiver (rpc_load_pipeline) drains FIFO per endpoint — see
//     ggml_backend_rpc_buffer_set_tensor_from_file_async.
//   - Order-preservation is achieved on the *main* thread: it keeps a FIFO
//     queue of in-flight tasks and sends acks from the front as each task
//     reports `done`. Workers may complete out of order; acks do not.
//   - Each worker uses a thread-indexed staging slot in `rpc_server::get_staging`
//     so per-connection pinned buffers don't contend.
//   - Pool is owned by a single connection; destroyed when the connection
//     ends, which drains any in-flight tasks first.

struct stff_task {
    std::vector<uint8_t> input;                      // owned copy of request payload
    rpc_msg_set_tensor_from_file_rsp response = {};
    bool ok_from_handler = true;                     // reflects handler's return value
    std::atomic<bool> done{false};
    std::mutex mu;
    std::condition_variable cv;

    void mark_done() {
        {
            std::lock_guard<std::mutex> lk(mu);
            done.store(true, std::memory_order_release);
        }
        cv.notify_all();
    }

    void wait_done() {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&]{ return done.load(std::memory_order_acquire); });
    }
};

class stff_pool {
public:
    stff_pool(rpc_server * srv, size_t n_threads) : srv_(srv), n_(n_threads) {
        workers_.reserve(n_threads);
        for (size_t i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this, i]() { worker_loop(i); });
        }
    }
    ~stff_pool() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        for (auto & t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    void submit(stff_task * t) {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push(t);
        }
        cv_.notify_one();
    }

    size_t size() const { return n_; }

private:
    void worker_loop(size_t thread_idx) {
        while (true) {
            stff_task * t = nullptr;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [&]{ return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) {
                    return;
                }
                t = queue_.front();
                queue_.pop();
            }
            // Execute the full read + (mutex-serialised) tset in one call.
            t->ok_from_handler = srv_->set_tensor_from_file(t->input, t->response, thread_idx);
            t->mark_done();
        }
    }

    rpc_server *             srv_;
    size_t                   n_;
    std::vector<std::thread> workers_;
    std::queue<stff_task *>  queue_;
    std::mutex               mu_;
    std::condition_variable  cv_;
    bool                     stop_ = false;
};

// Dedicated ack-sender for SET_TENSOR_FROM_FILE. The main thread pushes
// completed-or-in-flight tasks into a FIFO; this thread peeks the front,
// waits for that task to finish, sends the ack, then pops. Decoupling ack
// progress from the main recv loop avoids the deadlock where the head
// stops dispatching new STFF requests (waiting for acks) while the
// worker's main thread is blocked in recv_data() waiting for a command
// that never comes — leaving the last batch of acks undrained.
class stff_ack_sender {
public:
    stff_ack_sender(socket_ptr sock, size_t max_inflight)
        : sock_(std::move(sock)), max_inflight_(max_inflight) {
        sender_ = std::thread([this]() { sender_loop(); });
    }
    ~stff_ack_sender() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_push_.notify_all();
        if (sender_.joinable()) {
            sender_.join();
        }
    }

    // Enqueue a task whose worker submission follows immediately. Blocks
    // if the FIFO is full (bounds memory: each task holds its request
    // payload). Returns false if a prior ack send failed.
    bool push(std::unique_ptr<stff_task> t) {
        std::unique_lock<std::mutex> lk(mu_);
        cv_pop_.wait(lk, [&]{ return failed_ || q_.size() < max_inflight_; });
        if (failed_) {
            return false;
        }
        q_.push(std::move(t));
        cv_push_.notify_one();
        return true;
    }

    // Block until every queued ack has been sent. Returns false if any
    // send failed along the way.
    bool wait_empty() {
        std::unique_lock<std::mutex> lk(mu_);
        cv_pop_.wait(lk, [&]{ return failed_ || q_.empty(); });
        return !failed_;
    }

private:
    void sender_loop() {
        while (true) {
            stff_task * front_raw = nullptr;
            bool suppress_send = false;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_push_.wait(lk, [&]{ return stop_ || !q_.empty(); });
                if (q_.empty()) {
                    return;
                }
                front_raw = q_.front().get();
                // Once we've hit a send failure, keep draining wait_done()
                // so the pool destructor can safely join — but stop sending.
                suppress_send = failed_;
            }
            // Wait for the task outside the lock so pushes/backpressure
            // can progress while a slow worker finishes.
            front_raw->wait_done();
            bool ok = true;
            if (!suppress_send) {
                ok = front_raw->ok_from_handler &&
                     send_msg(sock_, &front_raw->response, sizeof(front_raw->response));
            }
            {
                std::lock_guard<std::mutex> lk(mu_);
                q_.pop();
                if (!ok) {
                    failed_ = true;
                }
            }
            cv_pop_.notify_all();
            // Loop even after failure: remaining tasks still need
            // wait_done() before the pool destructor joins workers.
        }
    }

    socket_ptr                             sock_;
    size_t                                 max_inflight_;
    std::queue<std::unique_ptr<stff_task>> q_;
    std::mutex                             mu_;
    std::condition_variable                cv_push_;
    std::condition_variable                cv_pop_;
    std::thread                            sender_;
    bool                                   stop_   = false;
    bool                                   failed_ = false;
};

static void rpc_serve_client(const std::vector<ggml_backend_t> & backends, const char * cache_dir,
                             socket_ptr sock) {
    rpc_server server(backends, cache_dir);
    uint8_t cmd;
    if (!sock->recv_data(&cmd, 1)) {
        return;
    }
    if (cmd != RPC_CMD_HELLO) {
        GGML_LOG_ERROR("Expected HELLO command, update client\n");
        return;
    }

    // Read input_size and validate protocol version
    uint64_t hello_input_size;
    if (!sock->recv_data(&hello_input_size, sizeof(hello_input_size))) {
        return;
    }

    if (hello_input_size != sizeof(rpc_msg_hello_req)) {
        GGML_LOG_ERROR("HELLO request size mismatch (%zu vs %zu) — client needs upgrade to protocol v%d.x\n",
                       (size_t)hello_input_size, sizeof(rpc_msg_hello_req), RPC_PROTO_MAJOR_VERSION);
        return;
    }

    rpc_msg_hello_req req = {};
    if (!sock->recv_data(&req, sizeof(req))) {
        return;
    }

    rpc_msg_hello_rsp rsp = {};
    server.hello(rsp);
    // Advertise server transport capabilities based on client's caps
    sock->get_caps(rsp.conn_caps);
    if (!send_msg(sock, &rsp, sizeof(rsp))) {
        return;
    }

    // Activate transport upgrade using client's caps
    sock->update_caps(req.conn_caps);

    // Set up the SET_TENSOR_FROM_FILE worker pool for QD>1 pread on the
    // rpc-server side. See the stff_pool comment block above for invariants.
    // Declaration order matters for teardown: the ack sender is destroyed
    // first (joining its thread, which calls wait_done() on pending tasks),
    // which requires the worker pool to still be alive.
    const size_t pool_n = rpc_server::pool_size();
    std::unique_ptr<stff_pool>       stff_workers;
    std::unique_ptr<stff_ack_sender> stff_acks;
    if (pool_n > 0) {
        stff_workers.reset(new stff_pool(&server, pool_n));
        stff_acks.reset(new stff_ack_sender(sock, pool_n * 2));
    }

    while (true) {
        if (!sock->recv_data(&cmd, 1)) {
            break;
        }
        if (cmd >= RPC_CMD_COUNT) {
            // fail fast if the command is invalid
            GGML_LOG_ERROR("Unknown command: %d\n", cmd);
            break;
        }
        // For any command other than SET_TENSOR_FROM_FILE, drain pending
        // acks first so subsequent operations see a consistent state. The
        // head can reorder across tensors but not across command types.
        // wait_empty() also serializes with the ack sender so the main
        // thread's send_msg calls below don't race with it on the socket.
        if (cmd != RPC_CMD_SET_TENSOR_FROM_FILE && stff_acks) {
            if (!stff_acks->wait_empty()) {
                return;
            }
        }
        switch (cmd) {
            case RPC_CMD_HELLO: {
                // HELLO command is handled above
                return;
            }
            case RPC_CMD_DEVICE_COUNT: {
                if (!recv_msg(sock, nullptr, 0)) {
                    return;
                }
                rpc_msg_device_count_rsp response;
                response.device_count = backends.size();
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_ALLOC_BUFFER: {
                rpc_msg_alloc_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_alloc_buffer_rsp response;
                if (!server.alloc_buffer(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALLOC_SIZE: {
                rpc_msg_get_alloc_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alloc_size_rsp response;
                if (!server.get_alloc_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_ALIGNMENT: {
                rpc_msg_get_alignment_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_alignment_rsp response;
                if (!server.get_alignment(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_MAX_SIZE: {
                rpc_msg_get_max_size_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_max_size_rsp response;
                if (!server.get_max_size(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_GET_BASE: {
                rpc_msg_buffer_get_base_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_buffer_get_base_rsp response;
                if (!server.buffer_get_base(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_FREE_BUFFER: {
                rpc_msg_free_buffer_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.free_buffer(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_BUFFER_CLEAR: {
                rpc_msg_buffer_clear_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.buffer_clear(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.set_tensor(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_HASH: {
                rpc_msg_set_tensor_hash_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_set_tensor_hash_rsp response;
                if (!server.set_tensor_hash(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_SET_TENSOR_FROM_FILE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (stff_workers) {
                    auto t = std::unique_ptr<stff_task>(new stff_task());
                    t->input = std::move(input);
                    stff_task * raw = t.get();
                    if (!stff_acks->push(std::move(t))) {
                        return;
                    }
                    stff_workers->submit(raw);
                } else {
                    // Pool disabled — serial fallback, main-thread slot.
                    rpc_msg_set_tensor_from_file_rsp response = {};
                    if (!server.set_tensor_from_file(input, response, /*thread_idx=*/ pool_n)) {
                        return;
                    }
                    if (!send_msg(sock, &response, sizeof(response))) {
                        return;
                    }
                }
                break;
            }
            case RPC_CMD_INIT_TENSOR: {
                rpc_msg_init_tensor_req request;
                if (!recv_msg(sock, &request,sizeof(request))) {
                    return;
                }
                if (!server.init_tensor(request)) {
                    return;
                }
                if (!send_msg(sock, nullptr, 0)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_TENSOR: {
                rpc_msg_get_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                std::vector<uint8_t> response;
                if (!server.get_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, response.data(), response.size())) {
                    return;
                }
                break;
            }
            case RPC_CMD_COPY_TENSOR: {
                rpc_msg_copy_tensor_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_copy_tensor_rsp response;
                if (!server.copy_tensor(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_COMPUTE: {
                std::vector<uint8_t> input;
                if (!recv_msg(sock, input)) {
                    return;
                }
                if (!server.graph_compute(input)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GRAPH_RECOMPUTE: {
                rpc_msg_graph_recompute_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                if (!server.graph_recompute(request)) {
                    return;
                }
                break;
            }
            case RPC_CMD_GET_DEVICE_MEMORY: {
                rpc_msg_get_device_memory_req request;
                if (!recv_msg(sock, &request, sizeof(request))) {
                    return;
                }
                rpc_msg_get_device_memory_rsp response;
                if (!server.get_device_memory(request, response)) {
                    return;
                }
                if (!send_msg(sock, &response, sizeof(response))) {
                    return;
                }
                break;
            }
            default: {
                GGML_LOG_ERROR("Unknown command: %d\n", cmd);
                return;
            }
        }
    }
}

void ggml_backend_rpc_start_server(const char * endpoint, const char * cache_dir,
                                   size_t n_threads, size_t n_devices, ggml_backend_dev_t * devices) {
    if (n_devices == 0 || devices == nullptr) {
        fprintf(stderr, "Invalid arguments to ggml_backend_rpc_start_server\n");
        return;
    }
    std::vector<ggml_backend_t> backends;
    printf("Starting RPC server v%d.%d.%d\n",
        RPC_PROTO_MAJOR_VERSION,
        RPC_PROTO_MINOR_VERSION,
        RPC_PROTO_PATCH_VERSION);
    printf("  endpoint       : %s\n", endpoint);
    printf("  local cache    : %s\n", cache_dir ? cache_dir : "n/a");
    printf("Devices:\n");
    for (size_t i = 0; i < n_devices; i++) {
        auto dev = devices[i];
        size_t free, total;
        ggml_backend_dev_memory(dev, &free, &total);
        printf("  %s: %s (%zu MiB, %zu MiB free)\n", ggml_backend_dev_name(dev), ggml_backend_dev_description(dev),
               total / 1024 / 1024, free / 1024 / 1024);
        auto backend = ggml_backend_dev_init(dev, nullptr);
        if (!backend) {
            fprintf(stderr, "Failed to create backend for device %s\n", dev->iface.get_name(dev));
            return;
        }
        backends.push_back(backend);
        ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
        if (reg) {
            auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
            if (ggml_backend_set_n_threads_fn) {
                ggml_backend_set_n_threads_fn(backend, n_threads);
            }
        }
    }

    std::string host;
    int port;
    if (!parse_endpoint(endpoint, host, port)) {
        return;
    }

#ifdef GGML_RPC_RDMA
    printf("  transport      : TCP (RDMA auto-negotiate enabled)\n");
#else
    printf("  transport      : TCP\n");
#endif // GGML_RPC_RDMA
    if (!rpc_transport_init()) {
        fprintf(stderr, "Failed to initialize RPC transport\n");
        return;
    }
    auto server_socket = socket_t::create_server(host.c_str(), port);
    if (server_socket == nullptr) {
        fprintf(stderr, "Failed to create server socket\n");
        return;
    }
    while (true) {
        auto client_socket = server_socket->accept();
        if (client_socket == nullptr) {
            fprintf(stderr, "Failed to accept client connection\n");
            return;
        }
        printf("Accepted client connection\n");
        fflush(stdout);
        rpc_serve_client(backends, cache_dir, client_socket);
        printf("Client connection closed\n");
        fflush(stdout);
    }
    rpc_transport_shutdown();
    for (auto backend : backends) {
        ggml_backend_free(backend);
    }
}

// device interface

struct ggml_backend_rpc_device_context {
    std::string endpoint;
    uint32_t    device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_rpc_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->name.c_str();
}

static const char * ggml_backend_rpc_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ctx->description.c_str();
}

static void ggml_backend_rpc_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    ggml_backend_rpc_get_device_memory(ctx->endpoint.c_str(), ctx->device, free, total);
}

static enum ggml_backend_dev_type ggml_backend_rpc_device_get_type(ggml_backend_dev_t dev) {
    // TODO: obtain value from the server
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_rpc_device_get_props(ggml_backend_dev_t dev, struct ggml_backend_dev_props * props) {
    props->name        = ggml_backend_rpc_device_get_name(dev);
    props->description = ggml_backend_rpc_device_get_description(dev);
    props->type        = ggml_backend_rpc_device_get_type(dev);
    ggml_backend_rpc_device_get_memory(dev, &props->memory_free, &props->memory_total);
    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_rpc_device_init(ggml_backend_dev_t dev, const char * params) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_init(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_rpc_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_rpc_device_context * ctx = (ggml_backend_rpc_device_context *)dev->context;

    return ggml_backend_rpc_buffer_type(ctx->endpoint.c_str(), ctx->device);

    GGML_UNUSED(dev);
}

static bool ggml_backend_rpc_device_supports_op(ggml_backend_dev_t dev, const struct ggml_tensor * op) {
    GGML_UNUSED(dev);
    GGML_UNUSED(op);
    //TODO: call the remote backend and cache the results
    return true;
}

static bool ggml_backend_rpc_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    if (!buft || buft->iface.get_name != ggml_backend_rpc_buffer_type_name) {
        return false;
    }
    ggml_backend_rpc_buffer_type_context * buft_ctx = (ggml_backend_rpc_buffer_type_context *)buft->context;
    ggml_backend_rpc_device_context * dev_ctx = (ggml_backend_rpc_device_context *)dev->context;
    return buft_ctx->endpoint == dev_ctx->endpoint && buft_ctx->device == dev_ctx->device;
}

static const struct ggml_backend_device_i ggml_backend_rpc_device_i = {
    /* .get_name             = */ ggml_backend_rpc_device_get_name,
    /* .get_description      = */ ggml_backend_rpc_device_get_description,
    /* .get_memory           = */ ggml_backend_rpc_device_get_memory,
    /* .get_type             = */ ggml_backend_rpc_device_get_type,
    /* .get_props            = */ ggml_backend_rpc_device_get_props,
    /* .init_backend         = */ ggml_backend_rpc_device_init,
    /* .get_buffer_type      = */ ggml_backend_rpc_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_rpc_device_supports_op,
    /* .supports_buft        = */ ggml_backend_rpc_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

// backend reg interface

struct ggml_backend_rpc_reg_context {
    std::string                     name;
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_rpc_reg_get_name(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->name.c_str() : "RPC";
}

static size_t ggml_backend_rpc_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    return ctx ? ctx->devices.size() : 0;
}

static ggml_backend_dev_t ggml_backend_rpc_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_rpc_reg_context * ctx = (ggml_backend_rpc_reg_context *)reg->context;
    if (ctx == nullptr) {
        GGML_ABORT("The RPC backend does not have enumerated devices - use ggml_backend_rpc_add_server instead");
    } else {
        GGML_ASSERT(index < ctx->devices.size());
        return ctx->devices[index];
    }
}

// Reset all per-endpoint load stats and start the wall-clock timer.
// Called by the loader before load_all_data.
extern "C" void ggml_backend_rpc_load_stats_reset(void) {
    std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
    g_rpc_stats.clear();
    g_rpc_stats_t_start_us = ggml_time_us();
}

// Log a per-endpoint summary of what set_tensor / init_tensor cost during load,
// plus the loader's overall wall time. Called by the loader after load_all_data
// when LLAMA_RPC_LOAD_PROFILE is set.
extern "C" void ggml_backend_rpc_log_load_stats(void) {
    std::lock_guard<std::mutex> lock(g_rpc_stats_mutex);
    const int64_t t_wall_us = g_rpc_stats_t_start_us > 0
        ? ggml_time_us() - g_rpc_stats_t_start_us
        : 0;
    if (g_rpc_stats.empty()) {
        if (t_wall_us > 0) {
            GGML_LOG_INFO("RPC load profile: no RPC tensor activity (wall %.3f s)\n",
                          (double) t_wall_us / 1.0e6);
        }
        return;
    }
    GGML_LOG_INFO("RPC load profile (per-worker):\n");
    GGML_LOG_INFO("  %-24s %8s %8s %8s %8s %10s %10s %10s %10s %9s %9s %9s\n",
                  "endpoint", "n_set", "n_hash", "n_hit", "n_wread",
                  "data_GB", "sent_GB", "skip_GB", "wread_GB",
                  "t_set_s", "t_hash_s", "t_wread_s");
    rpc_load_stats totals;
    for (const auto & kv : g_rpc_stats) {
        const auto & ep = kv.first;
        const auto & s  = kv.second;
        GGML_LOG_INFO("  %-24s %8" PRIu64 " %8" PRIu64 " %8" PRIu64 " %8" PRIu64
                      " %10.3f %10.3f %10.3f %10.3f %9.3f %9.3f %9.3f\n",
                      ep.c_str(),
                      s.n_set_tensor, s.n_hash_check, s.n_hash_hit, s.n_worker_read,
                      (double) s.bytes_data        / (1024.0 * 1024.0 * 1024.0),
                      (double) s.bytes_sent        / (1024.0 * 1024.0 * 1024.0),
                      (double) s.bytes_skipped     / (1024.0 * 1024.0 * 1024.0),
                      (double) s.bytes_worker_read / (1024.0 * 1024.0 * 1024.0),
                      (double) s.t_set_us         / 1.0e6,
                      (double) s.t_hash_us        / 1.0e6,
                      (double) s.t_worker_read_us / 1.0e6);
        totals.n_set_tensor       += s.n_set_tensor;
        totals.n_hash_check       += s.n_hash_check;
        totals.n_hash_hit         += s.n_hash_hit;
        totals.n_worker_read      += s.n_worker_read;
        totals.n_worker_read_miss += s.n_worker_read_miss;
        totals.bytes_data         += s.bytes_data;
        totals.bytes_sent         += s.bytes_sent;
        totals.bytes_skipped      += s.bytes_skipped;
        totals.bytes_worker_read  += s.bytes_worker_read;
        totals.t_set_us           += s.t_set_us;
        totals.t_hash_us          += s.t_hash_us;
        totals.t_worker_read_us   += s.t_worker_read_us;
    }
    GGML_LOG_INFO("  %-24s %8" PRIu64 " %8" PRIu64 " %8" PRIu64 " %8" PRIu64
                  " %10.3f %10.3f %10.3f %10.3f %9.3f %9.3f %9.3f\n",
                  "TOTAL",
                  totals.n_set_tensor, totals.n_hash_check, totals.n_hash_hit, totals.n_worker_read,
                  (double) totals.bytes_data        / (1024.0 * 1024.0 * 1024.0),
                  (double) totals.bytes_sent        / (1024.0 * 1024.0 * 1024.0),
                  (double) totals.bytes_skipped     / (1024.0 * 1024.0 * 1024.0),
                  (double) totals.bytes_worker_read / (1024.0 * 1024.0 * 1024.0),
                  (double) totals.t_set_us         / 1.0e6,
                  (double) totals.t_hash_us        / 1.0e6,
                  (double) totals.t_worker_read_us / 1.0e6);
    if (totals.n_worker_read_miss > 0) {
        GGML_LOG_INFO("  worker-read fallbacks: %" PRIu64 " (worker couldn't open/read)\n",
                      totals.n_worker_read_miss);
    }
    if (totals.t_set_us > 0) {
        const double mb_per_s = ((double) totals.bytes_sent / (1024.0 * 1024.0)) /
                                ((double) totals.t_set_us  / 1.0e6);
        GGML_LOG_INFO("  aggregate set_tensor throughput: %.1f MB/s (sum-of-sends, not wall)\n", mb_per_s);
    }
    if (t_wall_us > 0) {
        GGML_LOG_INFO("  loader wall time: %.3f s; effective wall throughput: %.1f MB/s\n",
                      (double) t_wall_us / 1.0e6,
                      ((double) totals.bytes_sent / (1024.0 * 1024.0)) /
                       ((double) t_wall_us / 1.0e6));
    }
}

static void * ggml_backend_rpc_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (std::strcmp(name, "ggml_backend_rpc_add_server") == 0) {
        return (void *)ggml_backend_rpc_add_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_start_server") == 0) {
        return (void *)ggml_backend_rpc_start_server;
    }
    if (std::strcmp(name, "ggml_backend_rpc_load_stats_reset") == 0) {
        return (void *)ggml_backend_rpc_load_stats_reset;
    }
    if (std::strcmp(name, "ggml_backend_rpc_log_load_stats") == 0) {
        return (void *)ggml_backend_rpc_log_load_stats;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_set_tensor_from_file") == 0) {
        return (void *)ggml_backend_rpc_buffer_set_tensor_from_file;
    }
    if (std::strcmp(name, "ggml_backend_rpc_buffer_set_tensor_from_file_async") == 0) {
        return (void *)ggml_backend_rpc_buffer_set_tensor_from_file_async;
    }
    if (std::strcmp(name, "ggml_backend_rpc_flush_pending_reads") == 0) {
        return (void *)ggml_backend_rpc_flush_pending_reads;
    }
    return NULL;

    GGML_UNUSED(reg);
}

static const struct ggml_backend_reg_i ggml_backend_rpc_reg_i = {
    /* .get_name         = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device       = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_reg(void) {
    static struct ggml_backend_reg ggml_backend_rpc_reg = {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_i,
        /* .context     = */ NULL,
    };

    return &ggml_backend_rpc_reg;
}

static uint32_t ggml_backend_rpc_get_device_count(const char * endpoint) {
    auto sock = get_socket(endpoint);
    if (sock == nullptr) {
        GGML_LOG_ERROR("Failed to connect to %s\n", endpoint);
        return 0;
    }
    rpc_msg_device_count_rsp response;
    bool status = send_rpc_cmd(sock, RPC_CMD_DEVICE_COUNT, nullptr, 0, &response, sizeof(response));
    RPC_STATUS_ASSERT(status);
    return response.device_count;
}

static const ggml_backend_reg_i ggml_backend_rpc_reg_interface = {
    /* .get_name          = */ ggml_backend_rpc_reg_get_name,
    /* .get_device_count  = */ ggml_backend_rpc_reg_get_device_count,
    /* .get_device        = */ ggml_backend_rpc_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_rpc_get_proc_address,
};

ggml_backend_reg_t ggml_backend_rpc_add_server(const char * endpoint) {
    static std::unordered_map<std::string, ggml_backend_reg_t> reg_map;
    static std::mutex mutex;
    static uint32_t dev_id = 0;
    std::lock_guard<std::mutex> lock(mutex);
    if (reg_map.find(endpoint) != reg_map.end()) {
        return reg_map[endpoint];
    }
    uint32_t dev_count = ggml_backend_rpc_get_device_count(endpoint);
    if (dev_count == 0) {
        return nullptr;
    }
    ggml_backend_rpc_reg_context * ctx = new ggml_backend_rpc_reg_context;
    ctx->name = "RPC[" + std::string(endpoint) + "]";
    for (uint32_t ind = 0; ind < dev_count; ind++) {
        std::string dev_name = "RPC" + std::to_string(dev_id);
        std::string dev_desc = std::string(endpoint);
        ggml_backend_rpc_device_context * dev_ctx = new ggml_backend_rpc_device_context {
            /* .endpoint    = */ endpoint,
            /* .device      = */ ind,
            /* .name        = */ dev_name,
            /* .description = */ dev_desc
        };

        ggml_backend_dev_t dev = new ggml_backend_device {
            /* .iface   = */ ggml_backend_rpc_device_i,
            /* .reg     = */ ggml_backend_rpc_reg(),
            /* .context = */ dev_ctx,
        };
        ctx->devices.push_back(dev);
        dev_id++;
    }
    ggml_backend_reg_t reg = new ggml_backend_reg {
        /* .api_version = */ GGML_BACKEND_API_VERSION,
        /* .iface       = */ ggml_backend_rpc_reg_interface,
        /* .context     = */ ctx
    };
    reg_map[endpoint] = reg;
    return reg;
}


GGML_BACKEND_DL_IMPL(ggml_backend_rpc_reg)
