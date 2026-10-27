// Generated from SDK declarations. Do not edit.
#include "HAL/M_driver_api.h"

M_driver_status_t M_driver_context_set_current(M_driver_context_t context) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_context_get_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t *driver_context) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_context_set_current_driver_context(
    M_driver_context_t context, M_driver_native_handle_t driver_context) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_context_memory_info(M_driver_context_t context,
                                               uint64_t *free, uint64_t *total) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_copy_intra_device_sync(M_driver_context_t context,
                                                  M_driver_memory_view_t dst,
                                                  M_driver_memory_view_t src) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_copy_inter_device_sync(
    M_driver_context_t context, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_set_memory_sync(M_driver_context_t context,
                                           M_driver_memory_view_t dst,
                                           uint8_t value) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_fill_sync(M_driver_context_t context,
                                     M_driver_memory_view_t dst, uint64_t value,
                                     uint64_t value_size) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_device_attribute(M_driver_device_t device,
                                            int32_t attribute, int32_t *value) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_function_property(M_driver_context_t context,
                                             M_driver_function_t function,
                                             const char *property_name,
                                             void *value) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_function_set_attribute(M_driver_context_t context,
                                                  M_driver_function_t function,
                                                  int32_t attribute,
                                                  int32_t value) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_function_occupancy_max_active_blocks(
    M_driver_context_t context, M_driver_function_t function,
    int32_t block_size, uint64_t dynamic_shared_memory_bytes,
    int32_t *num_blocks) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_memory_wrap(M_driver_context_t context,
                                       uint64_t address, uint64_t size,
                                       bool owning, M_driver_memory_t *memory) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_memory_unwrap(M_driver_context_t context,
                                         M_driver_memory_t memory,
                                         uint64_t *address) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_memory_alloc_async(M_driver_queue_t queue,
                                              uint64_t size,
                                              M_driver_memory_t *memory) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_memory_free_async(M_driver_queue_t queue,
                                             M_driver_memory_t memory) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_device_can_access_peer(M_driver_device_t device,
                                                  M_driver_device_t peer,
                                                  bool *out_can_access) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t
M_driver_context_enable_peer_access(M_driver_context_t context,
                                    M_driver_context_t peer) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_queue_copy_inter_device(
    M_driver_queue_t queue, M_driver_memory_view_t dst,
    M_driver_memory_view_t src, M_driver_context_t src_context) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_queue_property(M_driver_queue_t queue,
                                          const char *property_name,
                                          void *value) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_queue_launch_host_func(M_driver_queue_t queue,
                                                  void (*fn)(void *),
                                                  void *user_data) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_queue_wait_value64(M_driver_queue_t queue,
                                              uint64_t device_address,
                                              uint64_t value) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_tensor_map_encode_tiled(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const uint32_t *box_dim,
    const uint32_t *element_strides, int32_t interleave, int32_t swizzle,
    int32_t l2_promotion, int32_t oob_fill) { return M_DRIVER_STATUS_UNIMPLEMENTED; }

M_driver_status_t M_driver_tensor_map_encode_im2col(
    M_driver_context_t context, void *tensor_map, int32_t data_type,
    uint32_t rank, uint64_t global_device_address, const uint64_t *global_dim,
    const uint64_t *global_strides, const int32_t *pixel_box_lower_corner,
    const int32_t *pixel_box_upper_corner, int32_t channels_per_pixel,
    int32_t pixels_per_column, const uint32_t *element_strides,
    int32_t interleave, int32_t swizzle, int32_t l2_promotion,
    int32_t oob_fill) { return M_DRIVER_STATUS_UNIMPLEMENTED; }
