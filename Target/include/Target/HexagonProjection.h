// Copyright (c) 2026, Modular Inc. All rights reserved.
// Licensed under the MAX License: https://modular.com/legal/max-license
#pragma once
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
// Experimental single-session projector ABI, independent of the MAX HAL ABI.
typedef void *M_hexagon_projection_t;
// Upload centered group-128 affine weights once. Packed codes are [K/16,N],
// scales/bias are [K/128,N,2]. Errors return nonzero and fill the message buffer.
int M_hexagon_projection_create(uint32_t width, uint32_t reduction,
                                const uint32_t *packed, const float *affine,
                                M_hexagon_projection_t *out, char *error,
                                size_t error_size);
// Execute one FP16 activation vector and return N FP32 results. Weights remain
// resident; input/output transfers are limited to activations/results.
int M_hexagon_projection_execute(M_hexagon_projection_t handle,
                                 const uint16_t *x, float *y, char *error,
                                 size_t error_size);
void M_hexagon_projection_destroy(M_hexagon_projection_t handle);
#ifdef __cplusplus
}
#endif
