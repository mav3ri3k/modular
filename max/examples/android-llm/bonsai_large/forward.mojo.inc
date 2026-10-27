    status = launch_embedding(rt, @H@, w[0], w[1], b[1], token)
    if status != 0: return status
    status = rotate(rt, b[1], w[2], b, @H@, 1)
    if status != 0: return status
    status = launch_copy(rt, @H@, b[2], b[0])
    if status != 0: return status
    for layer in range(32):
        var off = 4 + layer * 25
        var history = b[19 + layer * 2]
        var state = b[20 + layer * 2]
        status = norm(rt, b[0], w[off], b[1], b[3])
        if status != 0: return status
        if (layer + 1) % 4 != 0:
            status = project(rt, w, b, off + 11, b[1], b[4], 8192, @H@)
            if status != 0: return status
            status = project(rt, w, b, off + 14, b[1], b[5], 4096, @H@, reuse_transform=True)
            if status != 0: return status
            status = launch_gate_partials(rt, 2048, b[1], w[off + 20], w[off + 21], b[2])
            if status != 0: return status
            status = launch_convolve(rt, 8192, b[4], history, w[off + 23], b[9])
            if status != 0: return status
            status = launch_normalize_qk(rt, 8192, b[9], b[4])
            if status != 0: return status
            status = launch_delta_gates(rt, 32, b[2], w[off + 22], b[8])
            if status != 0: return status
            status = launch_delta_update(rt, 4096, b[4], b[8], state, b[10])
            if status != 0: return status
            status = launch_gated_norm(rt, 4096, b[10], b[5], w[off + 24], b[1])
            if status != 0: return status
            status = project(rt, w, b, off + 17, b[1], b[1], @H@, 4096)
            if status != 0: return status
        else:
            status = project(rt, w, b, off + 11, b[1], b[5], 8192, @H@)
            if status != 0: return status
            status = project(rt, w, b, off + 14, b[1], b[15], 1024, @H@, reuse_transform=True)
            if status != 0: return status
            status = project(rt, w, b, off + 17, b[1], b[16], 1024, @H@, reuse_transform=True)
            if status != 0: return status
            status = launch_head_norm(rt, 4096, b[5], w[off + 23], b[14], 0)
            if status != 0: return status
            status = launch_head_norm(rt, 1024, b[15], w[off + 24], b[9], 1)
            if status != 0: return status
            status = launch_rope(rt, 512, b[14], b[18], 4096, position)
            if status != 0: return status
            status = launch_rope(rt, 128, b[9], b[18], 1024, position)
            if status != 0: return status
            status = launch_cache_kv(rt, 1024, b[9], b[16], history, state, position)
            if status != 0: return status
            status = launch_attention_scores(rt, 16 * (position + 1), b[14], history, b[13], position, capacity)
            if status != 0: return status
            status = launch_attention_values(rt, 4096, b[13], state, b[5], b[10], position, capacity)
            if status != 0: return status
            status = project(rt, w, b, off + 20, b[10], b[1], @H@, 4096)
            if status != 0: return status
        status = launch_residual(rt, @H@, b[0], b[1])
        if status != 0: return status
        status = norm(rt, b[0], w[off + 1], b[1], b[3])
        if status != 0: return status
        status = project(rt, w, b, off + 2, b[1], b[11], @FF@, @H@)
        if status != 0: return status
        status = project(rt, w, b, off + 5, b[1], b[12], @FF@, @H@, reuse_transform=True)
        if status != 0: return status
        status = launch_swiglu(rt, @FF@, b[11], b[12])
        if status != 0: return status
        status = project(rt, w, b, off + 8, b[11], b[1], @H@, @FF@)
        if status != 0: return status
        status = launch_residual(rt, @H@, b[0], b[1])
        if status != 0: return status
    status = norm(rt, b[0], w[3], b[1], b[3])
    if status != 0: return status
    status = project(rt, w, b, @HEAD@, b[1], b[17], 248320, @H@, head_gpu_columns=head_gpu_columns, head_input=head_input, gpu_greedy=gpu_greedy)
    if status != 0: return status
    return 0
