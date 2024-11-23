import os
os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import triton
import triton.language as tl

@triton.jit
def chunkwise_attn_fwd_kernel_h(k, v, h,
                                s_k_t, s_k_d, s_v_t, s_v_d, s_h_t,
                                T: tl.constexpr, D: tl.constexpr,
                                BT: tl.constexpr, BD: tl.constexpr):
    # k, v: (T, D)
    # h: (NT*D, D)

    i_k = tl.program_id(0)
    i_v = tl.program_id(1)

    b_h = tl.zeros([BD, BD], dtype=tl.float32)

    for i_t in range(triton.cdiv(T, BT)):
        p_k = tl.make_block_ptr(k, (D, T), (s_k_d, s_k_t), (i_k * BD, i_t * BT), (BD, BT), (0, 1))
        p_v = tl.make_block_ptr(v, (D, T), (s_v_d, s_v_t), (i_t * BT, i_v * BD), (BT, BD), (0, 1))
        p_h = tl.make_block_ptr(h, (D, D), (s_h_t, 1), (i_k * BD, i_v * BD), (BD, BD), (1, 0))

        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        b_k = tl.load(p_k, boundary_check=(0, 1)) # (BD, BT)
        b_v = tl.load(p_v, boundary_check=(0, 1)) # (BD, BT)

        b_h += tl.dot(b_k, b_v)

def chunkwise_attn_fwd_kernel_o(q, k, v, h, o, ):
    return

#B = 8
#H = 1
T = 256
D = 32

Q, K, V = torch.randn(T, D), torch.randn(T, D), torch.randn(T, D)

BT = 64
BD = 16

ND = triton.cdiv(D, BD)
NT = triton.cdiv(T, BT)

h = Q.new_empty(NT*D, D)
grid = (ND, ND)
chunkwise_attn_fwd_kernel_h[grid](K, V, h,
                                  K.stride(0), K.stride(1), V.stride(0), V.stride(1), h.stride(0),
                                  T, D,
                                  BT, BD)

print(h)