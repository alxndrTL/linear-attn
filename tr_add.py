import os

os.environ['TRITON_INTERPRET'] = '1' # needs to be set *before* triton is imported

import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, z_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)

    off = pid*BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = off < N
    
    x = tl.load(x_ptr+off, mask)
    y = tl.load(y_ptr+off, mask)
    z = x+y
    tl.store(z_ptr+off, z, mask)

def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output

torch.manual_seed(0)
size = 2046
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is 'f'{torch.max(torch.abs(output_torch - output_triton))}')