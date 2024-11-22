import argparse
import time

import torch

from linear_attn import chunkwise_attn, recurrent_attn
from fla.ops.linear_attn.naive import naive_chunk_linear_attn as fla_chunkwise_attn

### ensure same API for all functions
ATTN_FUNC = {'chunkwise': chunkwise_attn,
             'recurrent': recurrent_attn,
             'fla_chunkwise': lambda Q,K,V: fla_chunkwise_attn(Q.unsqueeze(1), K.unsqueeze(1), V.unsqueeze(1), scale=1).squeeze(1)}

#TODO : bfloat16??? (checker d'abord si les calculs se font bien en bfloat16 en vrai)
#TODO : comment, à partir de ça, faire des graphiques où l'on fait varier une variable?
#TODO : comment prendre en compte le chunk_size? (un paramètre de chunkwise mais pas de recurrent)

device = "cpu"
requires_grad = False
WARMUP_STEPS = 10
MEASURE_STEPS = 10

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, required=True, help="chunkwise,recurent")
parser.add_argument('--B', type=int, required=False, default=1)
parser.add_argument('--L', type=int, required=False, default=1024)
parser.add_argument('--D', type=int, required=False, default=128)
parser.add_argument('--profile', type=bool, required=False, default=False)

args = parser.parse_args()

Q = torch.randn(args.B, args.L, args.D, requires_grad=requires_grad).to(device=device)
K = torch.randn(args.B, args.L, args.D, requires_grad=requires_grad).to(device=device)
V = torch.randn(args.B, args.L, args.D, requires_grad=requires_grad).to(device=device)

func = ATTN_FUNC[args.type]

for _ in range(WARMUP_STEPS):
    func(Q, K, V)

st = time.time_ns()
for _ in range(MEASURE_STEPS):
    func(Q, K, V)
et = time.time_ns()

t = (et - st) / MEASURE_STEPS / 1000000
print(f'{t}ms')

if args.profile:
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs/bench_log_flash'),
            record_shapes=True,
            profile_memory=True,
            with_stack=False, # incurs an additional overhead, disable if not needed
            with_flops=True,
            with_modules=False, # only for torchscript models atm
    ) as prof:
        func(Q, K, V)
    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=10))