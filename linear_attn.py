import torch

CHUNK_SIZE = 128

def chunkwise_attn(Q, K, V, chunk_size=None):
    # Q, K, V : (B, L, d)

    B, L, d = Q.size()

    chunk_size = chunk_size or CHUNK_SIZE
    assert L%chunk_size==0
    n = L//chunk_size

    S_prev = torch.zeros(B, d, d)
    Q = Q.view(B, n, chunk_size, d)
    K = K.view(B, n, chunk_size, d)
    V = V.view(B, n, chunk_size, d)
    O = torch.zeros(B, n, chunk_size, d)
    
    for c in range(n):
        O[:, c] = Q[:, c] @ S_prev + torch.tril(Q[:, c] @ K[:, c].transpose(1, 2)) @ V[:, c]
        S_prev = S_prev + K[:, c].transpose(1, 2) @ V[:, c]

    O = O.view(B, L, d)
    return O

def recurrent_attn(Q, K, V):
    # Q, K, V : (B, L, d)

    B, L, d = Q.size()

    S = torch.zeros(B, d, d)
    O = torch.zeros(B, L, d)

    for t in range(L):
        S = S + K[:, [t]].transpose(1, 2) @ V[:, [t]]
        O[:, [t]] = Q[:, [t]] @ S
        
    return O
