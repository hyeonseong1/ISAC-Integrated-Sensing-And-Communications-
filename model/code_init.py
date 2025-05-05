import torch

def concatenated_code_initialization(enc32: torch.nn.Module,
                                     enc16: torch.nn.Module):
    """
    논문 Definition 2 (Concatenated Code)에 따라,
    Encoder(16, N//2)로부터 Encoder(32, N) 파라미터를
    [W16 0; 0 W16], [b16; b16] 형태로 초기화합니다.
    """
    with torch.no_grad():
        # named_parameters 순서가 동일하다고 가정
        for (_, p16), (_, p32) in zip(enc16.named_parameters(),
                                      enc32.named_parameters()):
            if p16.ndim == 2:
                # weight: [out16, in16] -> block_diag -> [2*out16, 2*in16]
                p32.copy_(torch.block_diag(p16, p16))
            else:
                # bias 혹은 batchnorm stats: [dim16] -> [2*dim16]
                p32.copy_(torch.cat([p16, p16], dim=0))