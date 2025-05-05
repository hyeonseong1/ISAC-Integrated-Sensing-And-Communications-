import torch
import numpy as np
import matplotlib.pyplot as plt


def per_sample_autocorr(c: torch.Tensor, N: int) -> torch.Tensor:
    """
    배치 c: [B, 2*N] 의 각 샘플별 ACSL 값을 계산해 반환합니다 (shape [B]).
    """
    re, im = c[:, :N], c[:, N:]
    c_complex = re + 1j * im  # [B, N]
    c_norm = c_complex / torch.linalg.norm(
        c_complex, axis=1, keepdims=True)  # [B, N]
    acsls = []
    for l in range(1, N):
        shifted = torch.roll(c_norm, shifts=l, dims=1)  # [B, N]
        corr = (c_norm.conj() * shifted).sum(dim=1)  # [B]
        acsls.append(torch.abs(corr) ** 2)  # [B]
    # [B, N-1] → 각 행(mean over lags) → [B]
    return torch.stack(acsls, dim=1).mean(dim=1)


def plot_fig2(enc16: torch.nn.Module,
              device: torch.device,
              full_enum: bool = False,
              sample_size: int = 10000):
    """
    K=16 인코더의 ACSL 분포를 그립니다.

    - full_enum=True: 2^16개 모든 메시지에 대해 그립니다 (메모리/속도 주의).
    - full_enum=False: 랜덤 sample_size 개 메시지에 대해 그립니다.
    """
    K = 16
    N = 2 * K
    enc16.eval()

    # 1) 메시지 생성
    if full_enum:
        B = 2 ** K
        # 전부 열거: shape [65536,16]
        m = torch.zeros(B, K, dtype=torch.float32, device=device)
        ar = torch.arange(B, device=device)
        for i in range(K):
            m[:, i] = ((ar >> i) & 1).float()
    else:
        B = sample_size
        m = torch.randint(0, 2, (B, K), dtype=torch.float32, device=device)

    # 2) 인코딩 & ACSL 계산
    with torch.no_grad():
        c = enc16(m)  # [B, 2*N]
        acsl_per = per_sample_autocorr(c, N).cpu().numpy()  # [B]

    # 3) dB 변환 및 플롯
    acsl_db = 10 * np.log10(acsl_per)
    idx = np.arange(len(acsl_db))

    plt.figure(figsize=(8, 4))
    plt.scatter(idx, acsl_db, s=1, alpha=0.5)
    median_db = np.median(acsl_db)
    plt.axhline(median_db, color='r', linestyle='--',
                label=f"median = {median_db:.2f} dB")

    plt.title("ACSL Distribution for K=16")
    plt.xlabel("Message index")
    plt.ylabel("ACSL (dB)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('figures/acsl_K=16.png')
    plt.show()
