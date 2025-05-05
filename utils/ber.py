import torch

def evaluate_ber(enc, dec, device, K, SNR_dBs, num_batches=100, batch_size=1000):
    """
    enc, dec: 학습된 Encoder/Decoder 모듈
    SNR_dBs  : BER 곡선을 그릴 SNR 리스트 (예: [0, 2, 4, 6, 8, 10])
    """
    enc.eval()
    dec.eval()
    total_bits = 0
    error_bits = {snr: 0 for snr in SNR_dBs}

    with torch.no_grad():
        for snr in SNR_dBs:
            sigma = 10 ** (-snr / 20)
            errors = 0
            bits = 0

            for _ in range(num_batches):
                # 1) 랜덤 정보 비트 생성
                m = torch.randint(0, 2, (batch_size, K)).float().to(device)
                # 2) 인코딩
                c = enc(m)
                # 3) AWGN 채널
                noise = sigma * torch.randn_like(c)
                c_noisy = c + noise
                # 4) 디코딩 (logit → bit 예측)
                logits = dec.net(c_noisy)  # [B, K*2]
                logits = logits.view(batch_size, K, 2)
                # argmax 로 0/1 예측
                m_hat = logits.argmax(dim=2).float()  # [B, K]

                # 5) 오류 개수 집계
                errors += (m_hat != m).sum().item()
                bits += m.numel()

            error_bits[snr] = errors
            total_bits = bits  # 마지막 batch_size*K*num_batches

    # 6) BER 계산
    ber = {snr: error_bits[snr] / total_bits for snr in SNR_dBs}
    return ber
