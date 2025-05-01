import torch


def autocorr_loss(c, N):
    # c: [B, 2*N] -> interpreted as complex numbers
    re, im = c[:, :N], c[:, N:]
    c_complex = re + 1j*im                  # [B, N]
    # Normalize each codeword
    c_norm = c_complex / torch.linalg.norm(c_complex, axis=1, keepdims=True)
    B = c_norm.shape[0]
    # Calculate ACSL for lag=1 to N-1
    acsls = []
    for l in range(1, N):
        shifted = torch.roll(c_norm, shifts=l, dims=1)
        corr = (c_norm.conj() * shifted).sum(dim=1)  # [B]
        acsls.append(torch.abs(corr)**2)
    acsl = torch.stack(acsls, dim=1).mean(dim=1)    # [B], mean over lags
    return acsl.mean()

def custom_loss(encoder, decoder, m, lambda_, snr_db, N):
    # 1) Auto-correlation loss
    c = encoder(m)
    acsl = autocorr_loss(c, N)

    # 2) Communication loss (decoder)
    sigma = 10**(-snr_db/20)
    noise = sigma * torch.randn_like(c)
    comms_loss = decoder(c + noise, m)

    # 3) Combined loss
    return lambda_ * acsl + (1-lambda_) * comms_loss, acsl.detach(), comms_loss.detach()