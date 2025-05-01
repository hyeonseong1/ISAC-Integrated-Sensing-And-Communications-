import torch
import torch.optim as optim
import numpy as np
from model.EncDec import Encoder, Decoder
from model.loss_fn import custom_loss

def train(K, N, lambda_, device):
    # Hyperparameters (Table I)
    batch_size = 1000
    num_epochs = 400
    Nenc, Ndec = 10, 50

    enc = Encoder(K, N).to(device)
    dec = Decoder(K, N).to(device)
    enc_opt = optim.Adam(enc.parameters(), lr=1e-4)
    dec_opt = optim.Adam(dec.parameters(), lr=1e-4)

    # Scheduler (LR × 0.9 if plateau over 10 epochs, min LR = 1e-6)
    enc_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        enc_opt, mode='min', factor=0.9, patience=10, min_lr=1e-6
    )
    dec_sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        dec_opt, mode='min', factor=0.9, patience=10, min_lr=1e-6
    )

    # Initialization for K=32 using pre-trained 16-bit model
    if K == 32:
        tmp_enc = Encoder(16, N//2).to(device)
        tmp_enc.load_state_dict(torch.load("pre-trained/encoder16.pth"), strict=True)

        # Initialize 32-bit encoder with block-diagonal weights from 16-bit encoder
        with torch.no_grad():
            for (p16_name, p16), (p32_name, p32) in zip(tmp_enc.named_parameters(), enc.named_parameters()):
                if p16.ndim == 2:
                    blk = torch.block_diag(p16, p16)
                    p32.copy_(blk)
                else:
                    p32.copy_(torch.cat([p16, p16], dim=0))

    history = {'training': [], 'acsl': [], 'comms': []}

    for epoch in range(num_epochs):
        # SNR scheduling
        if K == 16:
            snr = 3 if epoch < 200 else 6
        else:
            snr = 3

        # --- Encoder update ---
        enc.train(); dec.eval()
        for _ in range(Nenc):
            m = torch.randint(0, 2, (batch_size, K)).float().to(device)
            loss, acsl, comms = custom_loss(enc, dec, m, lambda_, snr, N)
            enc_opt.zero_grad()
            loss.backward()
            enc_opt.step()

        # --- Decoder update ---
        enc.eval(); dec.train()
        for _ in range(Ndec):
            loss, acsl, comms = custom_loss(enc, dec, m, lambda_, snr, N)
            dec_opt.zero_grad()
            loss.backward()
            dec_opt.step()

        # Record losses
        history['training'].append(loss.item())
        history['acsl'].append(acsl.item())
        history['comms'].append(comms.item())

        enc_sched.step(loss.item())
        dec_sched.step(loss.item())

        if epoch % 10 == 0:
            # Convert to dB scale
            tot_db = 10 * np.log10(loss.item())
            acsl_db = 10 * np.log10(acsl.item())
            comms_db = 10 * np.log10(comms.item())
            print(f"Epoch {epoch} | training {tot_db:.2f} dB | acsl {acsl_db:.2f} dB | comms {comms_db:.2f} dB")

    return history, enc
