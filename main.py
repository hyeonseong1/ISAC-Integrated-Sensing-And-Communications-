import torch
import os

from model.train_model import train

from utils.loss_plot import plot_fig5
from utils.ber_plot import plot_fig4
from utils.acsl_plot import plot_fig2


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("pre-trained", exist_ok=True)
    lambda_ = 0.9

    # When lambda=0.9
    history16, enc16, dec16 = train(K=16, N=32, lambda_=lambda_, device=device)
    torch.save(enc16.state_dict(), "pre-trained/encoder16.pth")
    torch.save(dec16.state_dict(), "pre-trained/decoder16.pth")
    history32, enc32, dec32 = train(K=32, N=64, lambda_=lambda_, device=device)
    torch.save(enc32.state_dict(), "pre-trained/encoder32.pth")
    torch.save(dec16.state_dict(), "pre-trained/decoder32.pth")

    # Plot training curves
    plot_fig5(history16, history32, lambda_=lambda_)

    plot_fig4(enc16, dec16, enc32, dec32, device, num_batches=1000, batch_size=1000)

    plot_fig2(enc16, device, full_enum=False, sample_size=20000)