import numpy as np
import matplotlib.pyplot as plt
import os

def plot_fig5(history16, history32, lambda_):
    """
    history*: dict with keys 'total','acsl','comms' containing lists of length num_epochs
    lambda_: float (e.g. 0.9)
    """
    epochs = np.arange(len(history16['training']))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Training (total) loss
    ax.plot(epochs, 10*np.log10(history16['training']), linestyle='-',  label='Training loss (K=16)')
    ax.plot(epochs, 10*np.log10(history32['training']), linestyle='--', label='Training loss (K=32)')

    # Communications loss
    ax.plot(epochs, 10*np.log10(history16['comms']), linestyle='-',  label='Comms loss (K=16)')
    ax.plot(epochs, 10*np.log10(history32['comms']), linestyle='--', label='Comms loss (K=32)')

    # ACSL
    ax.plot(epochs, 10*np.log10(history16['acsl']), linestyle='-',  label='ACSL (K=16)')
    ax.plot(epochs, 10*np.log10(history32['acsl']), linestyle='--', label='ACSL (K=32)')

    # SNR step vertical line
    # ax.axvline(200, color='k', linestyle='--', label='SNR step @ epoch 200')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (dB)')
    ax.set_title(f'λ={lambda_}')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    if lambda_ == 0.9:
        plt.savefig('figures/lambda 0.9 result.png')
    else:
        plt.savefig('figures/lambda 0 result.png')
    plt.show()