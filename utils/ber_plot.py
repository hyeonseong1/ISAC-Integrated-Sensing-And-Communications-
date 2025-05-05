import os
import matplotlib.pyplot as plt

from utils.ber import evaluate_ber


def plot_fig4(enc16, dec16, enc32, dec32, device,
              snr_list=None, num_batches=1000, batch_size=1000,
              save_path="figures/ber_curve.png"):
    """
    학습된 (enc16, dec16), (enc32, dec32) 에 대해 SNR = -5..8 dB 구간에서
    BER 곡선을 계산·플롯한 뒤 save_path에 저장하고 화면에 표시합니다.

    - snr_list: 리스트로 직접 전달하거나 기본값(range(-5,9))
    - num_batches, batch_size: BER 통계 샘플 수 조절
    """
    # 0) 디폴트 SNR 리스트 설정
    if snr_list is None:
        snr_list = list(range(-5, 9))  # [-5, -4, …, 7, 8]

    # 1) 각각의 K에 대해 BER 계산
    ber16 = evaluate_ber(enc=enc16, dec=dec16, device=device,
                         K=16, SNR_dBs=snr_list,
                         num_batches=num_batches, batch_size=batch_size)
    ber32 = evaluate_ber(enc=enc32, dec=dec32, device=device,
                         K=32, SNR_dBs=snr_list,
                         num_batches=num_batches, batch_size=batch_size)

    # 2) 디렉토리 생성
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 3) 한 그림에 K별 BER 곡선 그리기
    for K, ber_dict in [(16, ber16), (32, ber32)]:
        snrs = sorted(ber_dict.keys())
        bers = [ber_dict[s] for s in snrs]
        plt.semilogy(snrs, bers,
                     marker='o', linestyle='-',
                     label=f"K={K}")

    plt.xlabel("SNR (dB)")
    plt.ylabel("BER")
    plt.title("BER vs SNR for Multiple K")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.legend()
    plt.tight_layout()

    # 4) 파일 저장 & 출력
    plt.savefig(save_path)
    plt.show()
