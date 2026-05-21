import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import torch


def no_noise(density_map: np.ndarray) -> float:

    summ = density_map.sum()

    if summ <= 25:
        density_map = summ
    else:
        no_zero = density_map[density_map != 0]
        t1 = 500 / (summ ** 2)
        t2 = 100 * (no_zero.mean()/torch.median(no_zero) - 1.3) / summ
        threshold = no_zero.std() * (t1 + t2) if t2 > 0 else density_map[density_map != 0].std() * t1

        density_map = torch.where(density_map > threshold, density_map, torch.zeros_like(density_map))
        density_map = density_map.sum()

    return density_map


if __name__ == '__main__':
    density = np.load('D:/mygit/opd/FamNet/density_map_result.npy').squeeze()
    #density = np.load("D:/mygit/opd/FamNet/data/gt_density_map_adaptive_384_VarV2/295.npy")
    print(np.sum(density))
    #image = cv2.imread('D:/mygit/opd/FamNet/data/images_384_VarV2/2.jpg')
    t1 = 500 / (density[density != 0].sum() ** 2)
    t2 = 100 * (density[density != 0].mean()/np.median(density[density != 0]) - 1.3) / (density[density != 0].sum())
    print(t1, t2)

    threshold = density[density != 0].std() * ( t1 + t2 ) if t2 > 0 else density[density != 0].std() * (t1 + 2)
    print(density[density != 0].mean(), np.median(density[density != 0]), density[density != 0].mean()/np.median(density[density != 0]), density[density != 0].std(), threshold)
    density[density <= threshold] = 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 6))

    # Density map (левая ось)
    im = ax1.imshow(density, cmap='viridis')
    ax1.set_title(f"Sum = {density.sum():.2f}")
    ax1.axis('off')

    ax2.hist(density.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax2.axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold = {threshold:.3f}')
    ax2.set_xlabel('Значение')
    ax2.set_ylabel('Частота')
    ax2.set_title('Гистограмма плотности')
    ax2.legend()

    plt.tight_layout()
    plt.show()