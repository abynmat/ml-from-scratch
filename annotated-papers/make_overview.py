"""Stitch each paper's headline figure into figures/overview.png.  Author: Abyn Scaria."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.abspath(__file__))
PANELS = [("01 Adam", "01_adam/figures/bias_correction_stress.png"),
          ("02 BatchNorm", "02_batchnorm/figures/bn_ablation.png"),
          ("03 ResNet", "03_resnet/figures/degradation.png"),
          ("04 VAE", "04_vae/figures/beta_tradeoff.png"),
          ("05 GAN", "05_gan/figures/gan_mode_coverage.png"),
          ("06 DDPM", "06_ddpm/figures/ddpm_ablation.png")]

fig, axes = plt.subplots(3, 2, figsize=(16, 12))
for ax, (title, path) in zip(axes.flat, PANELS):
    ax.imshow(mpimg.imread(os.path.join(ROOT, path)))
    ax.set_title(title, fontsize=14)
    ax.axis("off")
fig.tight_layout()
os.makedirs(os.path.join(ROOT, "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "figures", "overview.png"), dpi=150)
print("wrote figures/overview.png")
