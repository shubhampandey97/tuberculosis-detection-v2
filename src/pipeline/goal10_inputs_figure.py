"""Goal 10 visual check: what the shortcut-removal pipeline does to 8 training images (4 Normal, 4 TB).

    $env:TB_CONFIG = "src/config/config_v10_arm1.yaml"; python -m src.pipeline.goal10_inputs_figure

Columns: (1) model input with the internal-crop box, (2) training augmentation as sampled (markers with
probability 0.5, label-independent), (3) the same augmentation with markers forced on, (4) what the backbone
receives for column 3 (central crop + per-image standardisation, the model's own layers).
Output: <clinical_dir>/../processed_inputs.png (Kaggle training images only; no NLM images).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from src.utils.config_loader import load_config  # noqa: E402


def main():
    import tensorflow as tf

    from src.components.data_loader import build_augmentation, load_image, read_split, shortcut_augment
    from src.components.model_builder import CentralCrop, PerImageStandardize, crop_pixels

    cfg = load_config()
    opts, params = cfg["model_options"], cfg["augmentation"]["shortcut"]
    size = tuple(cfg["training"]["img_size"])
    train = read_split(cfg, "train")
    picks = pd.concat([train[train.label == 0].sample(4, random_state=cfg["seed"]),
                       train[train.label == 1].sample(4, random_state=cfg["seed"])])
    rng = np.random.default_rng(cfg["seed"])
    geometric = build_augmentation(cfg)
    crop, standardize = CentralCrop(opts["crop_fraction"]), PerImageStandardize()
    c = crop_pixels(size[0], opts["crop_fraction"])

    fig, axes = plt.subplots(len(picks), 4, figsize=(10.5, 2.75 * len(picks)))
    for i, (_, r) in enumerate(picks.iterrows()):
        x = load_image(tf.constant(r.path), size).numpy()
        img = np.clip(np.round(x), 0, 255).astype(np.uint8)
        sampled, info_s = shortcut_augment(img, rng, params)
        forced, info_f = shortcut_augment(img, rng, params, force_markers=True)
        batch = np.stack([sampled, forced]).astype(np.float32)
        batch = geometric(batch, training=True).numpy()          # rotation / zoom, as in training
        seen = standardize(crop(batch[1:2])).numpy()[0]           # backbone input for column 3
        cells = [(x, "model input (box = internal crop)"),
                 (batch[0], f"augmented as sampled\nmarkers={info_s['markers']}, JPEG q={info_s['jpeg_quality']}"),
                 (batch[1], f"augmented, markers forced\nmarkers={info_f['markers']}, JPEG q={info_f['jpeg_quality']}"),
                 (seen, "backbone input (crop + standardise)\n"
                        f"mean {seen.mean():.0f}, sd {seen.std():.0f}, range {seen.min():.0f}-{seen.max():.0f}")]
        for j, (im, title) in enumerate(cells):
            ax = axes[i, j]
            ax.imshow(np.clip(im[..., 0], 0, 255), cmap="gray", vmin=0, vmax=255)
            ax.set_title(title, fontsize=7)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.add_patch(Rectangle((c, c), size[1] - 2 * c, size[0] - 2 * c, fill=False, ec="#eb6834", lw=1.2))
                ax.set_ylabel(f"{'TB' if r.label else 'Normal'}\n{os.path.basename(r.path)}", fontsize=8)
    fig.suptitle(f"Goal 10 inputs ({cfg['tag']}): crop {opts['crop_fraction']:.0%} per edge, per-image "
                 "standardisation, label-independent augmentation", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = os.path.join(os.path.dirname(cfg["artifacts"]["clinical_dir"]), "processed_inputs.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
