"""Grad-CAM for a single image, saved as a PNG (no GUI needed).

    python -m src.pipeline.gradcam_pipeline --model resnet --image "path/to/xray.jpg" [--out out.png]

Uses the documented last-conv layer per model (model_builder.LAST_CONV_LAYER) and the
predicted-class map at the val-tuned threshold from artifacts/clinical/thresholds.json
(falls back to 0.5 if that file does not exist).
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="resnet", choices=["vgg", "resnet", "efficientnet"])
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    import keras
    import tensorflow as tf
    from src.components.data_loader import load_image
    from src.components.gradcam import gradcam_signed, overlay, to_heatmap
    from src.components.model_builder import LAST_CONV_LAYER
    from src.pipeline.train_pipeline import model_path
    from src.utils.config_loader import load_config

    config = load_config()
    img_size = tuple(config["training"]["img_size"])
    threshold = 0.5
    if os.path.exists("artifacts/clinical/thresholds.json"):
        threshold = json.load(open("artifacts/clinical/thresholds.json"))["models"][args.model]["threshold"]

    model = keras.models.load_model(model_path(config, args.model))
    layer = LAST_CONV_LAYER[args.model]
    img = load_image(tf.constant(args.image), img_size).numpy()
    cams, _, prob = gradcam_signed(img[None], model, [layer])
    is_tb = prob[0] >= threshold
    heatmap, empty = to_heatmap(cams[layer], 1.0 if is_tb else -1.0)

    out = args.out or f"gradcam_{args.model}_{os.path.splitext(os.path.basename(args.image))[0]}.png"
    fig, axes = plt.subplots(1, 2, figsize=(8, 4.3))
    axes[0].imshow(img.astype(np.uint8)); axes[0].set_title(os.path.basename(args.image), fontsize=9)
    axes[1].imshow(overlay(img, heatmap[0]))
    axes[1].set_title(f"{args.model} ({layer}): P(TB)={prob[0]:.3f} -> {'TB' if is_tb else 'Normal'} "
                      f"@ {threshold:.3g}" + (" [empty map]" if empty[0] else ""), fontsize=9)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
