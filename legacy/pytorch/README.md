# Legacy PyTorch prototype: unused, not maintained

These three files were an early PyTorch version of the pipeline. Nothing in the project imports them, and as far as we can tell they were never run (docs/audit.md §5.7). They are kept here for reference only.

Known problems, which have **not** been fixed:
- `torch_dataset.py` validates on the full training folder, so there is no train/val split.
- No ImageNet normalisation, and no class weights.
- `torch_train.py` saves to `models/torch/`, which is never created, so it crashes after training.
- The imports still point at their old locations (`src.components.torch_*`), so the files won't run from here.
- `torch`, `torchvision` and `tqdm` are not in `requirements.txt`.

The maintained pipeline is TensorFlow/Keras: see the main README.
