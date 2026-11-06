import os

# Set to False if op determinism fails or slows training more than 2x (see docs/audit.md Status).
USE_OP_DETERMINISM = True


def set_global_seed(seed):
    """Seed Python, NumPy and TensorFlow (audit §5.5)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    import keras
    import tensorflow as tf

    keras.utils.set_random_seed(seed)  # python random, numpy, tf
    if USE_OP_DETERMINISM:
        tf.config.experimental.enable_op_determinism()
