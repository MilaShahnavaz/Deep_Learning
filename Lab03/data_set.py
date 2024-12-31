import os
import random
import tensorflow as tf
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io

def load_file_paths(input_dir, target_dir, verbose=True):
    input_img_paths = sorted(
        [
            os.path.join(input_dir, filename)
            for filename in os.listdir(input_dir)
            if filename.endswith(".jpg")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, filename)
            for filename in os.listdir(target_dir)
            if filename.endswith(".png") and not filename.startswith(".")
        ]
    )
    if verbose:
        print(f"Number of input images: {len(input_img_paths)}")
        print(f"Number of target masks: {len(target_img_paths)}")
        print("First 5 samples:")
        for input_path, target_path in zip(input_img_paths[:5], target_img_paths[:5]):
            print(input_path, "|", target_path)
    return input_img_paths, target_img_paths

def preprocess_image_mask(input_img_path, target_img_path, img_size=(160, 160)):
    """
    Preprocess a single image and its corresponding mask.
    """
    # Load and resize input image
    input_img = tf_io.read_file(input_img_path)
    input_img = tf_io.decode_png(input_img, channels=3)
    input_img = tf_image.resize(input_img, img_size)
    input_img = tf.cast(input_img, tf.float32) / 255.0  # Normalize to [0, 1]

    # Load and resize target mask
    target_img = tf_io.read_file(target_img_path)
    target_img = tf_io.decode_png(target_img, channels=1)
    target_img = tf_image.resize(target_img, img_size, method="nearest")
    target_img = tf.cast(target_img, tf.uint8)  # Keep masks as integers
    target_img -= 1  # Adjust classes to 0, 1, 2

    return input_img, target_img


def get_datasets(input_dir, target_dir, batch_size=16, val_split=0.2, img_size=(160, 160), verbose=True):
    input_img_paths, target_img_paths = load_file_paths(input_dir, target_dir, verbose)
    val_samples = int(len(input_img_paths) * val_split)
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = target_img_paths[-val_samples:]
    if verbose:
        print(f"Training samples: {len(train_input_img_paths)}")
        print(f"Validation samples: {len(val_input_img_paths)}")

    train_dataset = (
        tf_data.Dataset.from_tensor_slices((train_input_img_paths, train_target_img_paths))
        .map(lambda x, y: preprocess_image_mask(x, y, img_size), num_parallel_calls=tf_data.AUTOTUNE)
        .batch(batch_size)
    )

    val_dataset = (
        tf_data.Dataset.from_tensor_slices((val_input_img_paths, val_target_img_paths))
        .map(lambda x, y: preprocess_image_mask(x, y, img_size), num_parallel_calls=tf_data.AUTOTUNE)
        .batch(batch_size)
    )

    return train_dataset, val_dataset
