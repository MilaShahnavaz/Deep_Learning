import os  # Required for file and directory operations
import keras
import tensorflow as tf
import numpy as np
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io


def load_img_and_mask(input_img_path, target_img_path, img_size):
    try:
        # Load and preprocess the input image
        input_img = tf.io.read_file(input_img_path)
        # Change to decode_jpeg for better JPEG handling
        # input_img = tf.io.decode_jpeg(input_img, channels=3)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf.image.resize(input_img, img_size)
        input_img = tf.image.convert_image_dtype(input_img, tf.float32)

        # Load and preprocess the target mask
        target_img = tf.io.read_file(target_img_path)
        target_img = tf.io.decode_png(target_img, channels=1)
        target_img = tf.image.resize(target_img, img_size, method="nearest")
        target_img = tf.image.convert_image_dtype(target_img, tf.uint8)
        target_img -= 1  # Adjust class labels to start at 0
        return input_img, target_img
    except Exception as e:
        print(f"Error loading file {input_img_path} or {target_img_path}: {e}")
        return None, None


def prepare_datasets(input_dir, target_dir, img_size, batch_size, val_samples=1000):
    # Get sorted paths for input images and target masks
    input_img_paths = sorted([os.path.join(input_dir, fname)
                             for fname in os.listdir(input_dir) if fname.endswith(".jpg")])
    target_img_paths = sorted(
        [os.path.join(target_dir, fname) for fname in os.listdir(
            target_dir) if fname.endswith(".png") and not fname.startswith("._")]
    )

    # Filter target paths to only include matching input basenames
    input_basenames = [os.path.splitext(os.path.basename(f))[
        0] for f in input_img_paths]
    target_basenames = [os.path.splitext(os.path.basename(f))[
        0] for f in target_img_paths]
    filtered_target_img_paths = [
        t for t in target_img_paths if os.path.splitext(os.path.basename(t))[0] in input_basenames
    ]

    # Debug: Check lengths after filtering
    print(
        f"Filtered inputs: {len(input_img_paths)}, Filtered targets: {len(filtered_target_img_paths)}")
    assert len(input_img_paths) == len(
        filtered_target_img_paths), "Mismatch after filtering!"

    # Split into training and validation sets
    train_input_img_paths = input_img_paths[:-val_samples]
    train_target_img_paths = filtered_target_img_paths[:-val_samples]
    val_input_img_paths = input_img_paths[-val_samples:]
    val_target_img_paths = filtered_target_img_paths[-val_samples:]

    # Create TensorFlow datasets - does this + one above do the same as "max dataset lenghts" - keras har ju gjort så att den bara gör ett dataset på 1000.
    def create_dataset(input_img_paths, target_img_paths):
        dataset = tf.data.Dataset.from_tensor_slices(
            (input_img_paths, target_img_paths))
        dataset = dataset.map(lambda x, y: load_img_and_mask(
            x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
        return dataset.batch(batch_size).shuffle(100)

    train_dataset = create_dataset(
        train_input_img_paths, train_target_img_paths)
    valid_dataset = create_dataset(val_input_img_paths, val_target_img_paths)

    return train_dataset, valid_dataset
