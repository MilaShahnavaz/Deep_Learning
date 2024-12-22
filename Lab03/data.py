import os
import tensorflow as tf


def load_img_and_mask(input_img_path, target_img_path, img_size):
    """
    Load and preprocess an input image and its corresponding segmentation mask.
    """
    try:
        # Load and preprocess the input image
        input_img = tf.io.read_file(input_img_path)
        input_img = tf.io.decode_jpeg(input_img, channels=3)
        input_img = tf.image.resize(input_img, img_size)
        input_img = tf.image.convert_image_dtype(input_img, tf.float32)

        # Load and preprocess the target mask
        target_img = tf.io.read_file(target_img_path)
        target_img = tf.io.decode_png(target_img, channels=1)
        target_img = tf.image.resize(target_img, img_size, method="nearest")
        target_img = tf.image.convert_image_dtype(target_img, tf.uint8)
        target_img -= 1  # Adjust mask labels to start at 0

        return input_img, target_img
    except Exception as e:
        print(f"Error processing {input_img_path} or {target_img_path}: {e}")
        return None, None


def get_dataset(batch_size, img_size, input_img_paths, target_img_paths, max_dataset_len=None):
    """
    Create a TensorFlow dataset for the given input and target image paths.
    """
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]

    def load_func(input_img_path, target_img_path):
        return load_img_and_mask(input_img_path, target_img_path, img_size)

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_img_paths, target_img_paths))
    dataset = dataset.map(lambda x, y: load_func(
        x, y), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(lambda x, y: x is not None and y is not None)
    return dataset.batch(batch_size)


def prepare_paths(input_dir, target_dir):
    """
    Prepare sorted lists of input and target image paths.
    """
    input_img_paths = sorted(
        [os.path.join(input_dir, fname)
         for fname in os.listdir(input_dir) if fname.endswith(".jpg")]
    )
    target_img_paths = sorted(
        [os.path.join(target_dir, fname) for fname in os.listdir(
            target_dir) if fname.endswith(".png") and not fname.startswith("._")]
    )

    return input_img_paths, target_img_paths
