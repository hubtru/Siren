import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow import keras as k
from keras import layers
from sklearn.model_selection import train_test_split
import cv2
from generate_data import generate_data
from keras import backend as kb

# ============================== gpu info ==============================
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
print("GPUs: ", tf.config.list_physical_devices("GPU"))


# ============================== variables ==============================
# architecture
image_size = 256
filters = 256
depth = 4
columns = 4
kernel_size = 5
patch_size = 16
num_classes = 3

in_channel = 3
img_rows, img_cols = image_size, image_size
input_shape = (img_rows, img_cols, in_channel)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 8
epochs = 100

# ============================== paths ==============================
# names and paths; needed for generate_data
modality = "chip"
exp_name = f"siren_10"
sub_exp_name = f"base"
model_name = f"{exp_name}_{sub_exp_name}_{modality}"
model_save_path = f"{model_name}.h5"
generate_data_path = (
    f"/home/user/experiments/{exp_name}/{sub_exp_name}/{modality}"
)

# dataset
csv_path = "/home/user/dataset/labels.csv"
chip_path = f"/home/user/dataset/{modality}"


# ============================== load/prepare dataset ==============================
def create_dataset(test_tool: str):
    df = pd.read_csv(csv_path)
    np.random.seed(777)

    test_df = df[df["id"].str.contains(f"{test_tool}R")]  # all containing target tool
    train_df = df[~df["id"].str.contains(f"{test_tool}R")]  # everything else
    train_df, val_df = train_test_split(
        train_df, test_size=0.2
    )  # split to 80% training data
    print(
        "size check train/test/val",
        len(train_df),
        len(test_df),
        len(val_df),
        (len(train_df) + len(test_df) + len(val_df)),
    )

    train_df["chip"] = train_df.id.map(lambda id: f"{chip_path}/{id}.jpg")
    test_df["chip"] = test_df.id.map(lambda id: f"{chip_path}/{id}.jpg")
    val_df["chip"] = val_df.id.map(lambda id: f"{chip_path}/{id}.jpg")

    print("check if permutation is same on all models")
    print("train_head", train_df.head())
    print("test_head", test_df.head())
    print("val_head", val_df.head())

    print(f"check if train/val contains test_tool: ")
    if (
        len(train_df[train_df["id"].str.contains(f"{test_tool}R")]) > 0
        or len(val_df[val_df["id"].str.contains(f"{test_tool}R")]) > 0
        or len(test_df[~test_df["id"].str.contains(f"{test_tool}R")]) > 0
    ):
        raise ValueError(
            f"train or val dataframes contain {test_tool} or test contains other tool then {test_tool}"
        )
    else:
        print("check success")

    def read_tools(file_paths, img_rows, img_cols, channels):
        """
        Reads the spectogram files from disk and normalizes the pixel values
        @params:
            file_paths - Array of file paths to read from
            img_rows - The image height.
            img_cols - The image width.
            as_grey - Read the image as Greyscale or RGB.
            channels - Number of channels.
        @returns:
            The created and compiled model (Model)
        """
        images = []

        for file_path in file_paths:
            img = cv2.imread(file_path)
            res = cv2.resize(
                img, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC
            )
            images.append(res)

        images = np.asarray(images, dtype=np.float32)

        # normalize
        images = images / np.max(images)

        # reshape to match Keras expectaions
        images = images.reshape(images.shape[0], img_rows, img_cols, channels)

        return images

    x_train_chip = read_tools(train_df.chip.values, img_rows, img_cols, in_channel)
    x_test_chip = read_tools(test_df.chip.values, img_rows, img_cols, in_channel)
    x_val_chip = read_tools(val_df.chip.values, img_rows, img_cols, in_channel)

    labels_train = train_df.tool_label.values
    labels_test = test_df.tool_label.values
    labels_val = val_df.tool_label.values

    labels_train = labels_train - 1
    labels_test = labels_test - 1
    labels_val = labels_val - 1

    labels_train = tf.keras.utils.to_categorical(
        labels_train, num_classes=num_classes, dtype="float32"
    )
    labels_test = tf.keras.utils.to_categorical(
        labels_test, num_classes=num_classes, dtype="float32"
    )
    labels_val = tf.keras.utils.to_categorical(
        labels_val, num_classes=num_classes, dtype="float32"
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train_chip, labels_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test_chip, labels_test))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val_chip, labels_val))

    auto = tf.data.AUTOTUNE

    # ========== augment ==========
    data_augmentation = k.Sequential(
        [
            k.layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0, 1)),
            k.layers.RandomContrast(factor=0.2),
            k.layers.GaussianNoise(stddev=0.2),
        ],
        name="data_augmentation",
    )

    def generate_datasets(images, is_train=False, shuffle=False):
        dataset = images
        if shuffle:
            dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.batch(batch_size)
        if is_train:
            dataset = dataset.map(
                lambda x, y: (data_augmentation(x), y), num_parallel_calls=auto
            )
        return dataset.prefetch(auto)

    train_ds = generate_datasets(train_dataset, is_train=True, shuffle=True)
    val_ds = generate_datasets(val_dataset)
    test_ds = generate_datasets(test_dataset)

    return train_ds, val_ds, test_ds, labels_test


# ============================== build model ==============================
def activation_module(x):
    x = layers.Activation("gelu")(x)
    return layers.BatchNormalization()(x)


def base_module(x, filters: int, patch_size: int):
    x = layers.Conv2D(filters, kernel_size=patch_size, strides=patch_size)(x)
    return activation_module(x)


def multi_mixer_block(
    x, pe_block, column, level, depth, r_arr, filters: int, kernel_size: int
):
    if column == 0:
        # Depthwise convolution.
        x0 = x
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x)
        x = layers.Add()([activation_module(x), x0])  # Residual.

        # Pointwise convolution.
        x = layers.Conv2D(filters, kernel_size=1)(x)
        x = activation_module(x)

        r_arr = reverasable_output_array(x, r_arr, column, level)
        return x, r_arr
    else:
        if level == 0:
            x0 = fusion_block_double(r_arr[column - 1][level + 1], pe_block)
        elif level < len(r_arr[0]) - 1 and level != 0:
            x0 = fusion_block_double(r_arr[column - 1][level + 1], x)
        else:
            x0 = x
        x = layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same")(x0)
        x = layers.Add()([activation_module(x), x0])  # Residual.

        # Pointwise convolution.
        x = layers.Conv2D(filters, kernel_size=1)(x)
        x = activation_module(x)

        if column != 0:
            x = fusion_block_double(r_arr[column - 1][level], x)

        r_arr = reverasable_output_array(x, r_arr, column, level)

        return x, r_arr


def fusion_block_double(x, y):
    return layers.Add()([x, y])


def reverasable_output_array(x, r_arr, column, level):
    r_arr[column][level] = x
    return r_arr


def load_multi_mixer(
    image_size=image_size,
    filters=filters,
    depth=depth,
    columns=columns,
    kernel_size=kernel_size,
    patch_size=patch_size,
    num_classes=num_classes,
):
    r_arr = [[None for _ in range(depth)] for _ in range(columns)]

    inputs = k.Input(input_shape)
    # x = layers.Rescaling(scale=1.0 / 255)(inputs)         # already rescaled

    # Extract patch embeddings.
    x = base_module(inputs, filters, patch_size)
    pe_block = x

    # ConvMixer blocks.
    for column in range(columns):
        for level in range(depth):
            x, r_arr = multi_mixer_block(
                x, pe_block, column, level, depth, r_arr, filters, kernel_size
            )

    # Classification block.
    x = layers.GlobalAvgPool2D()(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return k.Model(inputs, outputs)


# ============================== run experiment ==============================

tools_list = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
for i in range(10):
    kb.clear_session()

    current_tool = tools_list[i]

    model_name = f"{exp_name}_{sub_exp_name}_{modality}_{current_tool}"
    model_save_path = f"{model_name}.h5"

    train_ds, val_ds, test_ds, labels_test = create_dataset(current_tool)

    def launch_experiment(model):
        optimizer = tfa.optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        model.compile(
            optimizer=optimizer,
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            verbose=2,
        )

        _, accuracy = model.evaluate(test_ds)
        print(f"Test accuracy {current_tool}: {round(accuracy * 100, 2)}%")

        return history, model

    model = load_multi_mixer()
    # model.summary()

    history, model = launch_experiment(model)
    model.save(model_save_path)

    pred = model.predict(test_ds)

    # ============================== generate data ==============================
    generate_data(
        model,
        history,
        test_ds,
        labels_test,
        num_classes,
        batch_size,
        name=model_name,
        path=generate_data_path,
    )
