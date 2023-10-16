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
image_size = 256
num_classes = 3

in_channel_tool = 3
in_channel_spec = 9
img_rows, img_cols = image_size, image_size
input_shape_tool = (img_rows, img_cols, in_channel_tool)
input_shape_spec = (img_rows, img_cols, in_channel_spec)

learning_rate = 0.001
weight_decay = 0.0001
batch_size = 8
epochs = 100

# ============================== paths ==============================
# names and paths; needed for generate_data
modality = "multi_ts"
exp_name = f"siren_10"
sub_exp_name = f"base"
model_name = f"{exp_name}_{sub_exp_name}_{modality}"
model_save_path = f"{model_name}.h5"
generate_data_path = (
    f"/home/user/experiments/{exp_name}/{sub_exp_name}/{modality}"
)

# dataset
csv_path = "/home/user/dataset/labels.csv"
tool_path = "/home/user/dataset/tool"
spec_path = "/home/user/dataset/spec"


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

    train_df["tool"] = train_df.id.map(lambda id: f"{tool_path}/{id}.jpg")
    test_df["tool"] = test_df.id.map(lambda id: f"{tool_path}/{id}.jpg")
    val_df["tool"] = val_df.id.map(lambda id: f"{tool_path}/{id}.jpg")

    train_df["spec_x"] = train_df.id.map(lambda id: f"{spec_path}/x/{id}.jpg")
    test_df["spec_x"] = test_df.id.map(lambda id: f"{spec_path}/x/{id}.jpg")
    val_df["spec_x"] = val_df.id.map(lambda id: f"{spec_path}/x/{id}.jpg")

    train_df["spec_y"] = train_df.id.map(lambda id: f"{spec_path}/y/{id}.jpg")
    test_df["spec_y"] = test_df.id.map(lambda id: f"{spec_path}/y/{id}.jpg")
    val_df["spec_y"] = val_df.id.map(lambda id: f"{spec_path}/y/{id}.jpg")

    train_df["spec_z"] = train_df.id.map(lambda id: f"{spec_path}/z/{id}.jpg")
    test_df["spec_z"] = test_df.id.map(lambda id: f"{spec_path}/z/{id}.jpg")
    val_df["spec_z"] = val_df.id.map(lambda id: f"{spec_path}/z/{id}.jpg")

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

    def read_specs(
        file_paths_x, file_paths_y, file_paths_z, img_rows, img_cols, channels
    ):
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

        for i in range(file_paths_x.size):
            img_x = cv2.imread(file_paths_x[i])
            res_x = cv2.resize(
                img_x, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC
            )
            images.append(res_x)
            img_y = cv2.imread(file_paths_y[i])
            res_y = cv2.resize(
                img_y, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC
            )
            images.append(res_y)
            img_z = cv2.imread(file_paths_z[i])
            res_z = cv2.resize(
                img_z, dsize=(img_rows, img_cols), interpolation=cv2.INTER_CUBIC
            )
            images.append(res_z)

        images = np.asarray(images, dtype=np.float32)

        # normalize
        images = images / np.max(images)

        # reshape to match Keras expectaions
        images = images.reshape(file_paths_x.shape[0], img_rows, img_cols, channels)

        return images

    x_train_tool = read_tools(train_df.tool.values, img_rows, img_cols, in_channel_tool)
    x_test_tool = read_tools(test_df.tool.values, img_rows, img_cols, in_channel_tool)
    x_val_tool = read_tools(val_df.tool.values, img_rows, img_cols, in_channel_tool)

    x_train_spec = read_specs(
        train_df.spec_x.values,
        train_df.spec_y.values,
        train_df.spec_z.values,
        img_rows,
        img_cols,
        in_channel_spec,
    )
    x_test_spec = read_specs(
        test_df.spec_x.values,
        test_df.spec_y.values,
        test_df.spec_z.values,
        img_rows,
        img_cols,
        in_channel_spec,
    )
    x_val_spec = read_specs(
        val_df.spec_x.values,
        val_df.spec_y.values,
        val_df.spec_z.values,
        img_rows,
        img_cols,
        in_channel_spec,
    )

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

    # ========== augment 1/2 ==========
    data_augmentation_tool = k.Sequential(
        [
            k.layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0, 1)),
            k.layers.RandomContrast(factor=0.2),
            k.layers.GaussianNoise(stddev=0.2),
        ],
        name="data_augmentation_tool",
    )

    # ========== augment 2/2 ==========
    data_augmentation_spec = k.Sequential(
        [
            k.layers.RandomBrightness(factor=(-0.2, 0.2), value_range=(0, 1)),
            k.layers.RandomContrast(factor=0.2),
            k.layers.GaussianNoise(stddev=0.2),
        ],
        name="data_augmentation_spec",
    )

    x_train_tool = data_augmentation_tool(x_train_tool)
    x_train_spec = data_augmentation_spec(x_train_spec)

    train_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_1_tool": x_train_tool,
                "input_1_spec": x_train_spec,
            },
            labels_train,
        )
    )
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_1_tool": x_test_tool,
                "input_1_spec": x_test_spec,
            },
            labels_test,
        )
    )
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (
            {
                "input_1_tool": x_val_tool,
                "input_1_spec": x_val_spec,
            },
            labels_val,
        )
    )

    auto = tf.data.AUTOTUNE

    def generate_datasets(images, is_train=False, shuffle=False):
        dataset = images
        if shuffle:
            dataset = dataset.shuffle(batch_size * 10)
        dataset = dataset.batch(batch_size)
        if is_train:
            pass
        return dataset.prefetch(auto)

    train_ds = generate_datasets(train_dataset, is_train=True, shuffle=True)
    val_ds = generate_datasets(val_dataset)
    test_ds = generate_datasets(test_dataset)

    return train_ds, val_ds, test_ds, labels_test


# ============================== build/load model(s) ==============================
def load_siren(model_tool, model_spec):
    inputs1 = model_tool.input
    inputs1._name = "input_tool"
    inputs2 = model_spec.input
    inputs2._name = "input_spec"

    truncated_model_tool = k.Model(
        inputs=model_tool.input, outputs=model_tool.layers[-2].output
    )
    truncated_model_spec = k.Model(
        inputs=model_spec.input, outputs=model_spec.layers[-2].output
    )
    mergedOut = k.layers.Concatenate()(
        [
            truncated_model_tool.output,
            truncated_model_spec.output,
        ]
    )
    output = layers.Dense(num_classes, activation="softmax", name="output_mult")(
        mergedOut
    )
    return k.Model(inputs=[inputs1, inputs2], outputs=output)


# ============================== run experiment ==============================
tools_list = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10"]
for i in range(10):
    kb.clear_session()

    current_tool = tools_list[i]

    model_name = f"{exp_name}_{sub_exp_name}_{modality}_{current_tool}"
    model_save_path = f"{model_name}.h5"

    train_ds, val_ds, test_ds, labels_test = create_dataset(current_tool)

    model_tool_path = f"/home/user/experiments/{exp_name}/{sub_exp_name}/tool/{exp_name}_{sub_exp_name}_tool_{current_tool}.h5"
    model_spec_path = f"/home/user/experiments/{exp_name}/{sub_exp_name}/spec/{exp_name}_{sub_exp_name}_spec_{current_tool}.h5"

    model_tool = k.models.load_model(model_tool_path, compile=False)
    model_spec = k.models.load_model(model_spec_path, compile=False)

    model_tool.trainable = False
    model_spec.trainable = False

    for layer in model_tool.layers:
        layer._name = layer.name + str("_tool")
    for layer in model_spec.layers:
        layer._name = layer.name + str("_spec")

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

    model = load_siren(model_tool, model_spec)
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
