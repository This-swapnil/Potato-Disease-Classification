import argparse
import logging
import os

import tensorflow as tf
from matplotlib import pyplot as plt

from src.utils.common import read_yaml

STAGE = "Training"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def train(
    config_path,
):
    ## read config files
    config = read_yaml(config_path)
    params = config["params"]

    # get ready the data

    PARENT_DIR = os.path.join(
        config["source_data"]["unzip_data"],
        config["source_data"]["parent_data_dir"],
    )

    logging.info(f"Read the data form {PARENT_DIR}")

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        validation_split=params["validation_split"],
        subset="training",
        seed=params["seed"],
        image_size=params["img_size"][:-1],
        batch_size=params["BATCH_SIZE"],
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        PARENT_DIR,
        validation_split=params["validation_split"],
        subset="validation",
        seed=params["seed"],
        image_size=params["img_size"][:-1],
        batch_size=params["BATCH_SIZE"],
    )

    train_ds = train_ds.prefetch(buffer_size=params["buffer_size"])
    val_ds = val_ds.prefetch(buffer_size=params["buffer_size"])

    # Data Augmentation
    data_augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.RandomFlip(
                "horizontal_and_vertical"
            ),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ]
    )
    # Applying Data Augmentation to Train Dataset
    train_ds = train_ds.map(
        lambda x, y: (data_augmentation(x, training=True), y)
    ).prefetch(buffer_size=params["buffer_size"])

    ## load the base model
    path_to_model = os.path.join(
        config["source_data"]["local_dir"],
        config["source_data"]["model_dir"],
        config["source_data"]["init_model_file"],
    )

    logging.info(f"load the base model from {path_to_model}")

    classifier = tf.keras.models.load_model(path_to_model)

    # training

    history = classifier.fit(train_ds, epochs=params["epochs"], validation_data=val_ds)

    print(f"History keys: \n{history.history.keys()}")

    trained_model_file = os.path.join(
        config["source_data"]["local_dir"],
        config["source_data"]["model_dir"],
        config["source_data"]["trained_model_file"],
    )

    classifier.save(trained_model_file)
    logging.info(f"trained model is saved at : {trained_model_file}")

    # logging acc and loss
    acc = history.history["Accuracy"]
    val_acc = history.history["val_Accuracy"]
    logging.info(
        f"\nTraining Accuracy : {trained_model_file} and Validation accuracy{val_acc}"
    )

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    logging.info(f"\nTraining loss : {trained_model_file} and Validation loss{val_acc}")

    ## Saving the "Training and Validation Accuracy" and "Training and Validation Loss" graph
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(range(params["epochs"]), acc, label="Training Accuracy")
    plt.plot(range(params["epochs"]), val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.title("Training and Validation Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(range(params["epochs"]), loss, label="Training Loss")
    plt.plot(range(params["epochs"]), val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.title("Training and Validation Loss")
    # plt.show()
    plt.savefig("Training_Validation_Accuracy_Loss.jpg")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        train(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e