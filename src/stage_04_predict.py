import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from src.utils.common import read_yaml

STAGE = "Prediction"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def main(config_path):
    ## read config files
    config = read_yaml(config_path)

    # loading the saved .h5 model
    saved_model_file = os.path.join(
        config["source_data"]["local_dir"],
        config["source_data"]["model_dir"],
        config["source_data"]["trained_model_file"],
    )

    print(f"Trained model file: {saved_model_file}")

    model = tf.keras.models.load_model(saved_model_file)
    img = plt.imread(
        "data/PlantVillage/Potato___healthy/1a1184f8-c414-4ead-a4c4-41ae78e29a82___RS_HL 1971.JPG"
    )

    def predict(model, img):
        class_name = [
            "Potato___Early_blight",
            "Potato___Late_blight",
            "Potato___healthy",
        ]
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)

        prediction = model.predict(img_array)
        print(f"prediction: {prediction}")

        predicted_class = class_name[np.argmax(prediction[0])]
        confidence = round(100 * (np.max(prediction[0])), 2)

        return predicted_class, confidence

    result = predict(model, img)
    return result


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e