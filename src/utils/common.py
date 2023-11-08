import os
from zipfile import ZipFile
import yaml
import logging
import time
import pandas as pd
import json
import shutil


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content


def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def unzip_file(source: str, dest: str) -> None:
    logging.info(f"Extraction Started...")
    with ZipFile(source, "r") as zip_f:
        zip_f.extractall(dest)
    logging.info(f"Extracted {source} to {dest}")


def remove_directories(source: str) -> None:
    logging.info(f"Deleting unwanted directories...")
    list_dir = os.listdir(os.path.join(source,'PlantVillage'))
    wanted_dir = [
        "Potato___Early_blight",
        "Potato___healthy",
        "Potato___Late_blight",
    ]
    for dir in list_dir:
        if dir not in wanted_dir:
            shutil.rmtree(os.path.join("data/PlantVillage/", dir))
    logging.info(f"Unwanted directories are deleted!!!")
