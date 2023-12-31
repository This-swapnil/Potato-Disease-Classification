import argparse
import logging
import os

import mlflow

STAGE = "STAGE_NAME"  ## <<< change stage name

logging.basicConfig(
    filename=os.path.join("logs", "running_logs.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a",
)


def main():
    with mlflow.start_run() as run:
        mlflow.run(".", "get_data", use_conda=False)
        mlflow.run(".", "base_model_creation", use_conda=False)
        mlflow.run(".", "training", use_conda=False)
        mlflow.run(".", "predicting", use_conda=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        main()
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e