import os
import warnings

warnings.filterwarnings("ignore")
import yaml
from easydict import EasyDict
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger

from models import TranMILMInterface
from dataloaders import ClAMDInterface
from utils.load_best_model import load_model_path_by_args
from utils import set_seeds
import pandas as pd
import datetime

def load_callbacks():
    callbacks = []
    callbacks.append(
        plc.EarlyStopping(monitor="val_acc", mode="max", patience=25, min_delta=1)
    )
    callbacks.append(
        plc.ModelCheckpoint(
            monitor="val_acc",
            filename="best-{epoch:02d}-{tra_acc:.3f}-{tra_loss:.3f}-{val_acc:.3f}-{val_loss:.3f}",
            save_top_k=1,
            mode="max",
        )
    )

    if config.lr_scheduler:
        callbacks.append(plc.LearningRateMonitor(logging_interval="epoch"))
    return callbacks


def main(config):
    set_seeds.set_seed(config.seed)
    data_module = ClAMDInterface(**vars(config))
    model = TranMILMInterface(**vars(config))
    callbacks = load_callbacks()
    # If you want to change the logger's saving folder

    logger = TensorBoardLogger(
        save_dir=config.log_dir, name=config.log_name, version="fold_{}".format(config.k)
    )
    trainer = Trainer(
        precision=config.precision,
        max_epochs=config.max_epochs,
        devices=config.cuda,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        accumulate_grad_batches=2
    )
    if config.mode == "train":
        print("Training")
        trainer.fit(model, data_module)
    best_model_path = load_model_path_by_args(config)
    return trainer.test(model, data_module, ckpt_path=best_model_path)[0]


if __name__ == "__main__":
    with open("config/transmil.yaml", encoding="utf-8") as f:
        file = f.read()
    config = yaml.load(file, yaml.FullLoader)
    config = EasyDict(config)
    results = {}
    for k in range(config.kfold):
        print(f"Training fold {k}.")
        config.k = k
        config.save_dir = config.osave_dir+f"/fold_{k}"
        result=main(config)
        results[f"fold_{k}"] = result

    
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    excel_file = config.osave_dir+f"/transmil_results_{timestamp}.xlsx"
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results).T
    df.loc['mean'] = df.mean()
    df.loc['std'] = df.std()
    df.to_excel(excel_file)
    print(f"Results saved to: {excel_file}")
