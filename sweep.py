import torch
import torch.nn as nn
import torch.nn.functional as F  # Some of the classes can be directly used as Functions
import torch.utils.data as data
from torch.cuda.amp import GradScaler

from train import Trainer
from utils import *
from model import *
from data import *

seed_everything(seed=42)  # The Answer to Everything

import wandb

wandb.login()


sweep_config = {
    "method": "random",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "BATCH_SIZE": {"values": [8, 16]},
        "NUM_EPOCHS": {"value": 10},
        "LR": {"values": [1e-3, 3e-4, 1e-4]},
        "model": {"values": ["resnet34", "resnet101", "resnext"]},
    },
}

ds = WaterLoggingDataset(root_dir="water_logging_dataset_v3")

train_data = ds("train", transforms=image_transforms["train"])
valid_data = ds("valid", transforms=image_transforms["valid"])
test_data = ds("test", transforms=image_transforms["valid"])

classes = train_data.classes

sweep_id = wandb.sweep(sweep_config, project="test")


def run_sweep(config=None):
    with wandb.init(project="water_logging", entity="Rohitkuk", config=config):
        config = wandb.config
        dataloaders = {
            "train": data.DataLoader(
                train_data,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            ),
            "valid": data.DataLoader(
                valid_data,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            ),
            "test": data.DataLoader(
                test_data,
                batch_size=config.BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=True,
                drop_last=True,
            ),
        }
        if config.model == "resnet34":
            model = ResNetModel(num_classes=2, model_name="resnet34")
        elif config.model == "resnet101":
            model = ResNetModel(num_classes=2, model_name="resnet101")
        elif config.model == "resnext":
            model = ResNextModel(num_classes=2, model_name="resnext50d_32x4d")

        print(f"The model has {count_parameters(model):,} trainable parameters")
        optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        criterion = nn.CrossEntropyLoss()
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(DEVICE)
        criterion = criterion.to(DEVICE)
        MODEL_PTH = "Sweep.pth"
        trainer = Trainer(
            model,
            optimizer,
            criterion,
            DEVICE,
            config.NUM_EPOCHS,
            dataloaders["train"],
            dataloaders["valid"],
            dataloaders["test"],
            MODEL_PTH,
            wandb,
            classes,
        )

        trainer.cycle()


wandb.agent(sweep_id, function=run_sweep, count=5)
