import torch
import torch.nn as nn
import torch.nn.functional as F  # Some of the classes can be directly used as Functions
import torch.utils.data as data

import wandb
from train import Trainer
from utils import *
from model import *
from data import *


# Freezing the Seed for Reproducibility
seed_everything(seed=42)  # The Answer to Everything

# Logging In to wandb
wandb.login()

# 🐝 initialise a wandb run
wandb.init(
    project="Final Runs",
    config={
        "LR": 1e-3,
        "NUM_EPOCHS": 30,
        "BATCH_SIZE": 16,
        "MODEL_PTH": "resnet101.pth",
    },
)
config = wandb.config

# Creating Dataset
ds = WaterLoggingDataset(root_dir="water_logging_dataset_v3")
train_data = ds("train", transforms=image_transforms["train"])
valid_data = ds("valid", transforms=image_transforms["valid"])
test_data = ds("test", transforms=image_transforms["valid"])

# Dataset Classes
classes = train_data.classes

# Creating DataLoader
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


# Model and Setups
# model = ResNextModel(num_classes=2)
model = ResNetModel(num_classes=2, model_name="resnet101")

print(f"The model has {count_parameters(model):,} trainable parameters")

# Defining Loss and Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
criterion = nn.CrossEntropyLoss()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initalizing Model
model = model.to(DEVICE)
criterion = criterion.to(DEVICE)

# Instantiating Trainer
trainer = Trainer(
    model,
    optimizer,
    criterion,
    DEVICE,
    config.NUM_EPOCHS,
    dataloaders["train"],
    dataloaders["valid"],
    dataloaders["test"],
    config.MODEL_PTH,
    wandb,
    classes,
)

# Running Training cycle
trainer.cycle()

# Completing wandb logging
wandb.finish()
