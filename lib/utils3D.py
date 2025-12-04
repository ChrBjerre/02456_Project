from monai.networks.nets import DenseNet121
from monai.networks.nets import resnet
from lib.datasets.BugNIST_3D import BugNIST_3D
from lib.models.models import ourConv3D
from torch.utils.data import DataLoader
from monai.transforms import Compose, EnsureChannelFirst, ScaleIntensity, ToTensor
import torch
from torch import nn, optim
import wandb
import argparse
from pathlib import Path
from lib.models.models import UNetEncoderClassifier
import time

def build_model_3d(name: str, dropout_prob: float=0.2, in_channels: int=1, num_classes: int=12):

    name = name.lower()

    # DenseNet121
    if name in ("densenet", "densenet121"):
        model = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=num_classes,
            pretrained=False,
            dropout_prob=dropout_prob
        )
        return model

    # ResNet18
    elif name == "resnet":
        model = resnet.resnet18(
            spatial_dims=3,
            n_input_channels=in_channels,
            num_classes=num_classes,
            pretrained=False,   # HAS BEEN CHANGED TO FALSE!!!!!
        )
        return model
    
    # UNet
    elif name == "unet":
        model = UNetEncoderClassifier(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            )

        return model
    
    # Our Conv3D
    elif name == "conv3d":
        model = ourConv3D(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
        )
        return model
    
    else:
        raise ValueError(f"Model {name} not recognized. Use 'densenet', 'resnet', 'unet', or 'conv3d'.")
    


def build_dataloaders_3d(dataset_path: str, batch_size: int=4, num_workers: int=4):

    # Testing transforms...

    transform = Compose([
        EnsureChannelFirst(channel_dim=0),  # add C and put it first
        ScaleIntensity(),                   # optional normalization
        ToTensor(dtype=torch.float32),      # return torch.Tensor
    ])

    dataset_train = BugNIST_3D(dataset_path, split='train', transform=transform)
    dataset_val = BugNIST_3D(dataset_path, split='val', transform=transform)
    dataset_test = BugNIST_3D(dataset_path, split='test', transform=transform)

    dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return dataloader_train, dataloader_val, dataloader_test



def train_one_run(args):
    # Create run name based on parameters
    run_name = (
        f"{args.model_name}"
        f"_lr{args.learning_rate}"
        f"_dp{args.dropout_prob}"
        f"_{Path(args.dataset_path).name}"
    )

    # Ensure save directory exists
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{run_name}.pt"


    # Read configuration from Weights&Biases
    wandb.init(
        entity="s224389-dl-project",
        project=args.project_name,
        config={
            "model_type": args.model_name,
            "learning_rate": args.learning_rate,
            "dataset": Path(args.dataset_path).name,
            "dataset_path": args.dataset_path,
            "dropout_prob": args.dropout_prob,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "save_dir": str(save_dir),
        },
        name=run_name,
    )
    config = wandb.config


    # Set up dataloaders and model
    train_loader, val_loader, test_loader = build_dataloaders_3d(config.dataset_path, config.batch_size)
    model = build_model_3d(config.model_type, config.dropout_prob, in_channels=1, num_classes=12)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Send to GPU if available
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)


    best_val_acc = 0.0

    print("-"*40)
    print(f"Starting training for {run_name}")
    print("-"*40)
    for epoch in range(config.epochs):
        time1 = time.time()

        ### Training
        model.train()
        train_loss_sum = 0.0
        train_correct = 0
        train_total = 0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            # Accumulate loss
            train_loss_sum += loss.item() * xb.size(0)

            # Accumulate correct predictions
            pred_labels = preds.argmax(dim=1)
            train_correct += (pred_labels == yb).sum().item()
            train_total += yb.size(0)

        avg_train_loss = train_loss_sum / train_total
        train_acc = train_correct / train_total


        ### Validation
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)

                val_loss_sum += loss.item() * xb.size(0)

                pred_labels = preds.argmax(dim=1)
                val_correct += (pred_labels == yb).sum().item()
                val_total += yb.size(0)

        avg_val_loss = val_loss_sum / val_total
        val_acc = val_correct / val_total

        ### Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            wandb.summary["best_val_accuracy"] = best_val_acc
            wandb.summary["best_model_path"] = str(model_path)

        # Track time
        time2 = time.time()
        epoch_time = time2 - time1

        ### Log to W&B
        wandb.log({
            'epoch': epoch+1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_acc,
            'val_loss': avg_val_loss,
            'val_accuracy': val_acc,
            'epoch_time_sec': epoch_time,
        })
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{config.epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, choices=["densenet", "resnet", "unet", "conv3d"], required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./models/")
    parser.add_argument("--project_name", type=str, required=True)
    return parser.parse_args()