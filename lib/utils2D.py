from lib.datasets.BugNIST_2D import BugNIST_2D, BugNIST_2D_1views, BugNIST_2D_2views
from lib.models.models import ourConv3D
from torch.utils.data import DataLoader
import torch
from torch import nn, optim
import wandb
import argparse
from pathlib import Path
from lib.models.models import UNetEncoderClassifier
import time
from lib.models.models2D import ourConv2D, LateFusionResNet18, LateFusionDenseNet121, LateFusionUNetEncoder
import torchvision.transforms as T
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np



def build_model_2d(name: str, dropout_prob: float=0.2, in_channels: int=3, num_classes: int=12, pretrained: bool=False, views: int=6):

    name = name.lower()

    # DenseNet121
    if name in ("densenet", "densenet121"):
        model = LateFusionDenseNet121(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_prob=dropout_prob,
            num_views=views
        )
        return model

    # ResNet18
    elif name == "resnet":
        model = LateFusionResNet18(
            num_classes=num_classes,
            pretrained=pretrained,
            dropout_prob=dropout_prob,
            num_views=views
        )
        return model
    
    # UNet
    elif name == "unet":
        model = LateFusionUNetEncoder(
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            num_views=views
            )

        return model
    
    # Our Conv2D
    elif name == "conv2d":
        model = ourConv2D(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_prob=dropout_prob,
            num_views=views
        )
        return model
    
    else:
        raise ValueError(f"Model {name} not recognized. Use 'densenet', 'resnet', 'unet', or 'conv2d'.")
    


def build_dataloaders_2d(dataset_path: str, batch_size: int=4, num_workers: int=4, views: int=6):

    if views == 1:
        mean = [0.9304, 0.9304, 0.9304]
        std = [0.1985, 0.1985, 0.1985]
        transform = T.Compose([T.Resize((88, 200)), T.ToTensor(), T.Normalize(mean=mean, std=std)]) 
        # HALVING THE IMAGES SIZE (close to 256**2)

        dataset_train = BugNIST_2D_1views(dataset_path, split='train', transform=transform)
        dataset_val = BugNIST_2D_1views(dataset_path, split='val', transform=transform)
        dataset_test = BugNIST_2D_1views(dataset_path, split='test', transform=transform)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    elif views == 2:
        mean = [0.9304, 0.9304, 0.9304]
        std = [0.1984, 0.1984, 0.1984]
        transform = T.Compose([T.Resize((88, 200)), T.ToTensor(), T.Normalize(mean=mean, std=std)]) 
        # HALVING THE IMAGES SIZE (close to 256**2)

        dataset_train = BugNIST_2D_2views(dataset_path, split='train', transform=transform)
        dataset_val = BugNIST_2D_2views(dataset_path, split='val', transform=transform)
        dataset_test = BugNIST_2D_2views(dataset_path, split='test', transform=transform)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    elif views == 6:
        mean = [0.9366, 0.9366, 0.9366]
        std = [0.1920, 0.1920, 0.1920]
        transform = T.Compose([T.Resize((88, 200)), T.ToTensor(), T.Normalize(mean=mean, std=std)]) 
        # HALVING THE IMAGES SIZE (close to 256**2)

        dataset_train = BugNIST_2D(dataset_path, split='train', transform=transform)
        dataset_val = BugNIST_2D(dataset_path, split='val', transform=transform)
        dataset_test = BugNIST_2D(dataset_path, split='test', transform=transform)

        dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        dataloader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)
        dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=True)

    else:
        raise ValueError("views must be 1, 2, or 6.")

    return dataloader_train, dataloader_val, dataloader_test





def train_one_run(args):
    # Create run name based on parameters
    run_name = (
        f"{args.model_name}"
        f"_lr{args.learning_rate}"
        f"_dp{args.dropout_prob}"
        f"_pretrained={args.pretrained}"
        f"_{Path(args.dataset_path).name}"
        f"_views{args.views}"
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
            "pretrained": args.pretrained,
            "views": args.views,
            "LRScheduler": "MultiStepLR (33%, 67% epochs, gamma=0.25)",
        },
        name=run_name,
    )
    config = wandb.config


    # Set up dataloaders and model
    train_loader, val_loader, test_loader = build_dataloaders_2d(config.dataset_path, config.batch_size, views=config.views)
    model = build_model_2d(config.model_type, config.dropout_prob, in_channels=3, num_classes=12, pretrained=config.pretrained, views=config.views)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Send to GPU if available
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    scheduler = MultiStepLR(optimizer, milestones=np.linspace(0, config.epochs, 4)[1:3], gamma=0.25) # decay LR at 33% and 67% of total epochs

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

        scheduler.step() # step the LR scheduler

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


    # After training: load BEST model and evaluate on TEST set
    print("-" * 40)
    print("Finished training. Loading best model for test evaluation...")
    print(f"Best model path: {model_path}")
    print("-" * 40)

    # Load best model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    test_loss_sum = 0.0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            loss = criterion(preds, yb)

            test_loss_sum += loss.item() * xb.size(0)

            pred_labels = preds.argmax(dim=1)
            test_correct += (pred_labels == yb).sum().item()
            test_total += yb.size(0)

    avg_test_loss = test_loss_sum / test_total
    test_acc = test_correct / test_total

    print(
        f"Test Loss: {avg_test_loss:.4f}, "
        f"Test Accuracy (best-val model): {test_acc:.4f}"
    )
    wandb.summary["test_loss"] = avg_test_loss
    wandb.summary["test_accuracy"] = test_acc


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, choices=["densenet", "resnet", "unet", "conv2d"], required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--dropout_prob", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_dir", type=str, default="./models/")
    parser.add_argument("--project_name", type=str, required=True)
    parser.add_argument("--pretrained", type=bool, default=False)
    parser.add_argument("--views", type=int, choices=[1, 2, 6], default=6)
    return parser.parse_args()
