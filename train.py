import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import listdir

import wandb
import os
import warnings
warnings.filterwarnings("ignore")

from Utils.categories import Categories
from Utils.data_loader import COCODataset, show_batch
from Model.model import YOLOv1
from Model.loss import YOLOv1_loss
from Model.metrics import mAP_tool, calculate_mAP
from Utils.inference import train, evaluate
from Utils.inference import predict_and_visualize
from Utils.utils import complete_path
from Utils.config import *


# Learning rate schedule
def lr_lambda(epoch):
    limits = np.array([0, 75, 105]) * EPOCHS / 135  # Scale limits to current number of epochs
    limits[0] = 5
    a = (1e-2 - 1e-3) / limits[0]
    b = 1e-3
    if epoch < limits[0]:
        res = a * epoch + b
    elif epoch < limits[1]:
        res = 1e-2
    elif epoch < limits[2]:
        res = 1e-3
    else:
        res = 1e-4
    return res / 1e-3

def show_evolution(losses, mAPs, savefig=True, figsize=(10, 10)):
    fig, axs = plt.subplots(1, 2, figsize=figsize)
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].plot(losses, color='tab:red')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('mAP')
    axs[1].plot(mAPs, color='tab:blue')
    plt.suptitle(f'Loss and mAP evolution, {model_name}, {EPOCHS} epochs')
    plt.tight_layout()
    
    if savefig:
        fig.savefig(f'{root}/plots/{model_name}loss_mAP_evolution{EPOCHS}.png')
    plt.show()

def save_model(model, optimizer, epoch, path, name):
    print(complete_path(path))
    if path and name:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f"{complete_path(path)}{name}.pt")
        print(f"Model saved to {complete_path(path)}{name}.pt")

def load_model(architecture, S, B, C, device, name=None, path=None, verbose=True, train=True, in_channels=3, image_resize=image_resize, linear_size=1024):
    model = YOLOv1(architecture, S, B, C, in_channels=in_channels, image_resize=image_resize, linear_size=linear_size).to(device).double()
    #if verbose:
        #summary(model, (in_channels, image_resize, image_resize), device=device.type)

    optimizer = None
    scheduler = None
    EPOCHS_DONE = 0

    if train:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda, verbose=True)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, threshold=0.1, verbose=True)

    if path and name:
        print(complete_path(path))
        try:
            checkpoint = torch.load(f"{complete_path(path)}{name}.pt", map_location=device) # Load to CPU or GPU
            model.load_state_dict(checkpoint['model_state_dict'])
            if train and optimizer is not None:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            EPOCHS_DONE = checkpoint['epoch']
            print(f"Model loaded from {complete_path(path)}{name}.pt, starting from epoch {EPOCHS_DONE}")

        except FileNotFoundError:
            print(f"Error: Model file not found at {complete_path(path)}{name}.pt.  Starting from scratch.")
    else:
        print("No model name specified, starting from scratch.")

    return model, optimizer, scheduler, EPOCHS_DONE



if __name__ == "__main__":
    # Initialize wandb
    wandb.init(project="Deep Learning for image Analysis", 
        name = "yolo-v1",
        config={
        "learning_rate": learning_rate,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "image_resize": image_resize,
        "architecture": architecture,
        "S": S,
        "B": B,
        "C": C,
        "lambda_coord": lambda_coord,
        "lambda_noobj": lambda_noobj,
        "iou_threshold": iou_threshold,
        "confidence_threshold": confidence_threshold,
        "device": str(device)
    })

    config = wandb.config  # Access wandb config

    # -------------------- Data Preparation --------------------
    categories = Categories(dataset_path, C)
    train_dataset = COCODataset('train', dataset_path, S, B, categories, image_resize=image_resize, grayscale=grayscale, max_nb_datapoints=ntrain)
    #val_dataset = COCODataset('val', dataset_path, S, B, categories, image_resize=image_resize, grayscale=grayscale, max_nb_datapoints=nval)
    val_dataset = COCODataset('val', dataset_path, S, B, categories, image_resize=image_resize, grayscale=grayscale, max_nb_datapoints=nval, in_channels=in_channels)
    
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Show a batch of training images with bounding box
    dataloader_show = DataLoader(train_dataset, batch_size=4, shuffle=True)
    for sample_batched in dataloader_show:
        show_batch(sample_batched, dataloader_show.dataset, rescaled=False, model=None)
        break # Show only one batch

    # -------------------- Model and Training Setup --------------------
    avg_precision_helper = mAP_tool(S, B, C, iou_threshold=config.iou_threshold, confidence_threshold=config.confidence_threshold)
    weights = categories.frequencies.iloc[:C].values / categories.frequencies.iloc[:C].sum()
    loss_fn = YOLOv1_loss(S, B, C, lambda_coord=config.lambda_coord, lambda_noobj=config.lambda_noobj, weights=weights)

    model, optimizer, scheduler, EPOCHS_DONE = load_model(architecture, S, B, C, device, name=model_name, path=LOAD_PATH,
                                                          train=True, in_channels=in_channels, image_resize=image_resize, linear_size=linear_size)
    # -------------------- Training Loop --------------------
    losses, mAPs = [], []

    # Check the the training process needs to run or not
    if EPOCHS > 0 and (not f'{root}/lossandmap/landm_{model_name}.csv' in listdir(f'{root}/lossandmap')):  
        with open(f'{root}/lossandmap/landm_{model_name}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'mAP'])

        for epoch in range(EPOCHS - EPOCHS_DONE):
            print(f"EPOCH: {epoch + 1 + EPOCHS_DONE}/{EPOCHS}\n")
            train(train_dataloader, model, loss_fn, optimizer, segments = segments)
            APs, recalls, precisions, F1s, test_loss = evaluate(val_dataloader, model, loss_fn, avg_precision_helper)

            # WANDB Logging:
            mAP = calculate_mAP(APs)
            wandb.log({"mAP": mAP, "val_loss": test_loss, "epoch": epoch + 1 + EPOCHS_DONE})


            if scheduler:
                scheduler.step(test_loss)
    
            save_model(model, optimizer, epoch + 1 + EPOCHS_DONE, SAVE_PATH, model_name)  # Pass model and optimizer
            
            with open(f'{root}/lossandmap/landm_{model_name}.csv', 'a') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, test_loss, np.mean(APs)])

            losses.append(test_loss)
            mAPs.append(calculate_mAP(APs))

    else:
        print("Skipping training as EPOCHS is 0 or lossandmap file exists")



    with open(f'{root}/lossandmap/landm_{model_name}.csv', 'r') as f:
        reader = csv.reader(f)
        next(reader)
        losses, mAPs = [], []
        for row in reader:
            losses.append(float(row[1]))
            mAPs.append(float(row[2]))

    # -------------------- Evaluation and Visualization --------------------
    APs, recalls, precisions, F1s, test_loss = evaluate(val_dataloader, model, loss_fn, avg_precision_helper, wandb_process = False) # set to false to prevent logging to wandb twice
    show_evolution(losses, mAPs, savefig=True)

    # Show a batch of validation images with predicted bounding boxes
    dataloader_val = DataLoader(val_dataset, batch_size=4, shuffle=True)
    for sample_batched in dataloader_val: 
        show_batch(sample_batched, dataloader_val.dataset, rescaled=True, compute_label=True, lbl_txt_scale=0.8, model=model)
        wandb.log({"validation_batch": [wandb.Image(sample_batched['image'][i].cpu().numpy().transpose((1, 2, 0)), caption=f"Image {i}") for i in range(len(sample_batched['image']))]})# Show image batch to wandb
        break # Show only one batch

    # -------------------- Inference and Visualization on a New Image --------------------
    
    new_image_path = "/content/input/val2017/000000000139.jpg"  # Replace with the path to your new image
    predict_and_visualize(f"{complete_path(SAVE_PATH)}{model_name}.pt", new_image_path, val_dataset, conf_threshold=0.5, rescaled=True)
    wandb.finish()
    