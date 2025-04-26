import torch
from tqdm import tqdm
from time import perf_counter as pc
import numpy as np
import pandas as pd
import wandb
from torch.utils.checkpoint import checkpoint_sequential
from Utils.config import device
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from Model.metrics import calculate_mAP
from Model.model import YOLOv1
from Utils.config import architecture, linear_size, root
from Utils.data_loader import Sample


def train(dataloader, model, loss_fn, optimizer, segments):
    model.train()
    loop = tqdm(dataloader, total=len(dataloader))
    loop.set_postfix_str(f"loss:{.0:>7f}")
    loop.set_description('Training')
    tt = 0
    tm = pc()
    for sample_batched in loop:
        t = pc()
        x, y_true = sample_batched['image'].to(device), sample_batched['label'].to(device)
        optimizer.zero_grad()

        modules = [module for _, module in model._modules.items()]
        y_pred = checkpoint_sequential(modules, segments, x) # Gradient checkpointing
        loss = loss_fn(y_pred, y_true)
        loop.set_postfix_str(f"loss:{loss.item():>7f}")

        loss.backward()
        optimizer.step()
        wandb.log({"batch_loss": loss.item()})  # Log batch loss to wandb
        tt += pc() - t
        tm = pc() - tm
    print(
        f"Time spent loading data: {tm - tt:.2f}s, time spent training: {tt:.2f}s ({(tm - tt) / tm:.2f}/{tt / tm:.2f})")
    
    

def evaluate(dataloader, model, loss_fn, avg_precision_helper, wandb_process = True, eps=1e-8):

    num_batches = len(dataloader)
    test_loss = 0
    df = pd.DataFrame(columns=['object_present', 'confidence', 'TP', 'FP', 'class'])
    model.eval()
    loop = tqdm(dataloader, total=num_batches)
    loop.set_description('Evaluating')
    with torch.no_grad():
        for sample_batched in loop:
            x, y_true = sample_batched['image'].to(device), sample_batched['label'].to(device)
            y_pred = model(x)
            test_loss += loss_fn(y_pred, y_true).item()
            temp = avg_precision_helper(y_pred, y_true)
            df = pd.concat([df, pd.DataFrame(temp, columns=df.columns)], ignore_index=True)

    df = df[df['object_present'] == 1].loc[:, ['confidence', 'TP', 'FP', 'class']].sort_values(by='confidence',
                                                                                                 ascending=False)
    APs = []
    recalls = []
    precisions = []
    F1s = []
    for k in range(dataloader.dataset.C):
        subdf = df[df['class'] == k]
        accTP = subdf.TP.cumsum()
        accFP = subdf.FP.cumsum()
        precision = accTP / (accTP + accFP + eps)
        recall = accTP / (len(subdf) + eps)
        F1 = 2 * precision * recall / (precision + recall + eps)
        APs.append(np.trapz(precision, recall))
        recalls.append(recall.to_numpy())
        precisions.append(precision.to_numpy())
        F1s.append(F1.to_numpy())

    test_loss /= num_batches
    #mAP = np.mean(APs)
    mAP = calculate_mAP(APs)
    print(f"\nmAP: {mAP:>0.2f}, Average loss: {test_loss:>8f}, {len(dataloader.dataset)} images tested")
    if wandb_process:
        wandb.log({"val_loss": test_loss, "mAP": mAP})  # Log validation loss and mAP to wandb
    return APs, recalls, precisions, F1s, test_loss


def load_image(image_path, image_resize, grayscale):
    """Loads and preprocesses a single image."""
    try:
        image = Image.open(image_path).convert('RGB')  # Ensure RGB
    except FileNotFoundError:
        print(f"Error: Image not found at path: {image_path}")
        return None

    image_width, image_height = image.size
    operations = [
            transforms.Resize([image_resize, image_resize]),
            transforms.ToTensor(), # Converts to [0, 1]
            transforms.Lambda(lambda x: x.double()) # Convert to double
        ]

    transform = transforms.Compose(operations)
    image_content = transform(image)

    return image_content, image_width, image_height


def predict_and_visualize(model_path, image_path, dataset, conf_threshold=0.5, rescaled=True, savefig = True):

    # Load the model architecture from the config
    model = YOLOv1(architecture, dataset.S, dataset.B, dataset.C, in_channels=dataset.in_channels, image_resize=dataset.image_resize, linear_size=linear_size).to(device).double()
    print("model_path", model_path)
    # Load the state dict into the model
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])

    model.eval()  # Set the model to evaluation mode
    model.to(device)

    image_content, image_width, image_height = load_image(image_path, dataset.image_resize, dataset.grayscale)

    if image_content is None:
        return  # Handle image loading failure

    with torch.no_grad():  # Disable gradient calculation
        image_content = image_content.unsqueeze(0).to(device)  # Add batch dimension
        sample = {'image': image_content[0], 'id': 'new_image', 'is_hflipped': False, 'affine_params': (0., (0, 0), 0., (0., 0.))}

        # Assuming you have defined Sample class
        new_sample = Sample(sample, dataset, compute_label=True, model=model, image_width=image_width, image_height=image_height) # pass image_width and image_height to the sample
        #new_sample.im_width = image_width  # No longer needed
        #new_sample.im_height = image_height # No longer needed
        fig, ax = plt.subplots(1, figsize=(10, 10))
        new_sample.show(rescaled=rescaled, conf_threshold=conf_threshold, ax=ax)
        if savefig:
            plt.savefig(f"{root}/plots/predict_images.png")
        plt.show()
