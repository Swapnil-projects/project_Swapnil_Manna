import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from model import DeblurModel  
from dataset import train_dataloader, val_dataloader 
from config import device, checkpoint_dir, num_epochs, learning_rate



print("Dataloaders ready")


# Gradient Loss function to better describe sharpness. Used in combination with mean squared error

def gradient_loss(output, target):
    def gradient(x):
        dx = x[:, :, :-1, :] - x[:, :, 1:, :]
        dy = x[:, :, :, :-1] - x[:, :, :, 1:]
        return dx, dy

    dx_out, dy_out = gradient(output)
    dx_tar, dy_tar = gradient(target)

    loss_dx = F.l1_loss(dx_out, dx_tar)
    loss_dy = F.l1_loss(dy_out, dy_tar)

    return loss_dx + loss_dy

def combined_loss(output, target, alpha=0.8):
    mse = F.mse_loss(output, target)
    grad = gradient_loss(output, target)
    return alpha * mse + (1 - alpha) * grad

# Setup
model = DeblurModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print("Model Initialized")

start_epoch = 0
checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pth")

# Load checkpoint if exists. This was used during training, so that model could be trained from previous checkpoint.
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("Starting training from scratch")

# Training loop. Tqdm used for loading bar
for epoch in range(start_epoch, num_epochs):
    model.train()
    running_loss = 0.0
    print(f"\nEpoch {epoch + 1}/{num_epochs}")

    loop = tqdm(train_dataloader, desc="Training", leave=False)
    for batch_idx, (blurred, sharp) in enumerate(loop):
        blurred, sharp = blurred.to(device), sharp.to(device)

        optimizer.zero_grad()
        output = model(blurred)
        loss = combined_loss(output, sharp)  # Use combined loss
        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        running_loss += batch_loss

        loop.set_postfix(loss=f"{batch_loss:.6f}")

    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch [{epoch + 1}] Training Loss: {avg_loss:.6f}")

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for blurred, sharp in val_dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)
            loss = combined_loss(output, sharp)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_dataloader)
    print(f"Epoch [{epoch + 1}] Validation Loss: {avg_val_loss:.6f}")

    # Save checkpoint (includes epoch number, optimizer - so that we can train again from exactly this point)
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, checkpoint_path)

    # Save checkpoint only with weights for prediction, without optimizer. This model is named so that it doesnt clash with final_weights.pth which is the trained model. 
    # "final_weights.pth" was generated in similar style during training.

    torch.save(model.state_dict(), "checkpoint_without_optimizer.pth")



