import torch
import os

#Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Hyperparameters
batch_size = 8
learning_rate = 1e-4
num_epochs = 10

#Image
image_size = (128, 128)

#Base Path
base_dir = os.path.dirname(os.path.abspath(__file__))  # this is script's directory


#Checkpoints directory
checkpoint_dir = os.path.join(base_dir, "checkpoints")

#Loss
loss_alpha = 0.8  # used for loss function defined as : alpha * MSE + (1-alpha) * Gradient Loss

#Other
num_workers = 0
pin_memory = True
