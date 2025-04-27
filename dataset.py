import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms



class DeblurDataset(Dataset):
    #Dataset function for loading blurred and sharp image pairs.
    def __init__(self, blur_folder, sharp_folder, transform=None):
        self.blur_folder = blur_folder
        self.sharp_folder = sharp_folder
        self.transform = transform
        self.image_filenames = sorted(os.listdir(blur_folder)) 

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_folder, self.image_filenames[idx])
        sharp_path = os.path.join(self.sharp_folder, self.image_filenames[idx])

        blur_image = Image.open(blur_path).convert("RGB")
        sharp_image = Image.open(sharp_path).convert("RGB")

        if self.transform:
            blur_image = self.transform(blur_image)
            sharp_image = self.transform(sharp_image)

        return blur_image, sharp_image
    



def get_dataloaders(blur_folder, sharp_folder, batch_size=8, val_ratio=0.2, num_workers=0):
    #Dataloaders function to create train and validation dataloaders.

    # Define transforms ( to resize + normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    #Full dataset
    dataset = DeblurDataset(blur_folder, sharp_folder, transform=transform)

    #Train-val split
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    #DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    print(f"Total images: {len(dataset)} | Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    return train_loader, val_loader

#Creating the dataloaders from functions
BLUR_FOLDER = r"data/blur"
SHARP_FOLDER = r"data/sharp"

train_dataloader, val_dataloader = get_dataloaders(BLUR_FOLDER, SHARP_FOLDER)