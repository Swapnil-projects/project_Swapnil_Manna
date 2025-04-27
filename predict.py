import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from config import device
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import os
from PIL import Image


# Inference Data loader

class DeblurDataset(Dataset):
    #Dataset function for loading blurred and sharp image pairs.
    def __init__(self, blur_folder, transform=None):
        self.blur_folder = blur_folder
        self.transform = transform
        self.image_filenames = sorted(os.listdir(blur_folder)) 

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        blur_path = os.path.join(self.blur_folder, self.image_filenames[idx])

        blur_image = Image.open(blur_path).convert("RGB")


        if self.transform:
            blur_image = self.transform(blur_image)

        return blur_image
    



def get_dataloaders(blur_folder,  batch_size=8, num_workers=0):
    #Dataloaders function to create train and validation dataloaders.

    # Define transforms ( to resize + normalize)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])


    dataset = DeblurDataset(blur_folder,  transform=transform)

    #DataLoaders
    inference_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)


    print(f"Total images: {len(dataset)} | Train: {len(dataset)} | Val: {len(dataset)}")

    return inference_loader




#Change num_images to display more images, images - (1 image consists of blurred input, model output, and ground truth) come one after another
# When one displayed image is closed the next one opens.

def show_predictions(path, num_images=1):


    from model import DeblurModel



    # Load model. Currently, this loads the model that i have trained.
    model = DeblurModel().to(device)
    checkpoint_path = "checkpoints/final_weights.pth"



    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))



    print("Loaded model checkpoint!, Displaying predictions")

    data = get_dataloaders(path)

    model.eval()
    shown = 0

    with torch.no_grad():
        for blurred in data:
            blurred= blurred.to(device)
            output = model(blurred)

            for i in range(blurred.size(0)):
                if shown >= num_images:
                    return

                #De-normalising from [-1, 1] to [0, 1] for visualization
                b_img = TF.to_pil_image((blurred[i].cpu() + 1) / 2)
                o_img = TF.to_pil_image((output[i].cpu() + 1) / 2)

                fig, axs = plt.subplots(1, 2, figsize=(12, 4))
                axs[0].imshow(b_img)
                axs[0].set_title("Blurred Input")
                axs[0].axis('off')

                axs[1].imshow(o_img)
                axs[1].set_title("Model Output")
                axs[1].axis('off')

                plt.tight_layout()
                plt.show()

                shown += 1

if __name__ == "__main__":

    path = r"data/blur"
    # Show predictions
    show_predictions(path, num_images=1)
