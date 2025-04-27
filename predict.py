import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF
from config import device


#Change num_images to display more images, images - (1 image consists of blurred input, model output, and ground truth) come one after another
# When one displayed image is closed the next one opens.

def show_predictions(model, dataloader, num_images=5):
    model.eval()
    shown = 0

    with torch.no_grad():
        for blurred, sharp in dataloader:
            blurred, sharp = blurred.to(device), sharp.to(device)
            output = model(blurred)

            for i in range(blurred.size(0)):
                if shown >= num_images:
                    return

                #De-normalising from [-1, 1] to [0, 1] for visualization
                b_img = TF.to_pil_image((blurred[i].cpu() + 1) / 2)
                o_img = TF.to_pil_image((output[i].cpu() + 1) / 2)
                s_img = TF.to_pil_image((sharp[i].cpu() + 1) / 2)

                fig, axs = plt.subplots(1, 3, figsize=(12, 4))
                axs[0].imshow(b_img)
                axs[0].set_title("Blurred Input")
                axs[0].axis('off')

                axs[1].imshow(o_img)
                axs[1].set_title("Model Output")
                axs[1].axis('off')

                axs[2].imshow(s_img)
                axs[2].set_title("Sharp Target")
                axs[2].axis('off')

                plt.tight_layout()
                plt.show()

                shown += 1

if __name__ == "__main__":
    from model import DeblurModel
    from dataset import val_dataloader

    # Load model. Currently, this loads the model that i have trained.
    model = DeblurModel.to(device)
    checkpoint_path = "checkpoints/final_weights.pth"




    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])



    print("Loaded model checkpoint!, Displaying predictions")

    # Show predictions
    show_predictions(model, val_dataloader, num_images=1)
