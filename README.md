# Image Deblurring Project

## Project Overview
This project focuses on deblurring blurred images using deep learning.  
The model is trained to take blurred images as input and produce sharp images as output.

## Dataset
- The dataset used is from Kaggle.  
- [Dataset Link](https://www.kaggle.com/datasets/emrehakanerdemir/face-deblurring-dataset-using-celeba)
- Note: The dataset mentioned in the earlier project proposal is not used. Only the link provided above is correct.

## Model Architecture
- The model follows an Encoder → Bottleneck → Decoder structure.
- It is trained using pairs of blurred and sharp images.
- Loss Function:
  - Initially uses MSE loss.
  - Later, switches to a hybrid loss combining MSE and Gradient Loss.
  - The balance between MSE and Gradient Loss is controlled by a parameter (alpha) defined in the `config` file.

## Project Structure
- Training:
  - The dataset is split into training and validation sets.
  - Cross-validation was not used; the split remains constant. This is because of availability 
  - Training progress is shown using tqdm loading bars.
  - The prediction is done on the validation dataset created.
- Prediction (predict.py):
  - Displays blurred input, model output, and ground truth images sequentially.
  - To display more images, modify the `num_images` variable in `predict.py`.
  - After closing one displayed image, the next one will appear automatically.

## Important Note
- A better model was built, but it exceeds GitHub’s 100MB file size limit, and therefore could not be uploaded.

## Requirements
- Python 3.x
- PyTorch
- tqdm
- (Other dependencies can be installed as needed.)

## Quick Start
```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Predict and visualize results
python predict.py
