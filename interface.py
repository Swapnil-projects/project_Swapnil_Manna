# replace MyCustomModel with the name of your model
from model import DeblurModel as TheModel

# change my_descriptively_named_train_function to 
# the function inside train.py that runs the training loop.  
from train import training_loop as the_trainer

# change cryptic_inf_f to the function inside predict.py that
# can be called to generate inference on a single image/batch.
from predict import show_predictions as the_predictor

# change UnicornImgDataset to your custom Dataset class.
from dataset import  DeblurDataset  as TheDataset

# change unicornLoader to your custom dataloader
from dataset import get_dataloaders as the_dataloader

# change batchsize, epochs to whatever your names are for these
#variables inside the config.py file
from config import batchsize as the_batch_size
from config import num_epochs as total_epochs