# Import necessary libraries and modules
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Grayscale
import argparse
import torch.nn.functional as F
from torch.optim import Adam
import torch
from trainer import train
from model import AnomalyAE
from datetime import datetime


# Define a function to create a data generator for a given data directory
def create_datagen(data_dir, batch_size=8):
    # Define a transformation pipeline to convert images to grayscale and tensors
    transform = Compose([Grayscale(), ToTensor()])
    # Create a dataset from the data directory using the transformation pipeline
    dataset = ImageFolder(data_dir, transform=transform)
    # Create a data loader from the dataset with the specified batch size
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    return dataloader


if __name__ == "__main__":
    # Define a parser to parse command-line arguments
    parser = argparse.ArgumentParser()
    # Add arguments for training and validation directories
    parser.add_argument('--train_dir', required=True, help="Please specify the train directory")
    parser.add_argument('--val_dir', required=True, help="Please specify the test directory")
    # Add arguments for logging interval, number of epochs, and batch sizes
    parser.add_argument("--log_interval", type=int, default=10,
                        help="how many batches to wait before logging training status")
    parser.add_argument('--epochs', type=int, default=25, help="Please specify the number of epochs")
    parser.add_argument('--train_batch_size', type=int, default=4, help="Please specify the batch_size")
    parser.add_argument('--val_batch_size', type=int, default=4, help="Please specify the batch_size")
    # Add arguments for logging directory and weight loading
    parser.add_argument("--log_dir", type=str, default=f'tensorboard_logs_{datetime.now().strftime("%d%m%Y_%H-%M")}',
                        help="log directory for Tensorboard log output")
    parser.add_argument('--load_weight_path', type=str, help="Please specify the weight path that needs to be loaded.")
    # Add an argument to save the network graph
    parser.add_argument('--save_graph', action='store_true', help="Specify this if you want to save the network graph.")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Define the optimizer, device, and loss function
    optimizer = Adam
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss = F.mse_loss

    # Create data loaders for training and validation
    train_loader = create_datagen(args.train_dir, args.train_batch_size)
    val_loader = create_datagen(args.val_dir, args.val_batch_size)

    # Create an instance of the AnomalyAE model
    model = AnomalyAE()

    # Train the model using the trainer function
    train(model, optimizer, loss, train_loader, val_loader, args.log_dir, device, args.epochs, args.log_interval,
          args.load_weight_path, args.save_graph)
