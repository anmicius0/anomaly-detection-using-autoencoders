# Import necessary libraries and modules
import os
import pathlib

import torch
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, DiskSaver, global_step_from_engine
from ignite.metrics import ConfusionMatrix
from ignite.metrics import Loss, RunningAverage
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms


# Define a function to create a TensorBoard summary writer
def create_summary_writer(model, train_loader, log_dir, save_graph, device):
    """
    Creates a TensorBoard summary writer

    Args:
        model (pytorch model): the model whose graph needs to be saved
        train_loader (dataloader): the training dataloader
        log_dir (str): the logging directory path
        save_graph (bool): if True, a graph is saved into the TensorBoard log folder
        device (torch.device): torch device object

    Returns:
        writer (SummaryWriter): TensorBoard SummaryWriter object
    """
    writer = SummaryWriter(log_dir=log_dir)
    if save_graph:
        # Get a batch of images from the training loader
        images, labels = next(iter(train_loader))
        images = images.to(device)
        try:
            # Add the model graph to the TensorBoard log
            writer.add_graph(model, images)
        except Exception as e:
            print("Failed to save model graph: {}".format(e))
    return writer


# Define a function to train the model
def train(model, optimizer, loss_fn, train_loader, val_loader,
          log_dir, device, epochs, log_interval,
          load_weight_path=None, save_graph=False):
    """
    Training logic for the wavelet model

    Args:
        model (pytorch model): the model to be trained
        optimizer (torch optim): optimizer to be used
        loss_fn: loss function
        train_loader (dataloader): training dataloader
        val_loader (dataloader): validation dataloader
        log_dir (str): the log directory
        device (torch.device): the device to be used (e.g. CPU or CUDA)
        epochs (int): the number of epochs
        log_interval (int): the log interval for train batch loss

    Keyword Args:
        load_weight_path (str): Model weight path to be loaded (default: None)
        save_graph (bool): whether to save the model graph (default: False)

    Returns:
        None
    """
    # Move the model to the specified device (CPU or CUDA)
    model.to(device)
    if load_weight_path is not None:
        # Load the model weights from the specified path
        model.load_state_dict(torch.load(load_weight_path))

    # Define the optimizer
    optimizer = optimizer(model.parameters())

    # Define the process function for training
    def process_function(engine, batch):
        # Set the model to training mode
        model.train()
        # Zero the gradients
        optimizer.zero_grad()
        # Get the input and label from the batch
        x, _ = batch
        x = x.to(device)
        # Forward pass
        y = model(x)
        # Calculate the loss
        loss = loss_fn(y, x)
        # Backward pass
        loss.backward()
        # Update the model parameters
        optimizer.step()
        # Return the loss
        return loss.item()

    # Define the evaluate function for validation
    def evaluate_function(engine, batch):
        # Set the model to evaluation mode
        model.eval()
        with torch.no_grad():
            # Get the input and label from the batch
            x, _ = batch
            x = x.to(device)
            # Forward pass
            y = model(x)
            # Calculate the loss
            loss = loss_fn(y, x)
            # Return the loss
            return loss.item()

    # Create the trainer and evaluator engines
    trainer = Engine(process_function)
    evaluator = Engine(evaluate_function)

    # Attach running average metrics to the trainer and evaluator
    RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x).attach(evaluator, 'loss')

    # Create a TensorBoard summary writer
    writer = create_summary_writer(model, train_loader, log_dir, save_graph, device)

    # Define a score function for the checkpoint handler
    def score_function(engine):
        return -engine.state.metrics['loss']

    # Create a checkpoint handler to save the model weights
    to_save = {'model': model}
    handler = Checkpoint(
        to_save,
        DiskSaver(os.path.join(log_dir, 'models'), create_dir=True),
        n_saved=5, filename_prefix='best', score_function=score_function,
        score_name="loss",
        global_step_transform=global_step_from_engine(trainer)
    )

    # Attach the checkpoint handler to the evaluator
    evaluator.add_event_handler(Events.COMPLETED, handler)

    # Define a function to log the training loss at each iteration
    @trainer.on(Events.ITERATION_COMPLETED(every=log_interval))
    def log_training_loss(engine):
        print(
            f"Epoch[{engine.state.epoch}] Iteration[{engine.state.iteration}/"
            f"{len(train_loader)}] Loss: {engine.state.output:.3f}"
        )
        writer.add_scalar("training/loss", engine.state.output, engine.state.iteration)

    # Define a function to log the training results at each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        print(
            f"Training Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.3f}"
        )
        writer.add_scalar("training/avg_loss", avg_loss, engine.state.epoch)

    # Define a function to log the validation results at each epoch
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        print(
            f"Validation Results - Epoch: {engine.state.epoch} Avg loss: {avg_loss:.3f}"
        )
        writer.add_scalar("validation/avg_loss", avg_loss, engine.state.epoch)

    # Run the trainer for the specified number of epochs
    trainer.run(train_loader, max_epochs=epochs)

    # Close the TensorBoard summary writer
    writer.close()
