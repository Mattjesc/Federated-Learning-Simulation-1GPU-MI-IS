import torch  # Import the PyTorch library for tensor operations and neural networks.
import torch.nn as nn  # Import the neural network module from PyTorch.
import torch.optim as optim  # Import the optimization module from PyTorch.
from torchvision import datasets, transforms, models  # Import datasets, data transformations, and models from TorchVision.
import flwr as fl  # Import the Flower library for federated learning.
from collections import OrderedDict  # Import OrderedDict for maintaining the order of dictionary items.
import sys  # Import the sys module for system-specific parameters and functions.
import logging  # Import the logging module for logging information.
import gc  # Import the garbage collector module for memory management.

# Configure logging to save logs to a file named 'simulation.log' with timestamps and log level set to INFO.
logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def replace_batchnorm_with_groupnorm(module, num_groups=2):
    """Recursively replace BatchNorm layers with GroupNorm layers in a module.
    
    Args:
        module (nn.Module): The neural network module to modify.
        num_groups (int): The number of groups for GroupNorm.
    
    Returns:
        nn.Module: The modified neural network module.
    """
    for name, child in module.named_children():  # Iterate over the child modules of the given module.
        if isinstance(child, nn.BatchNorm2d):  # Check if the child module is a BatchNorm2d layer.
            num_channels = child.num_features  # Get the number of channels in the BatchNorm layer.
            setattr(module, name, nn.GroupNorm(num_groups, num_channels))  # Replace BatchNorm with GroupNorm.
            # Justification: GroupNorm can be more robust to batch size variations compared to BatchNorm.
        else:
            replace_batchnorm_with_groupnorm(child, num_groups)  # Recursively apply the function to child modules.
    return module  # Return the modified module.

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader, device):
        self.net = net  # The neural network model.
        self.trainloader = trainloader  # DataLoader for training data.
        self.valloader = valloader  # DataLoader for validation data.
        self.device = device  # The device (CPU or GPU) to use for training.
        
        # Use Adam optimizer with a lower learning rate and weight decay to improve convergence.
        self.optimizer = optim.Adam(self.net.parameters(), lr=0.001, weight_decay=1e-4)
        # Justification: Adam optimizer is known for its efficiency and convergence speed.
        
        # Define the loss function as CrossEntropyLoss for classification tasks.
        self.criterion = nn.CrossEntropyLoss()
        # Justification: CrossEntropyLoss is suitable for classification tasks.

    def get_parameters(self, config):
        """Get the current model parameters as a list of NumPy arrays.
        
        Args:
            config (dict): Configuration dictionary.
        
        Returns:
            list: List of model parameters as NumPy arrays.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]  # Convert model parameters to NumPy arrays.
        # Justification: NumPy arrays are used for compatibility with the Flower framework.

    def set_parameters(self, parameters):
        """Set the model parameters from a list of NumPy arrays.
        
        Args:
            parameters (list): List of model parameters as NumPy arrays.
        """
        params_dict = zip(self.net.state_dict().keys(), parameters)  # Pair parameter names with their values.
        state_dict = OrderedDict({k: torch.tensor(v, dtype=torch.float32) for k, v in params_dict})  # Create an OrderedDict of tensors.
        self.net.load_state_dict(state_dict, strict=True)  # Load the state dictionary into the model.
        # Justification: OrderedDict ensures the order of parameters is preserved during loading.

    def fit(self, parameters, config):
        """Train the model for one round of federated learning.
        
        Args:
            parameters (list): List of model parameters as NumPy arrays.
            config (dict): Configuration dictionary.
        
        Returns:
            tuple: Updated model parameters, number of training samples, and additional configurations.
        """
        self.set_parameters(parameters)  # Set the model parameters.
        
        # Train the model and save the weights.
        train(self.net, self.trainloader, self.optimizer, epochs=10, device=self.device)

        return self.get_parameters(config={}), len(self.trainloader.dataset), {}  # Return updated parameters and dataset size.
        # Justification: Returning the dataset size helps in weighted aggregation on the server side.

    def evaluate(self, parameters, config):
        """Evaluate the model on the validation set.
        
        Args:
            parameters (list): List of model parameters as NumPy arrays.
            config (dict): Configuration dictionary.
        
        Returns:
            tuple: Loss, number of validation samples, and evaluation metrics.
        """
        self.set_parameters(parameters)  # Set the model parameters.
        loss, accuracy = test(self.net, self.valloader, device=self.device)  # Evaluate the model.
        logging.info(f"Client {self.device} - Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")  # Log the evaluation results.
        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}  # Return loss, dataset size, and accuracy.
        # Justification: Returning the dataset size helps in weighted aggregation on the server side.

def train(net, trainloader, optimizer, epochs, device):
    """Train the model on the training set.
    
    Args:
        net (nn.Module): The neural network model.
        trainloader (DataLoader): DataLoader for training data.
        optimizer (optim.Optimizer): The optimizer to use for training.
        epochs (int): Number of epochs to train.
        device (torch.device): The device (CPU or GPU) to use for training.
    """
    criterion = nn.CrossEntropyLoss()  # Define the loss function.
    net.train()  # Set the model to training mode.
    
    for epoch in range(epochs):  # Iterate over the number of epochs.
        running_loss = 0.0  # Initialize the running loss.
        correct = 0  # Initialize the number of correct predictions.
        total = 0  # Initialize the total number of samples.
        
        for batch_idx, (inputs, targets) in enumerate(trainloader):  # Iterate over the training data.
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to the specified device.
            optimizer.zero_grad()  # Clear the gradients.
            outputs = net(inputs)  # Forward pass.
            loss = criterion(outputs, targets)  # Compute the loss.
            loss.backward()  # Backward pass.
            optimizer.step()  # Update the model parameters.

            running_loss += loss.item()  # Accumulate the loss.
            _, predicted = outputs.max(1)  # Get the predicted class.
            total += targets.size(0)  # Accumulate the total number of samples.
            correct += predicted.eq(targets).sum().item()  # Accumulate the number of correct predictions.

            # Free up memory to avoid GPU memory overflow.
            del inputs, targets, outputs, loss  # Delete variables to free up memory.
            torch.cuda.empty_cache()  # Empty the GPU cache.
            gc.collect()  # Collect garbage to free up memory.
            # Justification: Freeing up memory helps in managing GPU resources efficiently.

            if batch_idx % 100 == 99:  # Log the loss and accuracy every 100 batches.
                logging.info(f'[Epoch {epoch + 1}, Batch {batch_idx + 1}] Loss: {running_loss / 100:.3f} | Acc: {100.*correct/total:.2f}%')
                running_loss = 0.0  # Reset the running loss.

    # Save the model weights after training.
    torch.save(net.state_dict(), f'client_{device}_model.pth')  # Save the model weights to a file.
    # Justification: Saving the model weights allows for later use or further training.

def test(net, testloader, device):
    """Evaluate the model on the test set.
    
    Args:
        net (nn.Module): The neural network model.
        testloader (DataLoader): DataLoader for test data.
        device (torch.device): The device (CPU or GPU) to use for evaluation.
    
    Returns:
        tuple: Average loss and accuracy.
    """
    criterion = nn.CrossEntropyLoss()  # Define the loss function.
    net.eval()  # Set the model to evaluation mode.
    test_loss = 0  # Initialize the test loss.
    correct = 0  # Initialize the number of correct predictions.
    total = 0  # Initialize the total number of samples.
    
    with torch.no_grad():  # Disable gradient calculation.
        for batch_idx, (inputs, targets) in enumerate(testloader):  # Iterate over the test data.
            inputs, targets = inputs.to(device), targets.to(device)  # Move inputs and targets to the specified device.
            outputs = net(inputs)  # Forward pass.
            loss = criterion(outputs, targets)  # Compute the loss.

            test_loss += loss.item()  # Accumulate the loss.
            _, predicted = outputs.max(1)  # Get the predicted class.
            total += targets.size(0)  # Accumulate the total number of samples.
            correct += predicted.eq(targets).sum().item()  # Accumulate the number of correct predictions.

            # Free up memory to avoid GPU memory overflow.
            del inputs, targets, outputs, loss  # Delete variables to free up memory.
            torch.cuda.empty_cache()  # Empty the GPU cache.
            gc.collect()  # Collect garbage to free up memory.
            # Justification: Freeing up memory helps in managing GPU resources efficiently.

    accuracy = 100. * correct / total  # Calculate the accuracy.
    average_loss = test_loss / (batch_idx + 1)  # Calculate the average loss.
    logging.info(f'Test Loss: {average_loss:.3f} | Acc: {accuracy:.2f}%')  # Log the evaluation results.
    return average_loss, accuracy / 100  # Return average loss and accuracy.
    # Justification: Returning the average loss and accuracy provides a summary of the model's performance.

def main(cid: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Determine the device to use.
    logging.info(f"Using device: {device}")  # Log the device being used.
    
    # Enhanced data augmentation for training to improve model generalization.
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),  # Randomly crop the image.
        transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally.
        transforms.RandomRotation(15),  # Randomly rotate the image.
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # Randomly change the brightness, contrast, and saturation.
        transforms.ToTensor(),  # Convert the image to a tensor.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize the image.
    ])
    # Justification: Data augmentation helps in improving the model's ability to generalize to new data.
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),  # Convert the image to a tensor.
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize the image.
    ])
    # Justification: Normalization ensures consistent input data distribution.
    
    trainset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)  # Load the CIFAR-10 training dataset.
    testset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)  # Load the CIFAR-10 test dataset.
    
    # Use a larger subset of the dataset for debugging to reduce training time.
    subset_size = 10000  # Adjust this number as needed.
    trainset = torch.utils.data.Subset(trainset, range(subset_size))  # Use a subset of the training dataset.
    testset = torch.utils.data.Subset(testset, range(subset_size))  # Use a subset of the test dataset.
    # Justification: Using a subset reduces training time for debugging purposes.
    
    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=128, shuffle=True, num_workers=0, pin_memory=True
    )  # Create a DataLoader for the training dataset.
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=128, shuffle=False, num_workers=0, pin_memory=True
    )  # Create a DataLoader for the test dataset.
    # Justification: DataLoaders facilitate efficient data loading and batching.
    
    # Load the ResNet18 model and modify it to replace BatchNorm with GroupNorm.
    net = models.resnet18(pretrained=False, num_classes=10)  # Load the ResNet18 model.
    net = replace_batchnorm_with_groupnorm(net)  # Replace BatchNorm with GroupNorm.
    net.fc = nn.Sequential(
        nn.Dropout(0.5),  # Apply dropout for regularization.
        nn.Linear(net.fc.in_features, 10)  # Modify the fully connected layer for 10 classes.
    )
    # Justification: Dropout helps in preventing overfitting.
    
    # Move the model to the appropriate device.
    net = net.to(device)  # Move the model to the specified device.
    # Justification: Moving the model to the appropriate device ensures efficient computation.
    
    return FlowerClient(net, trainloader, testloader, device)  # Return the FlowerClient instance.

if __name__ == "__main__":
    cid = sys.argv[1] if len(sys.argv) > 1 else "0"  # Get the client ID from the command line arguments.
    fl.client.start_client(server_address="[::]:8080", client=main(cid).to_client())  # Start the client.
    # Justification: Starting the client initiates the federated learning process.