import torch  # Import the PyTorch library for tensor operations and neural networks.
from torchvision import models, datasets, transforms  # Import models, datasets, and data transformations from TorchVision.
from PIL import Image  # Import the PIL library for image processing.
import matplotlib.pyplot as plt  # Import the matplotlib library for plotting.
from captum.attr import Saliency, IntegratedGradients, LayerGradCam  # Import interpretability methods from Captum.
import numpy as np  # Import NumPy for numerical operations.

# Load the saved model
# Determine the device to use (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Set the device to GPU if available, otherwise CPU.

# Load the ResNet18 model without pretrained weights and set the number of output classes to 10
model = models.resnet18(pretrained=False, num_classes=10)  # Load the ResNet18 model without pretrained weights.

# Modify the fully connected layer to match the saved model's structure
model.fc = torch.nn.Sequential(
    torch.nn.Dropout(0.5),  # Apply dropout for regularization.
    torch.nn.Linear(model.fc.in_features, 10)  # Modify the fully connected layer for 10 classes.
)
# Justification: Dropout helps in preventing overfitting.

# Load the saved model parameters from the file 'client_cuda_model.pth'
state_dict = torch.load('client_cuda_model.pth', map_location=device)  # Load the model parameters from the file.

# Remove the unexpected keys from the state dictionary
unexpected_keys = ['fc.1.weight', 'fc.1.bias']  # Define the unexpected keys.
for key in unexpected_keys:  # Iterate over the unexpected keys.
    if key in state_dict:  # Check if the key is in the state dictionary.
        del state_dict[key]  # Delete the key from the state dictionary.
# Justification: Removing unexpected keys ensures that the model is loaded correctly.

# Load the state dictionary, ignoring missing keys
model.load_state_dict(state_dict, strict=False)  # Load the state dictionary into the model.
# Justification: Loading the state dictionary with strict=False allows for missing keys.

# Move the model to the appropriate device (GPU or CPU)
model = model.to(device)  # Move the model to the specified device.
# Justification: Moving the model to the appropriate device ensures efficient computation.

# Set the model to evaluation mode
model.eval()  # Set the model to evaluation mode.
# Justification: Setting the model to evaluation mode ensures that dropout and batch normalization layers behave correctly.

# Define the image transformation pipeline
# Resize the image to 32x32 pixels, convert it to a tensor, and normalize it
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels.
    transforms.ToTensor(),  # Convert the image to a tensor.
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # Normalize the image.
])
# Justification: Data transformations ensure consistent input data distribution.

# Load the CIFAR-10 test dataset
# Download the dataset if it's not already available and apply the defined transformations
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)  # Load the CIFAR-10 test dataset.
# Justification: Loading the test dataset allows for evaluating the model on unseen data.

# Create a DataLoader for the test dataset with a batch size of 1 and no shuffling
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)  # Create a DataLoader for the test dataset.
# Justification: A DataLoader facilitates efficient data loading and batching.

# Generate saliency maps using different methods
# Initialize the Saliency, IntegratedGradients, and LayerGradCam methods from the Captum library
saliency = Saliency(model)  # Initialize the Saliency method.
integrated_gradients = IntegratedGradients(model)  # Initialize the IntegratedGradients method.
layer_grad_cam = LayerGradCam(model, model.layer1[0])  # Initialize the LayerGradCam method using the first layer of the first ResNet block.
# Justification: Captum provides interpretability methods to understand model predictions.

# Visualize the saliency map
def visualize_saliency(image, attributions, title, upscale_factor=8):
    """Visualize the saliency map.
    
    Args:
        image (torch.Tensor): The input image tensor.
        attributions (torch.Tensor): The attributions tensor.
        title (str): The title of the plot.
        upscale_factor (int): The factor by which to upscale the image and attributions.
    """
    # Upscale the image for visualization
    upscaled_image = transforms.Resize((32 * upscale_factor, 32 * upscale_factor))(image)  # Upscale the image.
    upscaled_image = upscaled_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)  # Convert the image to a NumPy array.

    # Process the attributions
    attributions = attributions.squeeze().cpu().detach().numpy()  # Convert the attributions to a NumPy array.
    if len(attributions.shape) == 2:  # Check if the attributions have a channel dimension.
        attributions = attributions[np.newaxis, :, :]  # Add channel dimension.
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())  # Normalize the attributions.

    # Upscale the attributions for visualization
    upscaled_attributions = transforms.Resize((32 * upscale_factor, 32 * upscale_factor))(torch.tensor(attributions).unsqueeze(0)).squeeze().numpy()  # Upscale the attributions.

    # Transpose the upscaled attributions if necessary
    if upscaled_attributions.shape[0] == 3:  # Check if the attributions have a channel dimension.
        upscaled_attributions = upscaled_attributions.transpose(1, 2, 0)  # Transpose the attributions.

    # Create a figure with two subplots
    plt.figure(figsize=(10, 5))  # Create a figure with a specific size.
    plt.subplot(1, 2, 1)  # Create the first subplot.
    plt.imshow(upscaled_image)  # Display the upscaled image.
    plt.title('Original Image')  # Set the title of the subplot.
    plt.axis('off')  # Hide the axis.

    plt.subplot(1, 2, 2)  # Create the second subplot.
    plt.imshow(upscaled_attributions.squeeze(), cmap='gray')  # Display the upscaled attributions.
    plt.title(title)  # Set the title of the subplot.
    plt.axis('off')  # Hide the axis.

    plt.show()  # Display the plot.
    # Justification: Visualizing the saliency map helps in understanding which parts of the image the model is focusing on.

# Overlay saliency map on the original image
def overlay_saliency(image, attributions, upscale_factor=8):
    """Overlay the saliency map on the original image.
    
    Args:
        image (torch.Tensor): The input image tensor.
        attributions (torch.Tensor): The attributions tensor.
        upscale_factor (int): The factor by which to upscale the image and attributions.
    """
    # Upscale the image for visualization
    upscaled_image = transforms.Resize((32 * upscale_factor, 32 * upscale_factor))(image)  # Upscale the image.
    upscaled_image = upscaled_image.squeeze().cpu().detach().numpy().transpose(1, 2, 0)  # Convert the image to a NumPy array.

    # Process the attributions
    attributions = attributions.squeeze().cpu().detach().numpy()  # Convert the attributions to a NumPy array.
    if len(attributions.shape) == 2:  # Check if the attributions have a channel dimension.
        attributions = attributions[np.newaxis, :, :]  # Add channel dimension.
    attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min())  # Normalize the attributions.

    # Upscale the attributions for visualization
    upscaled_attributions = transforms.Resize((32 * upscale_factor, 32 * upscale_factor))(torch.tensor(attributions).unsqueeze(0)).squeeze().numpy()  # Upscale the attributions.

    # Transpose the upscaled attributions if necessary
    if upscaled_attributions.shape[0] == 3:  # Check if the attributions have a channel dimension.
        upscaled_attributions = upscaled_attributions.transpose(1, 2, 0)  # Transpose the attributions.

    # Create a figure and overlay the saliency map on the original image
    plt.figure(figsize=(5, 5))  # Create a figure with a specific size.
    plt.imshow(upscaled_image)  # Display the upscaled image.
    plt.imshow(upscaled_attributions.squeeze(), cmap='gray', alpha=0.5)  # Overlay the upscaled attributions.
    plt.title('Overlay Saliency Map')  # Set the title of the plot.
    plt.axis('off')  # Hide the axis.
    plt.show()  # Display the plot.
    # Justification: Overlaying the saliency map on the original image helps in visualizing the model's focus.

# Generate saliency maps for multiple images
for i, (image, label) in enumerate(testloader):  # Iterate over the test dataset.
    # Move the image tensor to the appropriate device (GPU or CPU)
    input_tensor = image.to(device)  # Move the image tensor to the specified device.

    # Generate saliency maps using different methods
    attributions_saliency = saliency.attribute(input_tensor, target=label)  # Generate saliency map using the Saliency method.
    attributions_integrated_gradients = integrated_gradients.attribute(input_tensor, target=label, n_steps=20)  # Generate saliency map using the IntegratedGradients method.
    attributions_layer_grad_cam = layer_grad_cam.attribute(input_tensor, target=label)  # Generate saliency map using the LayerGradCam method.

    # Upscale GradCAM to input size
    upsampled_attributions_layer_grad_cam = LayerGradCam.interpolate(attributions_layer_grad_cam, input_tensor.shape[2:])  # Upscale the GradCAM attributions.

    # Visualize different saliency maps
    visualize_saliency(image, attributions_saliency, f'Saliency Map - Image {i+1}')  # Visualize the saliency map.
    visualize_saliency(image, attributions_integrated_gradients, f'Integrated Gradients - Image {i+1}')  # Visualize the integrated gradients.
    visualize_saliency(image, upsampled_attributions_layer_grad_cam, f'Layer GradCAM - Image {i+1}')  # Visualize the Layer GradCAM.

    # Overlay saliency map on the original image
    overlay_saliency(image, attributions_saliency)  # Overlay the saliency map on the original image.

    # Break after 5 images for demonstration purposes
    if i >= 4:  # Stop after processing 5 images.
        break
# Justification: Generating and visualizing saliency maps helps in understanding the model's decision-making process.