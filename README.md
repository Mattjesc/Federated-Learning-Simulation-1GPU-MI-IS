# Federated Learning Simulation on a Single GPU with Model Interpretability and Interactive Visualization

## Disclaimer

This project is configured on compute-restrained hardware, specifically a single GPU. The model training results may vary significantly depending on the system configuration, including but not limited to GPU type, CPU, RAM, and other hardware specifications. Additionally, external factors such as network conditions, software environment, and other runtime variables can influence the outcomes.

In this specific implementation, we simulate federated learning with 5 clients over 100 rounds, using the ResNet model and the CIFAR-10 dataset. The core idea of this project is to provide a flexible framework where users can choose any model, dataset, and training hyperparameters according to their specific needs and preferences. The provided scripts and code are designed to be adaptable and customizable for various use cases.

## Introduction

This project simulates federated learning on a single GPU, focusing on model interpretability and interactive visualization. Federated learning is a decentralized machine learning approach that allows multiple clients to collaboratively train a model without sharing their data. This simulation aims to provide insights into the model's behavior and performance through saliency maps and interactive dashboards.

## Project Overview

### Key Features

- **Federated Learning Simulation**: Simulates the federated learning process on a single GPU.
- **Model Interpretability**: Uses techniques like Saliency, Integrated Gradients, and Layer GradCAM to interpret the model.
- **Interactive Visualization**: Provides interactive visualizations using Streamlit for better understanding of model behavior.
- **Performance Analysis**: Logs and visualizes performance metrics such as loss and accuracy.

### Components

- **Server**: Manages the federated learning process, aggregating model updates from clients.
- **Clients**: Simulate individual clients with local datasets, training models, and sending updates to the server.
- **Model Interpretation**: Generates saliency maps and other interpretability visualizations.
- **Interactive Dashboard**: Uses Streamlit to create an interactive dashboard for visualizing model interpretability and performance metrics.

## Installation

To set up the project, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Mattjesc/Federated-Learning-Simulation-1GPU-MI-IS.git
   cd Federated-Learning-Simulation-1GPU-MI-IS
   ```

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Running the Simulation

1. **Run the Simulation Script**:
   ```bash
   python run_simulation.py
   ```
   This script will run both the server and multiple clients, simulating the federated learning process. After the simulation completes, it will save the trained model as `client_cuda_model.pth`.

### Visualizing Loss and Accuracy

1. **Visualize Loss and Accuracy Over Rounds**:
   ```bash
   python visualize.py
   ```
   This script will parse the log file generated during the simulation and visualize the loss and accuracy over the rounds.

### Interpreting the Model

1. **Generate Saliency Maps**:
   ```bash
   python interpret_model.py
   ```
   This script will load the saved model (`client_cuda_model.pth`) and generate saliency maps for a few test images from the CIFAR-10 dataset. The saliency maps will be displayed using Matplotlib.

### Interactive Visualization

1. **Start the Streamlit App**:
   ```bash
   streamlit run streamlit.py
   ```
   This will start the Streamlit app, which provides an interactive dashboard for visualizing model interpretability. You can upload an image using the file uploader in the Streamlit app to visualize the saliency maps for the uploaded image.

## Code Structure

- **`run_simulation.py`**: Orchestrates the simulation by running the server and multiple clients. It saves the trained model as `client_cuda_model.pth`.
- **`visualize.py`**: Parses the log file generated during the simulation and visualizes the loss and accuracy over the rounds.
- **`interpret_model.py`**: Loads the saved model (`client_cuda_model.pth`) and generates saliency maps for test images.
- **`streamlit.py`**: Provides an interactive dashboard for visualizing model interpretability using Streamlit.
- **`requirements.txt`**: Lists all the dependencies required to run the project.

## Dependencies

The project relies on the following key dependencies:

- **torch, torchaudio, torchvision**: Core PyTorch libraries for tensor operations, audio processing, and computer vision.
- **captum**: For model interpretability techniques like Saliency, Integrated Gradients, and Layer GradCAM.
- **streamlit**: For creating interactive dashboards.
- **matplotlib**: For plotting and visualization.
- **numpy**: For numerical operations.
- **Pillow**: For image processing.
- **requests**: For making HTTP requests (used by Streamlit).
- **protobuf**: For protocol buffer serialization (used by FLWR).
- **grpcio**: For gRPC communication (used by FLWR).
- **flwr**: For federated learning simulation.

For a complete list of dependencies, refer to `requirements.txt`.