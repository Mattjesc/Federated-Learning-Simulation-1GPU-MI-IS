import flwr as fl  # Import the Flower library for federated learning.
from flwr.server.strategy import FedProx  # Import the FedProx strategy for federated learning.
from flwr.server.app import ServerConfig  # Import the ServerConfig for configuring the server.
import logging  # Import the logging module for logging information.
import numpy as np  # Import NumPy for numerical operations.

# Configure logging to save logs to a file named 'simulation.log' with timestamps and log level set to INFO.
logging.basicConfig(filename='simulation.log', level=logging.INFO, format='%(asctime)s - %(message)s')

def fit_config(rnd: int):
    """Return training configuration dict for each round.
    
    Args:
        rnd (int): The current round number.
    
    Returns:
        dict: Training configuration dictionary.
    """
    config = {
        "learning_rate": 0.001 * (0.95 ** (rnd // 10)),  # Steeper decay of learning rate.
        "batch_size": 128,
        "num_epochs": 10,
    }
    return config  # Return the training configuration.
    # Justification: Adjusting the learning rate dynamically helps in fine-tuning the model.

def evaluate_config(rnd: int):
    """Return evaluation configuration dict for each round.
    
    Args:
        rnd (int): The current round number.
    
    Returns:
        dict: Evaluation configuration dictionary.
    """
    return {"val_steps": 10}  # Return the evaluation configuration.
    # Justification: Configuring the number of validation steps helps in controlling the evaluation process.

def evaluate(server_round, parameters, config):
    """Evaluate global model parameters using an evaluation function.
    
    Args:
        server_round (int): The current server round number.
        parameters (list): List of model parameters as NumPy arrays.
        config (dict): Configuration dictionary.
    
    Returns:
        None: Placeholder for actual evaluation logic.
    """
    # This is a placeholder for actual evaluation logic.
    return None  # Return None as a placeholder.
    # Justification: This function can be extended with actual evaluation logic in the future.

def weighted_average(metrics):
    """Aggregate metrics from multiple clients using a weighted average.
    
    Args:
        metrics (list): List of tuples containing the number of examples and metrics.
    
    Returns:
        dict: Aggregated metrics.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]  # Calculate weighted accuracies.
    examples = [num_examples for num_examples, _ in metrics]  # Get the number of examples.
    return {"accuracy": sum(accuracies) / sum(examples)}  # Return the weighted average accuracy.
    # Justification: Weighted average helps in aggregating metrics from clients with different dataset sizes.

def weighted_aggregation(results):
    """Aggregate model parameters from multiple clients using a weighted average.
    
    Args:
        results (list): List of tuples containing the number of examples and model parameters.
    
    Returns:
        list: Aggregated model parameters.
    """
    parameters_aggregated = [np.zeros_like(param) for param in results[0][1]]  # Initialize aggregated parameters.
    total_examples = sum(num_examples for num_examples, _ in results)  # Calculate the total number of examples.

    for num_examples, parameters in results:
        for i, param in enumerate(parameters):
            parameters_aggregated[i] += param * (num_examples / total_examples)  # Aggregate parameters.

    return parameters_aggregated  # Return the aggregated parameters.
    # Justification: Weighted aggregation ensures that clients with larger datasets have a greater influence on the global model.

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.002):
        """Initialize the EarlyStopping class.
        
        Args:
            patience (int): Number of rounds to wait for improvement before stopping.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience  # Set the patience level.
        self.min_delta = min_delta  # Set the minimum delta for improvement.
        self.counter = 0  # Initialize the counter.
        self.best_score = None  # Initialize the best score.

    def __call__(self, score):
        """Check if the early stopping condition is met.
        
        Args:
            score (float): The current score.
        
        Returns:
            bool: True if early stopping condition is met, False otherwise.
        """
        if self.best_score is None:  # If no best score is set, set the current score as the best score.
            self.best_score = score
            return False

        if score > self.best_score + self.min_delta:  # If the current score is better than the best score, update the best score.
            self.best_score = score
            self.counter = 0  # Reset the counter.
        else:
            self.counter += 1  # Increment the counter.
            if self.counter >= self.patience:  # If the counter exceeds the patience level, trigger early stopping.
                return True
        return False  # Return False if the early stopping condition is not met.
        # Justification: Early stopping helps in preventing overfitting by stopping training when no significant improvement is observed.

if __name__ == "__main__":
    server_config = ServerConfig(num_rounds=100)  # Configure the server with the number of rounds.
    
    strategy = FedProx(
        min_available_clients=5,  # Minimum number of clients required to start training.
        min_fit_clients=5,  # Minimum number of clients required to fit the model.
        min_evaluate_clients=5,  # Minimum number of clients required to evaluate the model.
        on_fit_config_fn=fit_config,  # Function to get fit configuration.
        on_evaluate_config_fn=evaluate_config,  # Function to get evaluation configuration.
        evaluate_fn=evaluate,  # Function to evaluate the global model.
        evaluate_metrics_aggregation_fn=weighted_average,  # Function to aggregate evaluation metrics.
        fit_metrics_aggregation_fn=weighted_aggregation,  # Function to aggregate fit metrics.
        fraction_fit=0.9,  # Fraction of clients to fit in each round.
        fraction_evaluate=0.9,  # Fraction of clients to evaluate in each round.
        proximal_mu=0.1  # Proximal term coefficient.
    )  # Justification: FedProx strategy helps in mitigating the impact of non-IID data.
    
    early_stopping = EarlyStopping(patience=5, min_delta=0.002)  # Initialize the early stopping mechanism.
    
    def server_main():
        # Start Flower server.
        history = fl.server.start_server(
            server_address="[::]:8080",  # Server address.
            config=server_config,  # Server configuration.
            strategy=strategy,  # Federated learning strategy.
        )

        # Check if early stopping condition is met.
        for round, metrics in enumerate(history.metrics_distributed.values()):
            if "accuracy" in metrics:
                accuracy = metrics["accuracy"][1]  # [1] index contains the value.
                if early_stopping(accuracy):  # Check if early stopping condition is met.
                    logging.info(f"Early stopping triggered at round {round}")  # Log the early stopping event.
                    break  # Stop the server if early stopping condition is met.

    server_main()  # Start the server.
    # Justification: Starting the server initiates the federated learning process.