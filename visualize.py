import matplotlib.pyplot as plt  # Import the matplotlib library for plotting.
import re  # Import the re module for regular expressions.

def parse_logs(log_file):
    """Parse the log file to extract round-wise loss and accuracy metrics.
    
    Args:
        log_file (str): Path to the log file.
    
    Returns:
        tuple: Lists of rounds, average losses, and average accuracies.
    """
    rounds = []  # Initialize a list to store round numbers.
    accuracies = []  # Initialize a list to store average accuracies.
    losses = []  # Initialize a list to store average losses.
    round_counter = 1  # Initialize the round counter starting at 1.
    round_losses = []  # Initialize a list to store losses for the current round.
    round_accuracies = []  # Initialize a list to store accuracies for the current round.
    client_results = 0  # Initialize a counter to track the number of clients processed for the current round.

    with open(log_file, 'r') as file:  # Open the log file for reading.
        for line in file:  # Iterate over each line in the log file.
            # Match the client metrics log line using a regular expression.
            match = re.search(r'Client (\w+) - Loss: (\d+\.\d+), Accuracy: (\d+\.\d+)', line)
            if match:  # If a match is found.
                round_losses.append(float(match.group(2)))  # Append the loss to the round_losses list.
                round_accuracies.append(float(match.group(3)))  # Append the accuracy to the round_accuracies list.
                client_results += 1  # Increment the client results counter.

                # Once results for all clients in the round are logged, calculate averages.
                if client_results == 5:  # Assuming 5 clients per round.
                    rounds.append(round_counter)  # Append the current round number to the rounds list.
                    losses.append(sum(round_losses) / len(round_losses))  # Calculate and append the average loss.
                    accuracies.append(sum(round_accuracies) / len(round_accuracies))  # Calculate and append the average accuracy.
                    round_counter += 1  # Increment the round counter.
                    client_results = 0  # Reset the client results counter for the next round.
                    round_losses = []  # Clear the round_losses list for the next round.
                    round_accuracies = []  # Clear the round_accuracies list for the next round.

    return rounds, losses, accuracies  # Return the lists of rounds, average losses, and average accuracies.
    # Justification: Parsing the log file helps in extracting and aggregating metrics for visualization.

def plot_metrics(rounds, losses, accuracies):
    """Plot the round-wise loss and accuracy metrics.
    
    Args:
        rounds (list): List of round numbers.
        losses (list): List of average losses per round.
        accuracies (list): List of average accuracies per round.
    """
    plt.figure(figsize=(12, 6))  # Create a figure with a specific size.

    plt.subplot(1, 2, 1)  # Create the first subplot.
    plt.plot(rounds, losses, marker='o')  # Plot the loss over rounds with markers.
    plt.title('Loss Over Rounds')  # Set the title of the subplot.
    plt.xlabel('Round')  # Set the x-axis label.
    plt.ylabel('Loss')  # Set the y-axis label.

    plt.subplot(1, 2, 2)  # Create the second subplot.
    plt.plot(rounds, [acc * 100 for acc in accuracies], marker='o')  # Plot the accuracy over rounds with markers.
    plt.title('Accuracy Over Rounds')  # Set the title of the subplot.
    plt.xlabel('Round')  # Set the x-axis label.
    plt.ylabel('Accuracy (%)')  # Set the y-axis label to indicate percentage.

    plt.tight_layout()  # Adjust the layout to prevent overlap.
    plt.show()  # Display the plot.
    # Justification: Plotting the metrics helps in visualizing the performance of the federated learning process.

if __name__ == "__main__":
    log_file = "simulation.log"  # Update with your log file path.
    rounds, losses, accuracies = parse_logs(log_file)  # Parse the log file to extract metrics.
    plot_metrics(rounds, losses, accuracies)  # Plot the extracted metrics.
    # Justification: Running the script as the main module initiates the parsing and plotting of metrics.