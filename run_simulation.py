import subprocess  # Import the subprocess module to run new processes.
import sys  # Import the sys module for system-specific parameters and functions.
import time  # Import the time module for time-related functions.
import signal  # Import the signal module for signal handling.
import os  # Import the os module for operating system-related functions.

def run_server():
    """Start the server process.
    
    Returns:
        subprocess.Popen: The server process.
    """
    return subprocess.Popen([sys.executable, "server.py"])  # Start the server process using the current Python interpreter.
    # Justification: Starting the server process is necessary to initiate the federated learning server.

def run_clients(num_clients):
    """Start multiple client processes.
    
    Args:
        num_clients (int): The number of client processes to start.
    
    Returns:
        list: A list of client processes.
    """
    return [subprocess.Popen([sys.executable, "client.py", str(i)]) for i in range(num_clients)]  # Start multiple client processes.
    # Justification: Starting multiple client processes simulates the federated learning environment with multiple clients.

def terminate_processes(processes):
    """Terminate a list of processes.
    
    Args:
        processes (list): A list of processes to terminate.
    """
    for p in processes:  # Iterate over the list of processes.
        if p.poll() is None:  # Check if the process is still running.
            os.kill(p.pid, signal.SIGTERM)  # Send a termination signal to the process.
            p.wait()  # Wait for the process to terminate.
    # Justification: Terminating processes ensures that all resources are freed and the simulation is properly shut down.

if __name__ == "__main__":
    num_clients = 5  # Set the number of client processes to start.
    
    try:
        server_process = run_server()  # Start the server process.
        time.sleep(5)  # Wait for the server to initialize.
        client_processes = run_clients(num_clients)  # Start the client processes.
        
        server_process.wait()  # Wait for the server process to complete.
        for client_process in client_processes:  # Iterate over the client processes.
            client_process.wait()  # Wait for each client process to complete.
    
    except KeyboardInterrupt:  # Catch a keyboard interrupt (Ctrl+C).
        print("Interrupted. Shutting down...")  # Print a message indicating the interruption.
    finally:
        terminate_processes([server_process] + client_processes)  # Terminate all processes.

    print("Simulation completed.")  # Print a message indicating that the simulation has completed.
    # Justification: Ensuring that all processes are properly terminated helps in managing resources and preventing orphaned processes.