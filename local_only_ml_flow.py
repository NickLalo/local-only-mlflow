"""
Basic example of how to use MLflow to log metrics and hyperparameters to a local MLflow server.
After running, you can view the information logged to the MLflow server by running the following command in the terminal:
    mlflow ui --backend-store-uri mlflow_data_storage
    
This will start a locally hosted server that you can view in your browser by navigating to http://127.0.0.1:5000
"""

import random
import time
import mlflow  # pip install mlflow
from tqdm import tqdm


def generate_sample_space():
    """
    Generates a logarithmic like distribution of numbers between 5 and 0.0001, 
    with a larger proportion of numbers between 0.1 and 0.0001.
    
    Returns:
        sample_space (list): List of generated numbers.
    """
    sample_space = []
    print(f"Generating a sample space to help simulate fake data and hyperparameters...")

    for i in range(1000):
        number = random.uniform(0.0001, 5)
        sample_space.append(number)

    for i in range(500):
        number = random.uniform(0.0001, 1)
        sample_space.append(number)

    for i in range(250):
        number = random.uniform(0.0001, 0.5)
        sample_space.append(number)

    for i in range(125):
        number = random.uniform(0.0001, 0.25)
        sample_space.append(number)

    for i in range(62):
        number = random.uniform(0.0001, 0.125)
        sample_space.append(number)

    for i in range(31):
        number = random.uniform(0.0001, 0.0625)
        sample_space.append(number)

    for i in range(15):
        number = random.uniform(0.0001, 0.03125)
        sample_space.append(number)

    return sample_space


def run_experiment(sample_space, mlflow_configs, number_of_runs=10):
    """
    Simulates training and logs metrics and hyperparameters to the MLflow server.

    Args:
        sample_space (list): A list of sample values for selecting final validation losses.
        mlflow_configs (dict): Dictionary containing MLflow configuration settings.
        number_of_runs (int): Number of runs to simulate.
    """
    start_time = time.time()

    # Set the MLflow tracking URI and experiment name
    mlflow.set_tracking_uri(f"{mlflow_configs['mlflow_storage_path']}")
    mlflow.set_experiment(mlflow_configs["experiment_name"])

    print(f"Starting up {number_of_runs} runs...")
    # Log metrics from runs to the MLflow server
    for run_num in tqdm(range(number_of_runs)):

        # Start a new run to be tracked in MLflow
        mlflow.start_run(run_name=str(run_num))

        # Set the hyperparameters for the run
        a = random.choice([1, 2, 4, 8, 16])
        b = random.choice([0.1, 0.01, 0.001, 0.0001])
        c = random.choice([32, 64, 128, 256, 512])

        # Simulate training over several epochs
        epochs = 20
        for num in range(epochs):
            # Simulate decreasing train and validation loss
            train_loss = random.randint(0, epochs - num)
            val_loss = random.randint(0, epochs - num)
            epoch = num

            # Log the train and val loss
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)

        # Pick a random final validation loss weighted towards the lower end of the sample space
        final_val_loss = random.choice(sample_space)

        # Modify params based on the final val loss
        a = a * final_val_loss
        b = b * final_val_loss
        c = c * final_val_loss

        # Log hyperparameters and final validation loss
        mlflow.log_param("a", a)
        mlflow.log_param("b", b)
        mlflow.log_param("c", c)
        mlflow.log_metric("final_val_loss", final_val_loss)

        mlflow.end_run()

    print(f"Time to log {number_of_runs} runs: {time.time() - start_time} seconds")
    return


if __name__ == "__main__":
    # MLflow configurations (these can be saved to a config file)
    mlflow_configs = {
        "mlflow_storage_path": "mlflow_data_storage",
        "experiment_name": "fake_experiment",
    }

    # Generate the sample space
    sample_space = generate_sample_space()

    # Run the experiment with the generated sample space
    run_experiment(sample_space, mlflow_configs, number_of_runs=10)
