# Import necessary libraries
# Import necessary libraries
import pandas as pd
import numpy as np
from surprise import accuracy, Dataset, Reader, SVD
from surprise.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit
import pickle
import time
import os
from pathlib import Path
from aim import Run

# Load Data
# Specify the exact .parquet file to load
PARQUET_NAME = '20241027_203929.parquet'

# Load the specified .parquet file into a DataFrame
df = pd.read_parquet(PARQUET_NAME)

# Define a Reader object with rating scale (0-5 based on the Stream data)
reader = Reader(rating_scale=(0, 5))

# Define hyperparameters for each model
models_hyperparameters = {
    'Model 1': {'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02, 'biased': True},
    'Model 2': {'n_factors': 50, 'n_epochs': 50, 'lr_all': 0.01, 'reg_all': 0.02, 'biased': True},
    'Model 3': {'n_factors': 104, 'n_epochs': 23, 'lr_all': 0.0013, 'reg_all': 0.0987, 'biased': True}
}

# Sample parameter function
def sample_params(range_dict):
    factor_low, factor_high = range_dict['n_factors']
    epochs_low, epochs_high = range_dict['n_epochs']
    lr_low, lr_high = range_dict['lr_all']
    reg_low, reg_high = range_dict['reg_all']
    return {
        "n_factors": np.random.randint(low=factor_low, high=factor_high),
        "n_epochs": np.random.randint(low=epochs_low, high=epochs_high),
        "lr_all": np.random.uniform(low=lr_low, high=lr_high),
        "reg_all": np.random.uniform(low=reg_low, high=reg_high),
    }

# Training function
def train(h_dict, trainset):
    model = SVD(**h_dict)
    start_time = time.time()
    model.fit(trainset)
    training_time = time.time() - start_time
    print(f"SVD Model Training Time: {training_time:.2f} seconds")
    return model, training_time

# Prediction function
def predict(model, user_id, df):
    unique_items = df['item'].unique()
    start_time = time.time()
    predictions = [(item, model.predict(user_id, item).est) for item in unique_items]
    inference_time = time.time() - start_time
    print(f"SVD Inference Time: {inference_time}")
    return predictions, inference_time

# Function to log metrics and parameters with Aim
def log_metrics(run, model_name, hyperparameters, training_time, model_size, rmse, inference_time):
    run[f'{model_name}/hyperparameters'] = hyperparameters
    run.track(training_time, name='Training Time', context={'model': model_name})
    run.track(model_size, name='Model Size (MB)', context={'model': model_name})
    run.track(rmse, name='RMSE', context={'model': model_name})
    run.track(inference_time, name='Inference Time', context={'model': model_name})

# Load dataset into Surprise format for training and testing
trainset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader).build_full_trainset()
testset = Dataset.load_from_df(df[['user', 'item', 'rating']], reader).build_full_trainset().build_testset()

# Loop through each model's hyperparameters
for model_name, hyperparameters in models_hyperparameters.items():
    # Initialize a new Aim run for each model
    run = Run()  # This creates a new, separate run for each model
    run["model_name"] = model_name  # Log the model name for easy identification
    
    print(f"{model_name} Training and Evaluation:")

    # Log hyperparameters to Aim
    run[f'{model_name}/hyperparameters'] = hyperparameters

    # Train model
    model, training_time = train(hyperparameters, trainset)

    # Save and log model artifact
    model_path = f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    model_size = os.path.getsize(model_path) / (1024 * 1024)  # Model size in MB
    run.track(model_size, name='Model Size (MB)', context={'model': model_name})

    # Log training time
    run.track(training_time, name='Training Time', context={'model': model_name})

    # Evaluate Model and log RMSE
    test_predictions = model.test(testset)
    rmse = accuracy.rmse(test_predictions)
    run.track(rmse, name='RMSE', context={'model': model_name})

    # Make predictions and log inference time
    user_id = "127764"  # Replace with actual user ID as needed
    _, inference_time = predict(model, user_id, df)
    run.track(inference_time, name='Inference Time', context={'model': model_name})

    print(f"{model_name} RMSE: {rmse}")
    print(f"{model_name} Model Size: {model_size:.2f} MB")
    print(f"{model_name} Training Time: {training_time:.2f} seconds")

    # Close the run after logging all metrics
    run.close()

