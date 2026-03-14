"""
Standard 2-qubit training and testing code for submission.
Data cleaning using IQR is applied during the testing phase.
Measurement basis coverage generation.
"""
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from model import MyNet_MLP_Improved_GELU_2qubit
from utiles import *
import time
import os
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import multiprocessing as mp
from functools import partial
import torch.nn as nn

np.random.seed(42)

# Define model paths
nProjAll = 19
pom = np.array(range(1, nProjAll))[::-1]
dictModel = {str(k): f"./model/2qubit_model_cover/tmp/50ModelPauliConcProjR0v7{2 * k}" for k in pom}
dictBestModel = {str(k): f"./model/2qubit_model_cover/50bestModelPauliConcProjR0v7{2 * k}" for k in pom}

dim = 4
noStates = 500000
noTrain = 300000
batch_size = 64
epochs = 50
lr = 0.001
Q = herbasis(dim)
GAll = gellmann(Q, dim) * np.sqrt(dim)
G = GAll[1:]

# Training function
def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, scheduler, device, writer, kk,
                write_log=False):
    """
    Train the model for a specified number of epochs and return the best model state dict.
    
    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        epochs: Number of training epochs.
        criterion: Loss function.
        optimizer: Optimization algorithm.
        scheduler: Learning rate scheduler.
        device: Compute device (CPU/GPU).
        writer: TensorBoard writer for logging.
        kk: Model identifier for logging.
        write_log: Flag to enable TensorBoard logging.
    
    Returns:
        best_model: State dict of the best model based on validation loss.
        best_mae: Best validation MAE achieved.
    """
    best_model = None
    best_mae = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                val_outputs = model(x_val)
                val_loss += criterion(val_outputs, y_val).item()

        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        if write_log:
            writer.add_scalar(f'Model_{kk}/Train/Loss', avg_train_loss, epoch)
            writer.add_scalar(f'Model_{kk}/Val/Loss', avg_val_loss, epoch)

        if avg_val_loss < best_mae:
            best_mae = avg_val_loss
            best_model = model.state_dict()

    return best_model, best_mae


# Single model training function
def train_single_model(nProj, device_id, dictModel, dictBestModel, gpu_semaphore):
    """
    Train a single model for a given nProj on a specified GPU device.
    
    This function handles initial model selection (best of 5 runs) and final fine-tuning.
    
    Args:
        nProj: Number of projections.
        device_id: GPU device ID.
        dictModel: Dictionary of temporary model paths.
        dictBestModel: Dictionary of final model paths.
        gpu_semaphore: Semaphore to control GPU concurrency.
    """
    with gpu_semaphore[device_id]:  # Control concurrent tasks per GPU
        torch.cuda.set_device(device_id)
        device = torch.device(f'cuda:{device_id}')
        print(f'nProj = {nProj}, running on device: {torch.cuda.current_device()}')

        writer = SummaryWriter(log_dir=f'./runs/tmp')

        # Load data
        x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict = localdataload8_cover(nProj, data_dir='./data/two_qubit_data')
        # print(f'nProj = {nProj}, x_train shape: {x_train.shape}, y_train shape: {y_train.shape}')

        # Subset data for initial model selection
        x_train, x_val, y_train, y_val = x_train[200:3000, :], x_val[200:3000, :], y_train[200:3000, :], y_val[200:3000, :]
        x_train, x_val = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 1, -1), torch.tensor(x_val,
                                                                                                                dtype=torch.float32).view(
            x_val.shape[0], 1, -1)
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        best_model_path = dictModel[str(int(nProj / 2))] + '.pth'

        # Ensure directory exists
        os.makedirs(os.path.dirname(best_model_path), exist_ok=True)

        best_model = None
        best_mae = float('inf')

        # Train 5 initial models and select the best
        for _ in range(5):
            model = MyNet_MLP_Improved_GELU_2qubit(nProj).to(device)
            criterion = nn.L1Loss()
            optimizer = optim.Adam(model.parameters(), lr=lr)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
            current_model, current_mae = train_model(model, train_loader, val_loader, epochs, criterion, optimizer,
                                                     scheduler, device, writer, int(nProj / 2), write_log=False)

            if current_mae < best_mae:
                best_mae = current_mae
                best_model = current_model
                torch.save(best_model, best_model_path)

        print(f'nProj = {nProj}, Best MAE: {best_mae}')

        # Final training with full dataset
        x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict = localdataload8_cover(nProj, data_dir='./data/two_qubit_data')
        # Shuffle and split training data
        indices = np.random.permutation(x_train.shape[0])
        x_train = x_train[indices]
        y_train = y_train[indices]
        x_train, x_val, y_train, y_val = x_train[0:noTrain, :], x_train[noTrain:noTrain + int(noTrain / 4), :], y_train[
                                                                                                                0:noTrain,
                                                                                                                :], y_train[
                                                                                                                    noTrain:noTrain + int(
                                                                                                                        noTrain / 4),
                                                                                                                    :]
        x_train, x_val = torch.tensor(x_train, dtype=torch.float32).view(x_train.shape[0], 1, -1), torch.tensor(x_val,
                                                                                                                dtype=torch.float32).view(
            x_val.shape[0], 1, -1)
        y_train, y_val = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        train_dataset = TensorDataset(x_train, y_train)
        val_dataset = TensorDataset(x_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        best_model_path_final = dictBestModel[str(int(nProj / 2))] + '.pth'

        # Ensure directory exists
        os.makedirs(os.path.dirname(best_model_path_final), exist_ok=True)

        model = MyNet_MLP_Improved_GELU_2qubit(nProj).to(device)
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        criterion = nn.L1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        writer_final = SummaryWriter(log_dir=f'./runs/two_qubit_cover_runs/nProj_{nProj}_final')

        final_best_model, final_best_mae = train_model(model, train_loader, val_loader, epochs, criterion, optimizer,
                                                       scheduler, device, writer_final, int(nProj / 2), write_log=True)

        torch.save(final_best_model, best_model_path_final)

        print(f'nProj = {nProj}, Final Validation MAE: {final_best_mae}')

        writer_final.close()


def N2_test_only_filter_CI_main(saved_model_path="./model/2qubit_model_cover/"):
    """
    Main function for testing pre-trained models with IQR-based data filtering and Bootstrap confidence intervals.
    
    Loads saved models, filters test data using IQR on errors, computes metrics including RMSE with Bootstrap CI,
    and saves results to CSV.
    
    Args:
        saved_model_path: Base path for saved model directories.
    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    nProjAll = 19
    pom = list(range(1, nProjAll))
    nProj_list = [2 * k for k in pom]
    nProj_list.sort()  # Ascending: 2,4,...,36

    dictBestModel = {str(k): f"{saved_model_path}/50bestModelPauliConcProjR0v7{2 * k}" for k in pom}

    results = []
    for nProj in nProj_list:
        print(f'Testing nProj = {nProj}')
        x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict = localdataload8_cover(nProj, data_dir='./data/two_qubit_data')

        # Filter outliers using IQR on mean absolute errors per sample
        maxlike_errors = np.mean(np.abs(maxlike_predict - y_test), axis=1)  # Shape: (N,)
        q1_maxlike = np.percentile(maxlike_errors, 25)
        q3_maxlike = np.percentile(maxlike_errors, 75)
        iqr_maxlike = q3_maxlike - q1_maxlike
        lower_maxlike = q1_maxlike - 1.5 * iqr_maxlike
        upper_maxlike = q3_maxlike + 1.5 * iqr_maxlike
        good_maxlike = (maxlike_errors >= lower_maxlike) & (maxlike_errors <= upper_maxlike)  # Shape: (N,)

        MLME_errors = np.mean(np.abs(MLME_predict - y_test), axis=1)  # Shape: (N,)
        q1_MLME = np.percentile(MLME_errors, 25)
        q3_MLME = np.percentile(MLME_errors, 75)
        iqr_MLME = q3_MLME - q1_MLME
        lower_MLME = q1_MLME - 1.5 * iqr_MLME
        upper_MLME = q3_MLME + 1.5 * iqr_MLME
        good_MLME = (MLME_errors >= lower_MLME) & (MLME_errors <= upper_MLME)  # Shape: (N,)

        # Use intersection for common good samples
        good_indices = good_maxlike & good_MLME  # Shape: (N,)

        # Apply filter to all data
        x_test = x_test[good_indices]
        y_test = y_test[good_indices]
        maxlike_predict = maxlike_predict[good_indices]
        MLME_predict = MLME_predict[good_indices]

        print(f'After filtering, remaining samples: {len(y_test)}')

        x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 1, -1).to(device)
        y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
        maxlike_predict = torch.tensor(maxlike_predict, dtype=torch.float32).to(device)
        MLME_predict = torch.tensor(MLME_predict, dtype=torch.float32).to(device)

        base_path = dictBestModel[str(int(nProj / 2))]
        model_path = base_path + '.pth'  # Assume one file per nProj

        model = MyNet_MLP_Improved_GELU_2qubit(nProj).to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        with torch.no_grad():
            model_predict = model(x_test)

        model_accuracy = calculate_negative_accuracy(y_test, model_predict)
        maxlike_accuracy = calculate_negative_accuracy(y_test, maxlike_predict)
        MLME_accuracy = calculate_negative_accuracy(y_test, MLME_predict)

        mae_criterion = nn.L1Loss(reduction='mean')
        model_mae = mae_criterion(model_predict, y_test).item()
        maxlike_mae = mae_criterion(maxlike_predict, y_test).item()
        MLME_mae = mae_criterion(MLME_predict, y_test).item()

        # Use Bootstrap CI to replace original error bars
        model_rmse_boot, model_ci_low, model_ci_high = bootstrap_rmse(y_test, model_predict)
        maxlike_rmse_boot, maxlike_ci_low, maxlike_ci_high = bootstrap_rmse(y_test, maxlike_predict)
        MLME_rmse_boot, MLME_ci_low, MLME_ci_high = bootstrap_rmse(y_test, MLME_predict)

        # Use bootstrap mean to replace original RMSE (optional; for consistency with CI)
        model_rmse = model_rmse_boot
        maxlike_rmse = maxlike_rmse_boot
        MLME_rmse = MLME_rmse_boot

        # Original-style RMSE and error bars after IQR filtering
        rmse_criterion = nn.MSELoss(reduction='mean')
        model_rmse_original = torch.sqrt(rmse_criterion(model_predict, y_test)).item()
        maxlike_rmse_original = torch.sqrt(rmse_criterion(maxlike_predict, y_test)).item()
        MLME_rmse_original = torch.sqrt(rmse_criterion(MLME_predict, y_test)).item()

        mse_criterion = nn.MSELoss(reduction='none')
        model_mse = mse_criterion(model_predict, y_test).mean(dim=1)
        maxlike_mse = mse_criterion(maxlike_predict, y_test).mean(dim=1)
        MLME_mse = mse_criterion(MLME_predict, y_test).mean(dim=1)

        model_rmse_err_bar = torch.std(torch.sqrt(model_mse)) / np.sqrt(model_mse.shape[0])
        maxlike_rmse_err_bar = torch.std(torch.sqrt(maxlike_mse)) / np.sqrt(maxlike_mse.shape[0])
        MLME_rmse_err_bar = torch.std(torch.sqrt(MLME_mse)) / np.sqrt(MLME_mse.shape[0])

        print(f'nProj = {nProj}, model_rmse: {model_rmse}, model_ci: [{model_ci_low}, {model_ci_high}]')

        results.append({
            'nProj': nProj,
            'model_mae': model_mae,
            'maxlike_mae': maxlike_mae,
            'MLME_mae': MLME_mae,
            'model_rmse': model_rmse,
            'maxlike_rmse': maxlike_rmse,
            'MLME_rmse': MLME_rmse,
            'model_rmse_original': model_rmse_original,
            'maxlike_rmse_original': maxlike_rmse_original,
            'MLME_rmse_original': MLME_rmse_original,
            'model_rmse_err_bar': model_rmse_err_bar.item(),
            'maxlike_rmse_err_bar': maxlike_rmse_err_bar.item(),
            'MLME_rmse_err_bar': MLME_rmse_err_bar.item(),
            'model_rmse_ci_low': model_ci_low,
            'model_rmse_ci_high': model_ci_high,
            'maxlike_rmse_ci_low': maxlike_ci_low,
            'maxlike_rmse_ci_high': maxlike_ci_high,
            'MLME_rmse_ci_low': MLME_ci_low,
            'MLME_rmse_ci_high': MLME_ci_high,
            'model_accuracy': model_accuracy,
            'maxlike_accuracy': maxlike_accuracy,
            'MLME_accuracy': MLME_accuracy
        })
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by='nProj')
    results_df.to_csv('test_results_with_bootstrap_ci_and_err_bar.csv', index=False)
    print("Results have been saved to 'test_results_with_bootstrap_ci_and_err_bar.csv'")


# Main function for training + testing
def main():
    """
    Main entry point for multi-GPU parallel training of models.
    Distributes tasks across available GPUs and trains models for different nProj values.
    """
    if not torch.cuda.is_available():
        print("CUDA not available, exiting.")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Number of available GPUs: {num_gpus}")
    if num_gpus < 2:
        print("Need at least two GPUs, exiting.")
        return

    nProj_list = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]
    nProj_list.sort(reverse=False)  # Sort nProj in ascending order

    # Alternate task assignment for load balancing
    gpu_tasks = [[], []]
    for i, nProj in enumerate(nProj_list):
        gpu_id = i % 2
        gpu_tasks[gpu_id].append(nProj)

    print("GPU 0 tasks:", gpu_tasks[0])
    print("GPU 1 tasks:", gpu_tasks[1])

    # Create task list with alternating assignment for initial 6 tasks
    tasks = []
    for i in range(max(len(gpu_tasks[0]), len(gpu_tasks[1]))):
        if i < len(gpu_tasks[0]):
            tasks.append((gpu_tasks[0][i], 0, dictModel, dictBestModel))
        if i < len(gpu_tasks[1]):
            tasks.append((gpu_tasks[1][i], 1, dictModel, dictBestModel))

    print("Initial task allocation:", [(nProj, device_id) for nProj, device_id, _, _ in tasks[:6]])

    # Use Manager to create semaphores limiting concurrent tasks per GPU
    manager = mp.Manager()
    gpu_semaphore = [manager.Semaphore(3), manager.Semaphore(3)]  # Max 3 tasks per GPU

    # Use process pool with total parallel processes limited to 6
    results = []
    with mp.Pool(processes=6) as pool:
        pool.starmap(partial(train_single_model, gpu_semaphore=gpu_semaphore), tasks)


if __name__ == '__main__':
    # Training + testing
    mp.set_start_method('spawn', force=True)
    main()

    # Testing only
    N2_test_only_filter_CI_main(saved_model_path="./model/2qubit_model_cover/")
