import os
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from model import MyNet_MLP_Improved_GELU_2qubit
from utiles import localdataload_exp_cover
import pandas as pd
from scipy import stats

# Configuration
data_dir = "./data/exp_data"
nProjAll = 19
pom = np.array(range(1, nProjAll))[::-1]
dictBestModel = {str(k): "50bestModelPauliConcProjR0v7" + str(2 * k) for k in pom}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
noStates = 5000  # Assuming same number of states as original

# Initialize results list
results = []

# Loop over nProj values
for nProj in tqdm([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]):
    print(f'nProj = {nProj}')

    # Load test data from new dataset
    x_test, y_test, maxlike_predict, MLME_predict = localdataload_exp_cover(nProj, noStates)

    # Convert test data to tensors
    x_test = torch.tensor(x_test, dtype=torch.float32).view(x_test.shape[0], 1, -1).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).to(device)
    maxlike_predict = torch.tensor(maxlike_predict, dtype=torch.float32).to(device)
    MLME_predict = torch.tensor(MLME_predict, dtype=torch.float32).to(device)

    # Load pre-trained model
    best_model_path = './model/2qubit_model_cover/' + dictBestModel[str(int(nProj / 2))] + '.pth'
    model = MyNet_MLP_Improved_GELU_2qubit(nProj).to(device)
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    model.eval()

    # Make predictions
    with torch.no_grad():
        model_predict = model(x_test)

    os.makedirs(f'./data/real_data/output/{nProj}/', exist_ok=True)
    np.savez(f'./data/real_data/output/{nProj}/predict_data_cover_{noStates}',
             y_test=y_test.cpu().numpy(),  # Transfer to CPU and convert to NumPy
             maxlike_predict=maxlike_predict.cpu().numpy(),
             MLME_predict=MLME_predict.cpu().numpy(),
             model_predict=model_predict.cpu().numpy())

    # Compute metrics
    mae_criterion = nn.L1Loss(reduction='mean')
    model_mae = mae_criterion(model_predict, y_test).item()
    maxlike_mae = mae_criterion(maxlike_predict, y_test).item()
    MLME_mae = mae_criterion(MLME_predict, y_test).item()

    rmse_criterion = nn.MSELoss(reduction='mean')
    model_rmse = torch.sqrt(rmse_criterion(model_predict, y_test)).item()
    maxlike_rmse = torch.sqrt(rmse_criterion(maxlike_predict, y_test)).item()
    MLME_rmse = torch.sqrt(rmse_criterion(MLME_predict, y_test)).item()

    # Compute RMSE error bars
    mse_criterion = nn.MSELoss(reduction='none')
    model_mse = mse_criterion(model_predict, y_test).mean(dim=1)
    maxlike_mse = mse_criterion(maxlike_predict, y_test).mean(dim=1)
    MLME_mse = mse_criterion(MLME_predict, y_test).mean(dim=1)
    model_rmse_err_bar = torch.std(torch.sqrt(model_mse)) / np.sqrt(model_mse.shape[0])
    maxlike_rmse_err_bar = torch.std(torch.sqrt(maxlike_mse)) / np.sqrt(maxlike_mse.shape[0])
    MLME_rmse_err_bar = torch.std(torch.sqrt(MLME_mse)) / np.sqrt(MLME_mse.shape[0])

    # Store results
    results.append({
        'nProj': nProj,
        'model_mae': model_mae,
        'maxlike_mae': maxlike_mae,
        'MLME_mae': MLME_mae,
        'model_rmse': model_rmse,
        'maxlike_rmse': maxlike_rmse,
        'MLME_rmse': MLME_rmse,
        'model_rmse_err_bar': model_rmse_err_bar.item(),
        'maxlike_rmse_err_bar': maxlike_rmse_err_bar.item(),
        'MLME_rmse_err_bar': MLME_rmse_err_bar.item()
    })


def calculate_statistics(data, confidence=0.95):
    n = data.shape[0]  # Sample size (2000)
    mean = np.mean(data, axis=0)  # Mean across columns
    std = np.std(data, axis=0, ddof=1)  # Sample standard deviation (ddof=1)
    sem = std / np.sqrt(n)  # Standard error
    t_val = stats.t.ppf((1 + confidence) / 2, df=n - 1)  # t critical value
    ci_lower = mean - t_val * sem
    ci_upper = mean + t_val * sem
    return mean, sem, ci_lower, ci_upper


for nProj in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36]:
    result_data = np.load(f'./data/real_data/output/{nProj}/predict_data_cover_{noStates}.npz')
    print("==nProj", nProj)
    y_pred = result_data['model_predict']
    y_true = result_data['y_test']
    maxlike_predict = result_data['maxlike_predict']

    # Compute statistics for the three datasets
    y_true_mean, y_true_sem, y_true_ci_lower, y_true_ci_upper = calculate_statistics(y_true)
    y_pred_mean, y_pred_sem, y_pred_ci_lower, y_pred_ci_upper = calculate_statistics(y_pred)
    maxlike_mean, maxlike_sem, maxlike_ci_lower, maxlike_ci_upper = calculate_statistics(maxlike_predict)

    # Create results DataFrame
    results_df = pd.DataFrame({
        'feature_index': range(12),
        'y_true_mean': y_true_mean,
        'y_true_sem': y_true_sem,
        'y_pred_mean': y_pred_mean,
        'y_pred_sem': y_pred_sem,
        'maxlike_mean': maxlike_mean,
        'maxlike_sem': maxlike_sem
    })

    # Save to CSV file
    results_df.to_csv(f'./data/real_data/output/cover_{nProj}_statistics_results.csv', index=False, float_format='%.6f')