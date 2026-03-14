import mindspore.numpy as mnp
import mindspore as ms
import numpy as np  # For complex number handling outside the model
# For data loading and sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from model import MyNet_MLP_Improved_GELU_3qubit

# Set MindSpore context (adjust device_target as needed: 'GPU', 'CPU', 'Ascend')
ms.set_context(mode=ms.PYNATIVE_MODE, device_target='CPU')


# Custom sqrtm for density matrices using eigh (using NumPy for complex ops)
def sqrtm(rho):
    """Compute the matrix square root of a Hermitian positive semi-definite matrix rho.
    Uses eigenvalue decomposition."""
    eigenvalues, eigenvectors = np.linalg.eigh(rho)
    # Clip negative eigenvalues to zero due to numerical errors
    eigenvalues = np.maximum(eigenvalues, 0.0)
    sqrt_eigenvalues = np.sqrt(eigenvalues)
    # Reconstruct sqrt(rho) = V @ diag(sqrt(eigenvalues)) @ V^\dagger
    diag_sqrt = np.diag(sqrt_eigenvalues)
    sqrt_rho = eigenvectors @ diag_sqrt @ np.conj(np.transpose(eigenvectors))
    return sqrt_rho


def load_data(nProj, data_path):
    def x_derefresh(n_x_data, nProj):
        n_samples = n_x_data.shape[0]  # Number of samples
        restored_x_data = []
        for i in range(n_samples):
            sample = []
            for j in range(nProj):
                # Each projection data length is 129 (64 real parts + 64 imag parts + 1 probListSRM)
                start_idx = j * 129
                encode_srmPOMs_input = n_x_data[i, start_idx:start_idx + 129]
                # Extract real part, imag part, and probListSRM
                real_part = encode_srmPOMs_input[:64]  # First 64 are real part
                imag_part = encode_srmPOMs_input[64:128]  # Next 64 are imag part
                probListSRM = encode_srmPOMs_input[128]  # Last one is probListSRM
                # Concatenate real part, imag part, and probListSRM into a real vector
                proj_data = np.concatenate([real_part, imag_part, np.array([probListSRM])], axis=0)
                sample.append(proj_data)
            # Concatenate all projection data
            restored_x_data.append(np.concatenate(sample, axis=0))
        return np.array(restored_x_data)

    data = np.load(data_path, allow_pickle=True)
    # Extract training, validation, test data and their corresponding labels
    x_train_and_val = data['x_train']
    y_train_and_val = data['quasi_present_train']
    x_train, x_val, y_train, y_val = train_test_split(
        x_train_and_val, y_train_and_val, test_size=0.2,  # Validation set 20%
        random_state=42  # Set random seed for reproducibility
    )
    x_test = data['x_test'][0:4000, :]
    y_test = data['quasi_present_test'][0:4000, :]
    maxlike_predict = data['maxlike_predict'][0:4000, :]
    MLME_predict = data['MLME_predict'][0:4000, :]
    # Process x data
    x_train = x_derefresh(x_train, nProj)
    x_val = x_derefresh(x_val, nProj)
    x_test = x_derefresh(x_test, nProj)
    # Return all data
    return x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict


# Global variables
N = 3
sigma_0 = np.array([[1, 0], [0, 1]], dtype=np.complex128)
sigma_x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
sigma_z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
pauli_matrices = [sigma_0, sigma_z, sigma_x, sigma_y]
S_combinations = np.array(np.meshgrid(*[[1, -1]] * N)).T.reshape(-1, N)


def precompute_C():
    """Precompute C tensor using NumPy for complex ops"""
    C = np.zeros((3, 2 ** N, 2 ** N, 2 ** N), dtype=np.complex128)
    for i in range(3):
        for index, S in enumerate(S_combinations):
            current_kron = pauli_matrices[0] + S[0] * pauli_matrices[i + 1]
            for j in range(1, N):
                current_kron = np.kron(current_kron, pauli_matrices[0] + S[j] * pauli_matrices[i + 1])
            C[i, index] = current_kron / np.trace(current_kron)
    return C


def recover_rho_from_quasi(quasi_present, C=None):
    """Recover rho_list from quasi_present using NumPy for complex ops"""
    if C is None:
        C = precompute_C()
    noStates = quasi_present.shape[0]
    dim = 2 ** N
    rho_list = np.zeros((noStates, dim, dim), dtype=np.complex128)
    for i in range(noStates):
        # Extract quasi_p (real part, shape (3, 2^N))
        quasi_p_flat = quasi_present[i, :-1]
        quasi_p = quasi_p_flat.reshape(3, dim)
        # Reconstruct rho_classical
        rho_rec = np.zeros((dim, dim), dtype=np.complex128)
        for dir_idx in range(3):
            for index in range(dim):
                rho_rec += quasi_p[dir_idx, index] * C[dir_idx, index]
        # Normalize trace to 1 (handle numerical errors)
        rho_rec = rho_rec / np.trace(rho_rec)
        rho_list[i] = rho_rec
    return rho_list


def fidelity(rho1, rho2):
    """Compute fidelity F(rho1, rho2) = [Tr sqrt(sqrt(rho1) rho2 sqrt(rho1))]^2
    Assumes rho1, rho2 are (dim, dim) complex matrices. Uses NumPy."""
    sqrt_rho1 = sqrtm(rho1)
    temp = sqrt_rho1 @ rho2 @ sqrt_rho1
    sqrt_temp = sqrtm(temp)
    fid = np.real(np.trace(sqrt_temp)) ** 2
    return np.clip(fid, 0, 1)  # Clamp to [0,1]

def purity(rho):
    """Compute purity Tr(rho^2) using NumPy"""
    return np.real(np.trace(rho @ rho))

def von_neumann_entropy(rho):
    """
    计算密度矩阵 rho 的冯·诺伊曼熵
    rho: 必须是 Hermitian 正半定矩阵，迹为1
    """
    # 只保留数值上大于机器精度的特征值，避免数值误差
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = eigvals[eigvals > 1e-14]  # 过滤数值噪声
    return -np.sum(eigvals * np.log2(eigvals + 1e-30))  # 加小量避免 log(0)


def partial_trace(rho, keep, dims=[2,2,2]):
    """
    对多体密度矩阵求部分迹
    keep: 保留的子系统索引列表（从0开始）
    dims: 每个子系统的维度（默认 2,2,2）
    """
    n = len(dims)
    total_dim = np.prod(dims)
    rho = rho.reshape(dims + dims)  # 变成张量形式

    # 需要 trace 掉的腿
    trace_out = [i for i in range(n) if i not in keep]

    # 先 trace 掉所有不需要保留的系统
    for i in sorted(trace_out, reverse=True):
        rho = np.trace(rho, axis1=i, axis2=i+n)
    return rho


def mutual_information_matrix(rho_3qubit):
    """
    输入: 3-qubit 密度矩阵 (8×8 复数矩阵)
    输出: 3×3 的量子互信息矩阵
    """
    dims = [2, 2, 2]
    I = np.zeros((3, 3))

    # 单比特熵
    S1 = von_neumann_entropy(partial_trace(rho_3qubit, [0], dims))
    S2 = von_neumann_entropy(partial_trace(rho_3qubit, [1], dims))
    S3 = von_neumann_entropy(partial_trace(rho_3qubit, [2], dims))

    # 双比特熵
    S12 = von_neumann_entropy(partial_trace(rho_3qubit, [0,1], dims))
    S13 = von_neumann_entropy(partial_trace(rho_3qubit, [0,2], dims))
    S23 = von_neumann_entropy(partial_trace(rho_3qubit, [1,2], dims))

    # 两两互信息
    I[0,1] = I[1,0] = S1 + S2 - S12
    I[0,2] = I[2,0] = S1 + S3 - S13
    I[1,2] = I[2,1] = S2 + S3 - S23

    return I


# Main computation part
nProj_list = np.linspace(start=6 ** N, stop=(6 ** N) / 30, num=10, dtype=int)[:-1]
print(f"nProj_list: {nProj_list}")
C = precompute_C()  # Global C using NumPy

# Storage for results
results = {
    'nProj': [],
    'model_fid_rmse': [],
    'maxlike_fid_rmse': [],
    'MLME_fid_rmse': [],
    'model_pur_rmse': [],
    'maxlike_pur_rmse': [],
    'MLME_pur_rmse': [],
    'model_fid_rmse_err': [],
    'maxlike_fid_rmse_err': [],
    'MLME_fid_rmse_err': [],
    'model_pur_rmse_err': [],
    'maxlike_pur_rmse_err': [],
    'MLME_pur_rmse_err': []
}

dictBestModel = {}
for k in nProj_list:
    dictBestModel[str(k)] = "./model/3qubit_model/50bestModelPauliConcProjR0v7" + str(
        k) + '.ckpt'  # Changed to .ckpt for MindSpore

for nProj in nProj_list:
    print(f"\nProcessing nProj = {nProj}")
    # Load data
    data_path = f'./data/three_qubit_data/nProj{nProj}_train_and_test.npz'
    x_train, y_train, x_val, y_val, x_test, y_test, maxlike_predict, MLME_predict = load_data(nProj,
                                                                                              data_path=data_path)
    # Recover rho_true from y_test (using NumPy)
    rho_true = recover_rho_from_quasi(y_test, C)

    # Load model and predict
    model = MyNet_MLP_Improved_GELU_3qubit(nProj)
    best_model_path_final = dictBestModel[str(nProj)]
    x_test_tensor = ms.Tensor(x_test, dtype=ms.float32).reshape((x_test.shape[0], 1, -1))
    # Load checkpoint (assume model is converted to MindSpore .ckpt format)
    param_dict = ms.load_checkpoint(best_model_path_final)
    ms.load_param_into_net(model, param_dict)
    model.set_train(False)
    model_predict = model(x_test_tensor).asnumpy()  # To numpy

    # Convert to numpy for recovery (already numpy from load_data)
    maxlike_predict_np = maxlike_predict
    MLME_predict_np = MLME_predict

    # Recover rho_pred from predictions (using NumPy)
    rho_model = recover_rho_from_quasi(model_predict, C)
    rho_maxlike = recover_rho_from_quasi(maxlike_predict_np, C)
    rho_MLME = recover_rho_from_quasi(MLME_predict_np, C)

    # Compute Fidelity RMSE
    fid_model = [fidelity(rho_t, rho_m) for rho_t, rho_m in zip(rho_true, rho_model)]
    fid_maxlike = [fidelity(rho_t, rho_ml) for rho_t, rho_ml in zip(rho_true, rho_maxlike)]
    fid_MLME = [fidelity(rho_t, rho_mle) for rho_t, rho_mle in zip(rho_true, rho_MLME)]
    rmse_fid_model = np.sqrt(np.mean((np.array(fid_model) - 1) ** 2))  # RMSE w.r.t. ideal 1
    rmse_fid_maxlike = np.sqrt(np.mean((np.array(fid_maxlike) - 1) ** 2))
    rmse_fid_MLME = np.sqrt(np.mean((np.array(fid_MLME) - 1) ** 2))

    # Compute Fidelity RMSE error bars (standard error, based on individual |error| std / sqrt(n))
    fid_errors_model = 1 - np.array(fid_model)
    fid_rmse_err_model = np.std(fid_errors_model) / np.sqrt(len(fid_errors_model))
    fid_errors_maxlike = 1 - np.array(fid_maxlike)
    fid_rmse_err_maxlike = np.std(fid_errors_maxlike) / np.sqrt(len(fid_errors_maxlike))
    fid_errors_MLME = 1 - np.array(fid_MLME)
    fid_rmse_err_MLME = np.std(fid_errors_MLME) / np.sqrt(len(fid_errors_MLME))

    # Compute Purity RMSE
    pur_true = [purity(rho) for rho in rho_true]
    pur_model = [purity(rho) for rho in rho_model]
    pur_maxlike = [purity(rho) for rho in rho_maxlike]
    pur_MLME = [purity(rho) for rho in rho_MLME]
    rmse_pur_model = np.sqrt(np.mean((np.array(pur_model) - np.array(pur_true)) ** 2))
    rmse_pur_maxlike = np.sqrt(np.mean((np.array(pur_maxlike) - np.array(pur_true)) ** 2))
    rmse_pur_MLME = np.sqrt(np.mean((np.array(pur_MLME) - np.array(pur_true)) ** 2))

    # Compute Purity RMSE error bars (standard error, based on individual |error| std / sqrt(n))
    pur_errors_model = np.array(pur_model) - np.array(pur_true)
    pur_rmse_err_model = np.std(np.abs(pur_errors_model)) / np.sqrt(len(pur_errors_model))
    pur_errors_maxlike = np.array(pur_maxlike) - np.array(pur_true)
    pur_rmse_err_maxlike = np.std(np.abs(pur_errors_maxlike)) / np.sqrt(len(pur_errors_maxlike))
    pur_errors_MLME = np.array(pur_MLME) - np.array(pur_true)
    pur_rmse_err_MLME = np.std(np.abs(pur_errors_MLME)) / np.sqrt(len(pur_errors_MLME))

    # Store
    results['nProj'].append(nProj)
    results['model_fid_rmse'].append(rmse_fid_model)
    results['maxlike_fid_rmse'].append(rmse_fid_maxlike)
    results['MLME_fid_rmse'].append(rmse_fid_MLME)
    results['model_pur_rmse'].append(rmse_pur_model)
    results['maxlike_pur_rmse'].append(rmse_pur_maxlike)
    results['MLME_pur_rmse'].append(rmse_pur_MLME)
    results['model_fid_rmse_err'].append(fid_rmse_err_model)
    results['maxlike_fid_rmse_err'].append(fid_rmse_err_maxlike)
    results['MLME_fid_rmse_err'].append(fid_rmse_err_MLME)
    results['model_pur_rmse_err'].append(pur_rmse_err_model)
    results['maxlike_pur_rmse_err'].append(pur_rmse_err_maxlike)
    results['MLME_pur_rmse_err'].append(pur_rmse_err_MLME)

    print(f"nProj {nProj}:")
    print(f" Model Fidelity RMSE: {rmse_fid_model:.6f} ± {fid_rmse_err_model:.6f}")
    print(f" Maxlike Fidelity RMSE: {rmse_fid_maxlike:.6f} ± {fid_rmse_err_maxlike:.6f}")
    print(f" MLME Fidelity RMSE: {rmse_fid_MLME:.6f} ± {fid_rmse_err_MLME:.6f}")
    print(f" Model Purity RMSE: {rmse_pur_model:.6f} ± {pur_rmse_err_model:.6f}")
    print(f" Maxlike Purity RMSE: {rmse_pur_maxlike:.6f} ± {pur_rmse_err_maxlike:.6f}")
    print(f" MLME Purity RMSE: {rmse_pur_MLME:.6f} ± {pur_rmse_err_MLME:.6f}")

# Output summary table
df_results = pd.DataFrame(results)
print("\nSummary Table:")
print(df_results.to_string(index=False))

# Save to CSV
df_results.to_csv('3qubit_test_F_and_P.csv', index=False)
print("\nResults saved to '3qubit_test_F_and_P.csv'")