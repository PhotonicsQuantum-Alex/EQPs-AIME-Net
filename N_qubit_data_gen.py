import numpy as np
import logging
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
import os
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from mindquantum.core.gates import H, RZ

"""代码参数设定"""
N = 3
step_size = 0.1
"""
当step=0.01时，2626999（260W）组数据 ;
当step=0.03时，95876（10W）组数据 ;
当step=0.05时，19799（2W）组数据 ;"""
nProj_list = np.linspace(start = 6 ** N, stop = (6 ** N)/2, num = 5, dtype=int)

"""模型需要的参数"""
nProjAll = 19

"""log dir"""
save_dir = f"./data_generatino/{N}_qubits_data/step_size_{step_size}"
os.makedirs(save_dir, exist_ok=True)
log_dir = os.path.join(save_dir, "log.txt")
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler(log_dir, mode="a")])
logging.info(f"{N}qubit")
logging.info(f"step_size:{step_size}")
logging.info(f"nProj list:{nProj_list}")
logging.info(f"存储路径为：{save_dir}")

def x_rechange(x_data, nProj):
    n_x_data = []
    for i in range(x_data.shape[0]):  # noState
        input = []
        for j in range(nProj):  # nProj
            srmPOMs = x_data[i, j * ((2 ** N) ** 2 + 1): j * ((2 ** N) ** 2 + 1) + ((2 ** N) ** 2)]  # 获取 SRM POMs
            probListSRM = np.real(x_data[i, j * ((2 ** N) ** 2 + 1) + ((2 ** N) ** 2)])  # 获取 probListSRM 的实部
            # 将实部和虚部拼接到一起并附加 probListSRM
            # encode_srmPOMs_input = np.append(np.concatenate([np.real(srmPOMs),np.imag(srmPOMs)]),probListSRM)
            encode_srmPOMs_input = np.concatenate([np.real(srmPOMs), np.imag(srmPOMs), probListSRM.reshape(1)], axis=0)
            input.append(encode_srmPOMs_input)
        n_x_data.append(np.concatenate(input, axis=0))  # 拼接所有样本
    return np.array(n_x_data)


def mubpom():
    p1 = np.array([1, 0])
    p2 = np.array([0, 1])
    mub = np.zeros([6, 1, 2]) + 1j * np.zeros([6, 1, 2])
    mub[0] = p1
    mub[1] = p2
    mub[2] = 1 / np.sqrt(2) * (p1 + p2)
    mub[3] = 1 / np.sqrt(2) * (p1 - p2)
    mub[4] = 1 / np.sqrt(2) * (p1 + 1j * p2)
    mub[5] = 1 / np.sqrt(2) * (p1 - 1j * p2)
    mubp = [np.transpose(mub[i]) @ np.conjugate(mub[i]) for i in range(6)]
    return mubp


# def PauliForLearning(nProj):
#     mubsN = linspace(0, 6**N-1, nProj, dtype=int)
#     mub = mubpom()  #(list:6)
#     mub2 = array([kron(mub[i], mub[j]) / 9 for i in range(6) for j in range(6)])#(36,4,4)
#     mub3 = array([mub2[n] for n in sort(mubsN[0:nProj])])
#     return mub3
#
# def probdists(rho0,povm):
#     l = shape(povm)[0]
#     probtrue = array([real(trace(rho0@povm[i])) for i in range(l)])
#     probtrue = probtrue/sum(probtrue)
#     return probtrue

def generate_pauli_proj(nProj):
    """
    生成N量子比特的POVM投影算符。

    eigvecs_w：一个包含6个基础向量的矩阵，每列是一个基向量。
    N：量子比特数。

    返回：一个包含6^N个投影算符的列表，每个投影算符的形状为(2^N, 2^N)。
    """
    # 计算所有可能的6^N种测量结果
    num_measurements = 6 ** N
    povm_list = []
    # 创建所有可能的Pauli测量组合
    for i in range(num_measurements):
        # 根据 i 对应的基构造每个量子比特的测量方向
        binary_rep = np.base_repr(i, base=6).zfill(N)  # 获取N位的6进制数
        # 初始化投影矩阵为单位矩阵
        proj = np.eye(2 ** N)
        # 遍历所有量子比特并构造投影矩阵
        for qubit_idx in range(N):
            basis_idx = int(binary_rep[qubit_idx])
            # 选择对应的基向量
            proj_qubit = np.outer(eigvecs_w[:, basis_idx], eigvecs_w[:, basis_idx].conj())
            # 计算投影矩阵（通过Kronecker积组合）
            if qubit_idx == 0:
                proj = proj_qubit  # 第一量子比特直接取其投影
            else:
                proj = np.kron(proj, proj_qubit)
        # 将构造的投影矩阵添加到povm_list
        povm_list.append(proj)
    povm_list = np.array(povm_list)
    mubsN = np.linspace(0, 6 ** N - 1, nProj, dtype=int)
    povm_list = np.array([povm_list[n] for n in np.sort(mubsN[0:nProj])])  # no use np.sort
    return povm_list


"""pauli测量"""
sigma_0 = np.array([[1, 0], [0, 1]])
sigma_x = np.array([[0, 1], [1, 0]])
sigma_y = np.array([[0, -1j], [1j, 0]])
sigma_z = np.array([[1, 0], [0, -1]])
pauli_matrices = [sigma_0, sigma_z, sigma_x, sigma_y]
z_plus = np.array([1, 0])
z_minus = np.array([0, 1])
x_plus = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
x_minus = np.array([1 / np.sqrt(2), -1 / np.sqrt(2)])
y_plus = np.array([1 / np.sqrt(2), 1j / np.sqrt(2)])
y_minus = np.array([1 / np.sqrt(2), -1j / np.sqrt(2)])
eigvecs_w = np.column_stack((z_plus, z_minus, x_plus, x_minus, y_plus, y_minus))
S_combinations = np.array(np.meshgrid(*[[1, -1]] * N)).T.reshape(-1, N)
"""生成量子态"""


def state_gen(p_set):
    rho = np.zeros((2 ** N, 2 ** N), dtype=complex)
    # 创建N个量子比特的Kronecker积
    for i in range(4):
        # 初始化一个基于当前Pauli矩阵的Kronecker积
        current_kron = pauli_matrices[i]
        # 对于N个量子比特，执行Kronecker积扩展
        for _ in range(N - 1):
            current_kron = np.kron(current_kron, pauli_matrices[i])
        # 累加Kronecker积结果到rho
        rho += current_kron * p_set[i]
    rho /= 2 ** N
    return rho


def cover_state_probabilities(step_size=0.05):
    """
    生成所有量子态
    :param1 step_size: 步长.
    param setting reference:
        当step=0.01时，2626999（260W）组数据 ;
        当step=0.03时，95876（10W）组数据 ;
        当step=0.05时，19799（2W）组数据 ;
    :return:rho_list(num_of_state, N**2, N**2)密度矩阵
            valid_solutions(num_of_state, 4)对应的
    """
    # 设定步长
    range_vals = np.arange(-1, 1, step_size)

    # 检查特征值是否大于1
    def check_inequalities(pz, px, py):
        expr1 = (1 + pz + px - py) / 4
        expr2 = (1 + pz - px + py) / 4
        expr3 = (1 - pz + px + py) / 4
        expr4 = (1 - pz - px - py) / 4
        return expr1 > 0 and expr2 > 0 and expr3 > 0 and expr4 > 0

    # 遍历所有px, py, pz的组合
    p_set_solutions = []
    for pz in range_vals:
        for px in range_vals:
            for py in range_vals:
                if check_inequalities(pz, px, py):
                    p_set_solutions.append((1, pz, px, py))
    p_set_solutions = np.array(p_set_solutions)
    logging.info(f"构建的量子态数目为：{len(p_set_solutions)}")
    rho_list = np.array([state_gen(p_set) for p_set in p_set_solutions])
    logging.info(f"密度矩阵生成完成，形状为：{rho_list.shape}")  # (19799, 8, 8)
    logging.info(f"p_sets生成完成，形状为：{p_set_solutions.shape}")  # (19799, 4)
    return rho_list, p_set_solutions


def get_quasiprobs(rho_list, p_set_solutions):
    """
    根据密度矩阵和p_set返回准概率分布和残差值（根据优化算法计算rq）
    :param rho_list:
    :param p_set_solutions:
    :return: quais_p_list(num_of_state,3*2^N+1)
    """
    noStates = len(p_set_solutions)
    quasi_present = []
    for i in range(noStates):
        rho = rho_list[i]
        p_set = p_set_solutions[i]
        quasi_p_p2 = np.zeros((3, 2 ** N), dtype=complex)
        C = np.zeros((3, 2 ** N, 2 ** N, 2 ** N), dtype=complex)
        for i in range(3):
            # 根据S进行排列
            for index, S in enumerate(S_combinations):
                # 计算C集合
                current_kron = pauli_matrices[0] + S[0] * pauli_matrices[i + 1]  # SIGMA[0]+1*SIGMA[1]_z
                for j in range(1, N):
                    current_kron = np.kron(current_kron, pauli_matrices[0] + S[j] * pauli_matrices[
                        i + 1])  # SIGMA[0]+1*SIGMA[1]_z,SIGMA[0]+1*SIGMA[1]_z
                C[i, index] = current_kron / np.trace(current_kron)
                # 计算准概率的第二部分
                symbol_product = np.prod(S)  # 符号乘积
                quasi_p_p2[i, index] = symbol_product * p_set[i + 1] / 2 ** N
        """第五步:优化rq"""

        # 根据r和q计算准概率
        def get_quasi_p(r, q):
            quasi_p = np.zeros_like(quasi_p_p2)
            quasi_p[0, :] = quasi_p_p2[0, :] + r + q  # 更新z
            quasi_p[1, :] = quasi_p_p2[1, :] - r  # 更新x
            quasi_p[2, :] = quasi_p_p2[2, :] - q  # 更新y
            quasi_p += 1 / 3 / 2 ** N
            return quasi_p.real

        def loss(params):
            r, q = params  # r 和 q 的值
            quasi_p = get_quasi_p(r, q)
            negative_penalty = np.sum(np.minimum(0, quasi_p) ** 2)
            return negative_penalty

        # 优化 r 和 q，最小化损失函数
        initial_guess = np.array([0.25, -0.25])  # rq初始值
        result = minimize(loss, initial_guess, method='L-BFGS-B')
        # 输出最优的 r 和 q
        r_opt, q_opt = result.x
        quasi_p = get_quasi_p(r_opt, q_opt)
        # 重构密度矩阵,计算residual component
        rho_classical = np.zeros_like(rho, dtype=complex)
        for i in range(3):
            for index, S in enumerate(S_combinations):
                rho_classical += quasi_p[i, index] * C[i, index, :, :]
        residual_component = np.linalg.norm(rho - rho_classical)
        if residual_component < 1e-1:
            residual_component = 0.0
        quasi_present_i = np.append(quasi_p, residual_component)
        quasi_present.append(quasi_present_i)
    return np.array(quasi_present)


def get_quasiprob_from_rho(rho_list):
    # for i in N:
    # 假设M, M_Z, M_X, M_Y是已知矩阵
    M_Z = np.array([[...], [...], ...])  # 已知矩阵 M_Z
    M_X = np.array([[...], [...], ...])  # 已知矩阵 M_X
    M_Y = np.array([[...], [...], ...])  # 已知矩阵 M_Y
    # 将矩阵展平为列向量
    vec_M_Z = M_Z.flatten()
    vec_M_X = M_X.flatten()
    vec_M_Y = M_Y.flatten()
    A = np.column_stack((vec_M_Z, vec_M_X, vec_M_Y))

    for rho in rho_list:
        vec_M = rho.flatten()
        # 使用最小二乘法求解p_z, p_x, p_y
        p_z, p_x, p_y = np.linalg.lstsq(A, vec_M, rcond=None)[0]
        p_set = [p_z, p_x, p_y]
        get_quasiprobs(rho, p_set)


# ---------- 单量子比特Pauli本征态 ----------
_Z_PLUS = np.array([1.0, 0.0], dtype=complex)
_Z_MINUS = np.array([0.0, 1.0], dtype=complex)
_X_PLUS = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
_X_MINUS = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)
_Y_PLUS = np.array([1.0, 1.0j], dtype=complex) / np.sqrt(2)
_Y_MINUS = np.array([1.0, -1.0j], dtype=complex) / np.sqrt(2)

_LOCAL_EIGENSTATES = {
    "z+": _Z_PLUS,
    "z-": _Z_MINUS,
    "x+": _X_PLUS,
    "x-": _X_MINUS,
    "y+": _Y_PLUS,
    "y-": _Y_MINUS,
}


def _normalize_state(v):
    v = np.asarray(v, dtype=complex).reshape(-1)
    nrm = np.linalg.norm(v)
    if nrm < 1e-12:
        raise ValueError("State vector norm is zero.")
    return v / nrm


def _same_ray(v, w, tol=1e-6):
    """
    判断两个态矢是否只差一个全局相位。
    """
    v = _normalize_state(v)
    w = _normalize_state(w)
    overlap = np.vdot(w, v)
    if abs(overlap) < tol:
        return False
    phase = overlap / abs(overlap)
    return np.linalg.norm(v - phase * w) < tol


def _extract_rank1_state_from_projector(P, tol=1e-8):
    """
    从 rank-1 投影算符 P = |psi><psi| 提取 |psi>.
    """
    evals, evecs = np.linalg.eigh(P)
    idx = np.argmax(np.real(evals))
    lam = np.real(evals[idx])
    if abs(lam - 1.0) > 1e-5:
        raise ValueError("POVM element is not close to a rank-1 projector with eigenvalue 1.")
    psi = evecs[:, idx]
    return _normalize_state(psi)


def _factor_product_state(psi, n_qubits, tol=1e-8):
    """
    将 |psi> 分解成单比特直积态：
        |psi> = |v0> ⊗ |v1> ⊗ ... ⊗ |v_{n-1}>
    若不是严格直积态，则报错。
    """
    psi = _normalize_state(psi)
    tensor = psi.reshape([2] * n_qubits)

    local_states = []
    current = tensor

    for _ in range(n_qubits - 1):
        mat = current.reshape(2, -1)
        u, s, vh = np.linalg.svd(mat, full_matrices=False)

        if len(s) > 1 and s[1] > tol:
            raise ValueError("Projector eigenstate is not a product state; this replacement assumes product Pauli projectors.")

        local = u[:, 0] * np.sqrt(s[0])
        rest = vh[0, :] * np.sqrt(s[0])

        local = _normalize_state(local)
        local_states.append(local)

        current = rest.reshape([2] * (len(current.shape) - 1))

    last = _normalize_state(current.reshape(2))
    local_states.append(last)
    return local_states


def _identify_local_pauli_eigenstate(v, tol=1e-6):
    """
    识别单比特态属于 z±, x±, y± 中哪一个。
    """
    for label, ref in _LOCAL_EIGENSTATES.items():
        if _same_ray(v, ref, tol=tol):
            return label
    raise ValueError("Local state is not one of the Pauli eigenstates z±/x±/y±.")


def _basis_change_and_target_bit(label, qubit):
    """
    返回：
    1) 把该本征态旋到计算基的电路
    2) 目标比特值（0 或 1）

    约定：
    - z+ -> |0>
    - z- -> |1>
    - x± 先 H
    - y± 先 RZ(-pi/2) 再 H
    """
    circ = Circuit()

    if label == "z+":
        target = 0
    elif label == "z-":
        target = 1
    elif label == "x+":
        circ += H.on(qubit)
        target = 0
    elif label == "x-":
        circ += H.on(qubit)
        target = 1
    elif label == "y+":
        circ += RZ(-np.pi / 2).on(qubit)
        circ += H.on(qubit)
        target = 0
    elif label == "y-":
        circ += RZ(-np.pi / 2).on(qubit)
        circ += H.on(qubit)
        target = 1
    else:
        raise ValueError(f"Unknown label: {label}")

    return circ, target


def _big_endian_rho_to_mq_little_endian(rho, n_qubits):
    """
    你的 numpy / kron 生成方式通常按 big-endian 张量顺序组织；
    MindQuantum 模拟器采用 little-endian。
    这里把 rho 从 big-endian 重新排列到 MindQuantum 的顺序。
    """
    rho = np.asarray(rho, dtype=complex)
    dim = 2 ** n_qubits
    if rho.shape != (dim, dim):
        raise ValueError(f"rho shape should be {(dim, dim)}, got {rho.shape}")

    tensor = rho.reshape([2] * n_qubits + [2] * n_qubits)
    perm = list(range(n_qubits - 1, -1, -1)) + list(range(2 * n_qubits - 1, n_qubits - 1, -1))
    tensor_le = np.transpose(tensor, axes=perm)
    return tensor_le.reshape(dim, dim)


def probdists(rho, povm, tol=1e-6):
    """
    MindQuantum 版本的测量概率估计函数（精确模拟版）

    参数
    ----
    rho  : (2^N, 2^N) 密度矩阵
    povm : (l, 2^N, 2^N) 投影算符列表

    返回
    ----
    probtrue : (l,) 概率分布

    说明
    ----
    这版假设每个 POVM 元素都是“单比特 Pauli 本征态的张量积投影”。
    这与当前 generate_pauli_proj / mub3 的生成逻辑是一致的。
    """
    rho = np.asarray(rho, dtype=complex)
    l = povm.shape[0]
    dim = rho.shape[0]
    n_qubits = int(np.log2(dim))

    if 2 ** n_qubits != dim:
        raise ValueError("rho dimension is not a power of 2.")

    # 变到 MindQuantum 的 little-endian 顺序
    rho_mq = _big_endian_rho_to_mq_little_endian(rho, n_qubits)

    sim = Simulator('mqmatrix', n_qubits)
    probtrue = np.zeros(l, dtype=float)

    for i in range(l):
        P = np.asarray(povm[i], dtype=complex)

        # 1) 从 rank-1 projector 提取其本征态 |psi>
        psi = _extract_rank1_state_from_projector(P)

        # 2) 分解成单比特直积态
        local_states = _factor_product_state(psi, n_qubits, tol=1e-8)

        # 3) 为每个单比特构造“基变换”电路，并记录目标 bit
        rot_circ = Circuit()
        target_bits_big = []

        for q, local_v in enumerate(local_states):
            label = _identify_local_pauli_eigenstate(local_v, tol=tol)
            c, bit = _basis_change_and_target_bit(label, q)
            rot_circ += c
            target_bits_big.append(bit)

        # 4) target_bits_big 是按你原来 kron 的 big-endian 顺序；
        #    MindQuantum 内部是 little-endian，所以要反过来算索引
        target_bits_little = target_bits_big[::-1]
        # target_index = sum(bit << k for k, bit in enumerate(target_bits_little))
        target_index = sum((bit << k) for k, bit in enumerate(target_bits_little))
        # 5) 在 mqmatrix 上精确演化并取对角元
        sim.set_qs(rho_mq.copy())
        sim.apply_circuit(rot_circ)
        rho_rot = sim.get_qs()

        p = np.real(rho_rot[target_index, target_index])
        if p < 0 and abs(p) < 1e-10:
            p = 0.0
        probtrue[i] = p

    s = probtrue.sum()
    if s <= 0:
        raise ValueError("All probabilities are zero; please check POVM/state consistency.")
    probtrue /= s
    return probtrue


def data_save(x, y, p_set_solutions, rho_list, nProj):
    # 将数据转换为tensor（如果它们已经是numpy数组）
    x = torch.tensor(x, dtype=torch.float32).view(x.shape[0],1,-1)
    y = torch.tensor(y, dtype=torch.float32)
    p_set_solutions = torch.tensor(p_set_solutions, dtype=torch.float32)
    rho_list = torch.tensor(rho_list, dtype=torch.complex64)

    # 划分训练集、验证集和测试集（60%训练集，20%测试集，20%验证集）
    x_train, x_temp, y_train, y_temp, p_train, p_temp, rho_train, rho_temp = train_test_split(x, y, p_set_solutions,
                                                                                              rho_list, test_size=0.4,
                                                                                              random_state=42)
    x_val, x_test, y_val, y_test, p_val, p_test, rho_val, rho_test = train_test_split(x_temp, y_temp, p_temp, rho_temp,
                                                                                      test_size=0.5, random_state=42)
    # 将数据转换为TensorDataset
    train_dataset = torch.utils.data.TensorDataset(x_train, y_train, p_train, rho_train)
    val_dataset = torch.utils.data.TensorDataset(x_val, y_val, p_val, rho_val)
    test_dataset = torch.utils.data.TensorDataset(x_test, y_test, p_test, rho_test)
    # 存储数据集为.pt文件
    data_dir = os.path.join(save_dir,f"nProj_{nProj}")
    os.makedirs(data_dir, exist_ok=True)
    torch.save(train_dataset, os.path.join(data_dir,'train_dataset.pt'))
    torch.save(val_dataset, os.path.join(data_dir, 'val_dataset.pt'))
    torch.save(test_dataset, os.path.join(data_dir, 'test_dataset.pt'))


def data_gen(nProj_list):
    """
    根据nProj_list观测基的数量，生成并存储数据
    :param nProj_list:(start = 6 ** N, stop = (6 ** N)/2, num = 5, dtype=int)
    :return: none
    """
    """compute rho&quasipresent"""
    rho_list, p_set_solutions = cover_state_probabilities(step_size)
    quasi_present = get_quasiprobs(rho_list, p_set_solutions)
    logging.info(f"准概率计算完成，形状为{quasi_present.shape}")
    noStates = len(rho_list)
    """log logging.info:"""
    logging.info(f"noStates:{noStates}")
    """random shuffle"""
    index = np.arange(noStates)
    np.random.shuffle(index)
    rho_list = rho_list[index]
    p_set_solutions = p_set_solutions[index]
    quasi_present = quasi_present[index]
    """
    不同测量基下生成noState数据，存有：
        1.rho_list:(noStates, 2**N, 2**N)
        2.p_set_solutions:(noStates, 4)
        3.quasi_present:(noStates, 3*2^N+1)
        4.maxlike:(noStates, 2**N, 2**N)    which is almost impossible to compute the quasiprobabilities from rho matrix, but we can map the most simulation_rho to get the closest p_set.but we should compare the different ways (map or step p_set_solution)
        5.model_input:(noStates, nProj*((2**N)^2+1))
    """
    for nProj in tqdm(nProj_list):  # 示例for nProj in tqdm(range(26, 38, 2)):
        # get model_input:x
        povm_list = generate_pauli_proj(nProj)  # (nProj:选择的测量组,2^N：,2^N)
        srmPOMs = np.array([povm_list for _ in range(noStates)])  # noState, nProj,   # (noState, nProj, 2^N, 2^N)
        probListSRM = np.array([probdists(rho_list[n], srmPOMs[n]) for n in range(noStates)])

        x_org = np.array([np.hstack((srmPOMs[n].reshape(-1, (2 ** N) ** 2), probListSRM[n].reshape(-1, 1))).flatten() for n in
                       range(noStates)])  # 复数展平
        x = x_rechange(x_org, nProj)  # 拼接实部与虚部，模型输入(noState,nProj*(2*(2^N)^2+1))
        data_save(x, quasi_present, p_set_solutions, rho_list, nProj)# 模型输出(num_of_state,3*2^N+1)
        logging.info(f"数据生成记录：./data_generatino/{N}_qubits_data/nProj_{nProj}_{noStates}")


if __name__ == "__main__":
    np.random.seed(0)
    data_gen(nProj_list)
