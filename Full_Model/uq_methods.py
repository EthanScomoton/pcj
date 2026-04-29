"""
不确定性量化基线方法
====================
P0 需求: 修复 reparam UQ 基线或用 MC Dropout / Deep Ensembles 替代
对应调研报告 §三、3.3

三种方法:
  - mc_dropout     : 推理时保持 dropout 开启, 采样 N 次, 取均值和方差
  - reparam        : 模型输出 (mu, logvar), 经 delta 方法反归一化 (旧基线)
  - deep_ensembles : 训练 N 个模型, 用它们的方差作为 epistemic 不确定性

论文参考:
  - Gal & Ghahramani (2016) "Dropout as a Bayesian approximation" (MC Dropout)
  - Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty
    Estimation using Deep Ensembles"
"""
import numpy as np
import torch
import torch.nn as nn


def _set_dropout_train(model):
    """把模型中所有 Dropout 层设为 train 模式 (其他层保持 eval)"""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def _denorm_mu(mu_raw, scaler_y, use_log_y=True):
    """反归一化: StandardScaler → [log1p 可选]"""
    if scaler_y is not None:
        mu = scaler_y.inverse_transform(np.asarray(mu_raw).reshape(-1, 1)).flatten()
    else:
        mu = np.asarray(mu_raw, dtype=float).flatten()
    if use_log_y:
        mu = np.expm1(mu)
    return mu


# ======================================================================
# 1) MC Dropout
# ======================================================================
def mc_dropout_predict(model, x, n_samples=30, scaler_y=None, use_log_y=True,
                       device=None):
    """
    在保持 dropout 开启的前提下对 x 做 n_samples 次前向, 返回均值和标准差。

    Parameters
    ----------
    model     : torch.nn.Module, 必须含 nn.Dropout 层
    x         : torch.Tensor 或 ndarray, 形状 [B, seq_len, feat] (已归一化)
    n_samples : int, 采样次数 (典型 20-50)
    scaler_y  : sklearn scaler, 用于反归一化 (可选)
    use_log_y : bool, 若训练目标做过 log1p, 则反归一化时需 expm1

    Returns
    -------
    mu    : ndarray [B], 原尺度均值预测
    sigma : ndarray [B], 原尺度标准差 (epistemic 不确定性)
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)

    # 关键: model.eval() 然后手动开启 Dropout
    model.eval()
    _set_dropout_train(model)

    preds_norm = []  # 归一化空间的样本
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            # 模型若输出 (mu, logvar), 只取 mu 分量
            if out.ndim > 1 and out.shape[-1] >= 2:
                out = out[..., 0]
            preds_norm.append(out.detach().cpu().numpy().flatten())

    preds_norm = np.asarray(preds_norm)   # [N, B]

    # 每个样本分别反归一化, 再取均值/方差以保持原尺度的统计量
    B = preds_norm.shape[1]
    preds_real = np.zeros_like(preds_norm)
    for i in range(preds_norm.shape[0]):
        preds_real[i] = _denorm_mu(preds_norm[i], scaler_y, use_log_y)

    mu = preds_real.mean(axis=0)
    sigma = preds_real.std(axis=0, ddof=1) if n_samples > 1 else np.zeros(B)

    # 重置为 eval 模式 (关闭 Dropout)
    model.eval()
    return mu, sigma


def mc_dropout_interval(model, x, n_samples=30, alpha=0.10,
                        scaler_y=None, use_log_y=True, device=None):
    """
    返回 MC Dropout 的 (1-alpha) 分位区间 (经验分位数, 无高斯假设)。

    返回 (mu, lower, upper)
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)

    model.eval()
    _set_dropout_train(model)

    preds_norm = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            if out.ndim > 1 and out.shape[-1] >= 2:
                out = out[..., 0]
            preds_norm.append(out.detach().cpu().numpy().flatten())
    preds_norm = np.asarray(preds_norm)   # [N, B]

    # 反归一化
    preds_real = np.array([_denorm_mu(p, scaler_y, use_log_y)
                           for p in preds_norm])

    mu    = preds_real.mean(axis=0)
    lower = np.quantile(preds_real, alpha / 2,     axis=0)
    upper = np.quantile(preds_real, 1 - alpha / 2, axis=0)
    # 修复 Plot 8: 防止 lower > upper (经验分位本身有序, 这里仅作保险), 防 lower < 0
    lower = np.minimum(lower, upper)
    lower = np.maximum(lower, 0.0)

    model.eval()
    return mu, lower, upper


# ======================================================================
# 2) Reparam (旧基线, 修正版)
# ======================================================================
def reparam_predict(model, x, scaler_y=None, use_log_y=True, device=None):
    """
    单次前向取模型的 (mu_raw, logvar_raw), 用 delta 方法反归一化。

    sigma_real = (1 + |mu_real|) * scaler_y.scale_ * sigma_norm
    这里 (1 + |mu_real|) 来自 log1p 链式求导 d expm1(u) / du = exp(u) ≈ 1+mu_real
    """
    if device is None:
        device = next(model.parameters()).device

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)

    model.eval()
    with torch.no_grad():
        out = model(x).detach().cpu().numpy()
    if out.ndim == 1:
        # 模型仅输出点预测, 没有 logvar, 退回 sigma=0
        mu = _denorm_mu(out, scaler_y, use_log_y)
        return mu, np.zeros_like(mu)

    mu_raw     = out[..., 0].flatten()
    logvar_raw = out[..., 1].flatten() if out.shape[-1] >= 2 else np.zeros_like(mu_raw)

    mu_real = _denorm_mu(mu_raw, scaler_y, use_log_y)
    sigma_norm = np.exp(0.5 * logvar_raw)

    scale_y = float(scaler_y.scale_[0]) if scaler_y is not None else 1.0
    sigma_real = (1.0 + np.abs(mu_real)) * scale_y * sigma_norm

    return mu_real, np.maximum(sigma_real, 1.0)


# ======================================================================
# 3) Deep Ensembles (stub — 需多模型训练)
# ======================================================================
def deep_ensembles_predict(models_list, x, scaler_y=None, use_log_y=True,
                           device=None):
    """
    多个独立训练的模型投票。如果没有多个 checkpoint, 调用方应捕获 ValueError
    并回退到 MC Dropout。

    Parameters
    ----------
    models_list : list[nn.Module]   N 个独立训练的模型 (推荐 N=5)
    """
    if not models_list or len(models_list) < 2:
        raise ValueError(
            "Deep Ensembles 需要 ≥ 2 个独立训练的模型. "
            "目前仅提供 1 个 checkpoint, 请使用 MC Dropout 替代.")

    if device is None:
        device = next(models_list[0].parameters()).device

    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    x = x.to(device)

    preds_real = []
    for m in models_list:
        m.eval()
        with torch.no_grad():
            out = m(x)
            if out.ndim > 1 and out.shape[-1] >= 2:
                out = out[..., 0]
            out_np = out.detach().cpu().numpy().flatten()
        preds_real.append(_denorm_mu(out_np, scaler_y, use_log_y))

    preds_real = np.asarray(preds_real)
    mu = preds_real.mean(axis=0)
    sigma = preds_real.std(axis=0, ddof=1)
    return mu, sigma


# ======================================================================
# 统一入口
# ======================================================================
def get_uq(method, model, x, scaler_y=None, use_log_y=True,
           n_samples=30, device=None, extra_models=None):
    """
    统一 UQ 入口, 根据 method 调用对应方法。

    method ∈ {'mc_dropout', 'reparam', 'deep_ensembles'}
    返回 (mu, sigma) 都是原尺度 ndarray
    """
    method = method.lower()
    if method == 'mc_dropout':
        return mc_dropout_predict(model, x, n_samples=n_samples,
                                  scaler_y=scaler_y, use_log_y=use_log_y,
                                  device=device)
    elif method == 'reparam':
        return reparam_predict(model, x, scaler_y=scaler_y,
                               use_log_y=use_log_y, device=device)
    elif method == 'deep_ensembles':
        if extra_models is None:
            raise ValueError("deep_ensembles 需要 extra_models 参数")
        return deep_ensembles_predict([model] + list(extra_models), x,
                                      scaler_y=scaler_y, use_log_y=use_log_y,
                                      device=device)
    else:
        raise ValueError(f"未知 UQ 方法: {method}")
