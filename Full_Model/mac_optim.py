"""
Apple Silicon (M1/M2/M3/M4) 优化工具
====================================
针对 macOS + Metal + Accelerate 的性能优化:

1. auto_device()      : 自动选 MPS / CUDA / CPU, 统一 fallback
2. configure_blas()   : 设置 veclib / OMP 线程数 = P 核数
3. parallel_map()     : 封装 joblib/multiprocessing 跑并行实验
4. try_compile()      : 有条件开启 torch.compile (2.0+)
5. batch_infer()      : 通用批量推理 helper, 避免一次一个序列的 kernel launch

使用示例 (在 main.py 最上方):
    >>> from mac_optim import auto_device, configure_blas, log_system
    >>> configure_blas()
    >>> device = auto_device()
    >>> log_system(device)
"""
from __future__ import annotations
import os
import platform
import sys
import numpy as np


# ======================================================================
# 1) 设备自动选择
# ======================================================================
def auto_device(verbose: bool = True,
                prefer: str = 'auto',
                fallback_on_mps_bug: bool = True):
    """
    返回 torch.device, 顺序: MPS > CUDA > CPU。

    Parameters
    ----------
    prefer : 'auto' / 'mps' / 'cuda' / 'cpu'
    fallback_on_mps_bug : 若 MPS 遇到不支持的算子 (如 LSTM 特定变体),
                          设置环境变量让它自动回退 CPU
    """
    import torch

    if fallback_on_mps_bug:
        # PyTorch 对不支持的 MPS 算子回退 CPU (不报错)
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

    if prefer == 'cpu':
        dev = torch.device('cpu')
    elif prefer == 'cuda' and torch.cuda.is_available():
        dev = torch.device('cuda')
    elif prefer == 'mps' and torch.backends.mps.is_available():
        dev = torch.device('mps')
    else:
        # auto
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            dev = torch.device('mps')
        elif torch.cuda.is_available():
            dev = torch.device('cuda')
        else:
            dev = torch.device('cpu')

    if verbose:
        print(f"  [mac_optim] torch device = {dev}")
    return dev


# ======================================================================
# 2) BLAS / OMP 线程配置
# ======================================================================
def _detect_perf_cores() -> int:
    """
    检测 Apple Silicon 的性能核 (P-cores) 数。
    M1 = 4 P, M1 Pro = 6/8 P, M1 Max = 8 P, M2 = 4 P, M3 = 4-6 P, M4 = 4-10 P
    若不是 Apple Silicon, 返回物理核数。
    """
    try:
        import subprocess
        # macOS: sysctl 可以分别读 performance / efficiency 核数
        r = subprocess.run(
            ['sysctl', '-n', 'hw.perflevel0.physicalcpu'],
            capture_output=True, text=True, timeout=2)
        if r.returncode == 0 and r.stdout.strip().isdigit():
            return int(r.stdout.strip())
    except Exception:
        pass
    # 兜底: 逻辑核的 2/3, 留点给系统
    try:
        return max(1, (os.cpu_count() or 4) * 2 // 3)
    except Exception:
        return 4


def configure_blas(n_threads: int | None = None, verbose: bool = True):
    """
    设置 NumPy/SciPy 的 BLAS 线程数 = P 核数。必须在 import numpy/torch 之前
    或至少在任何计算密集调用之前设置才有效。

    在 Apple Silicon 上, NumPy 默认用 Accelerate (veclib), 关键变量:
      - VECLIB_MAXIMUM_THREADS   (Accelerate)
      - OPENBLAS_NUM_THREADS     (若换装了 openblas 版 numpy)
      - MKL_NUM_THREADS          (若装了 Intel MKL — ARM 下无效)
      - OMP_NUM_THREADS          (OpenMP, 影响 sklearn / scipy / torch CPU)
      - NUMEXPR_MAX_THREADS      (pandas.eval)
    """
    if n_threads is None:
        n_threads = _detect_perf_cores()
    n = str(int(max(1, n_threads)))

    for v in ['VECLIB_MAXIMUM_THREADS',
              'OPENBLAS_NUM_THREADS',
              'MKL_NUM_THREADS',
              'OMP_NUM_THREADS',
              'NUMEXPR_MAX_THREADS']:
        os.environ[v] = n

    # torch CPU 线程
    try:
        import torch
        torch.set_num_threads(int(n))
    except ImportError:
        pass

    if verbose:
        print(f"  [mac_optim] BLAS/OMP 线程数 = {n} "
              f"(Apple Silicon P-core 检测)")


# ======================================================================
# 3) 并行实验 helper (多进程, 充分利用 M 芯片多核)
# ======================================================================
def parallel_map(func, iterable, n_jobs: int = -1,
                 backend: str = 'loky', verbose: int = 0):
    """
    轻量级并行 map, 优先 joblib, 失败回退 multiprocessing, 再失败回退串行。

    注意: func 必须是 picklable (顶层函数, 不能是 lambda / 嵌套函数)。
    backend ∈ {'loky', 'threading', 'multiprocessing'}
      - 'loky' / 'multiprocessing': 真并行, CPU 密集
      - 'threading': 仅 IO 密集或 C 扩展释放 GIL 时有效
    """
    # n_jobs = -1 → 所有 P-cores
    if n_jobs == -1:
        n_jobs = _detect_perf_cores()

    items = list(iterable)
    if len(items) == 0:
        return []
    if n_jobs == 1 or len(items) == 1:
        return [func(x) for x in items]

    # 尝试 joblib
    try:
        from joblib import Parallel, delayed
        return Parallel(n_jobs=n_jobs, backend=backend, verbose=verbose) \
               (delayed(func)(x) for x in items)
    except Exception as e:
        if verbose:
            print(f"  [mac_optim] joblib 失败 ({e}), 回退 multiprocessing")

    # 回退 multiprocessing
    try:
        import multiprocessing as mp
        ctx = mp.get_context('spawn')   # macOS 上 fork 可能崩, 用 spawn
        with ctx.Pool(n_jobs) as pool:
            return pool.map(func, items)
    except Exception as e:
        if verbose:
            print(f"  [mac_optim] multiprocessing 失败 ({e}), 串行")

    # 串行兜底
    return [func(x) for x in items]


# ======================================================================
# 4) torch.compile (可选)
# ======================================================================
def try_compile(model, mode: str = 'reduce-overhead', verbose: bool = True):
    """
    尝试用 torch.compile 加速模型。对 MPS 后端要求 PyTorch 2.3+。
    失败自动返回原模型。
    """
    try:
        import torch
        if not hasattr(torch, 'compile'):
            if verbose:
                print("  [mac_optim] torch.compile 不可用 (需 PyTorch 2.0+)")
            return model
        # MPS 上 compile 支持不稳定, 仅 CPU/CUDA 上启用
        dev = next(model.parameters()).device
        if dev.type == 'mps':
            if verbose:
                print("  [mac_optim] MPS 上跳过 torch.compile (兼容性问题)")
            return model
        compiled = torch.compile(model, mode=mode)
        if verbose:
            print(f"  [mac_optim] torch.compile({mode}) 成功")
        return compiled
    except Exception as e:
        if verbose:
            print(f"  [mac_optim] torch.compile 失败 ({e}), 使用原模型")
        return model


# ======================================================================
# 5) 批量推理 helper
# ======================================================================
def batch_infer(model, X_all, batch_size: int = 256, device=None,
                to_numpy: bool = True):
    """
    对形状为 [N, ...] 的 X_all 分批前向, 返回 [N, out_dim] 或 [N]。

    比 "for 循环单样本 forward" 通常快 10-30×, 因为:
    - 减少 Metal/CUDA kernel launch 次数
    - 更好利用 GPU SIMD / matmul 吞吐
    - 避免反复 CPU↔GPU 拷贝
    """
    import torch
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    X_all = np.asarray(X_all, dtype=np.float32)
    n = len(X_all)
    outs = []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = torch.from_numpy(X_all[i:i + batch_size]).to(device)
            out = model(batch)
            # 若模型输出 [B, 2] (mu, logvar), 只取 mu
            if out.ndim > 1 and out.shape[-1] >= 2:
                out = out[..., 0]
            outs.append(out.detach().cpu().numpy() if to_numpy
                        else out.detach())
    if to_numpy:
        return np.concatenate(outs).flatten()
    return outs


# ======================================================================
# 6) 系统信息日志
# ======================================================================
def log_system(device=None):
    """打印硬件 + 环境信息, 方便 reproduce / debug"""
    print("─" * 60)
    print(f"  系统    : {platform.platform()}")
    print(f"  Python  : {sys.version.split()[0]}")
    try:
        import torch
        print(f"  PyTorch : {torch.__version__}")
        if device is None:
            device = auto_device(verbose=False)
        print(f"  Device  : {device}")
        if device.type == 'mps':
            print(f"           MPS built = {torch.backends.mps.is_built()}, "
                  f"avail = {torch.backends.mps.is_available()}")
    except ImportError:
        pass
    try:
        # NumPy BLAS 后端
        cfg_txt = str(np.show_config(mode='dicts') if hasattr(np, 'show_config')
                      else '')
        if 'accelerate' in cfg_txt.lower() or 'veclib' in cfg_txt.lower():
            blas = 'Accelerate / veclib (Apple 优化)'
        elif 'openblas' in cfg_txt.lower():
            blas = 'OpenBLAS'
        elif 'mkl' in cfg_txt.lower():
            blas = 'Intel MKL'
        else:
            blas = '未知'
        print(f"  NumPy   : {np.__version__}  BLAS = {blas}")
    except Exception:
        pass
    for v in ['VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS']:
        print(f"  {v:<25s}= {os.environ.get(v, '(未设置)')}")
    print("─" * 60)
