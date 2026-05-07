"""
DRL (Deep Reinforcement Learning) 策略基线 —— SAC
=================================================
P0 需求: 至少添加 1 个 DRL 基线 (推荐 SAC, 因为连续动作)

设计:
  State  s_t = [ SoC_t, norm(pred_demand_t), norm(price_t), norm(renewable_t),
                 norm(carbon_intensity_t), hour_sin, hour_cos, dow_sin, dow_cos ]
  Action a_t ∈ [-1, 1]
         -1 → 最大充电率 ; +1 → 最大放电率
  Reward r_t = - (electricity_cost + α_CO2 · co2_cost + β_peak · peak_excess)

为保持依赖最小, 本文件仅需 PyTorch, 无需 stable-baselines3 / gym.
训练完的 actor 可序列化, 作为 DRLSACStrategy 被调用。

参考:
  Haarnoja et al. 2018 "Soft Actor-Critic: Off-Policy Maximum Entropy Deep RL"
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
# 网络
# ======================================================================
class _ActorSAC(nn.Module):
    """高斯策略, 输出 μ, log_σ → 重参数化 → tanh 压缩到 [-1, 1]"""
    def __init__(self, state_dim, hidden=128, action_dim=1):
        super().__init__()
        self.l1 = nn.Linear(state_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.mu_head     = nn.Linear(hidden, action_dim)
        self.logstd_head = nn.Linear(hidden, action_dim)

    def forward(self, s):
        h = F.relu(self.l1(s))
        h = F.relu(self.l2(h))
        mu     = self.mu_head(h)
        logstd = self.logstd_head(h).clamp(-5.0, 2.0)
        return mu, logstd

    def sample(self, s):
        mu, logstd = self(s)
        std = logstd.exp()
        eps = torch.randn_like(mu)
        u = mu + std * eps
        a = torch.tanh(u)
        # log_prob 修正: a = tanh(u) → log π(a|s) = log π(u|s) - Σ log(1 - a^2)
        log_prob = (-0.5 * (eps ** 2) - logstd - 0.5 * np.log(2 * np.pi)).sum(-1, keepdim=True)
        log_prob -= torch.log(1 - a.pow(2) + 1e-6).sum(-1, keepdim=True)
        return a, log_prob, torch.tanh(mu)


class _CriticSAC(nn.Module):
    """双 Q 网络 (Clipped Double Q)"""
    def __init__(self, state_dim, hidden=128, action_dim=1):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1))

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


# ======================================================================
# 回放缓冲
# ======================================================================
class _ReplayBuffer:
    def __init__(self, capacity, state_dim, action_dim=1):
        self.cap = int(capacity)
        self.s  = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.a  = np.zeros((self.cap, action_dim), dtype=np.float32)
        self.r  = np.zeros((self.cap, 1), dtype=np.float32)
        self.s2 = np.zeros((self.cap, state_dim), dtype=np.float32)
        self.d  = np.zeros((self.cap, 1), dtype=np.float32)
        self.ptr, self.size = 0, 0

    def add(self, s, a, r, s2, d):
        self.s[self.ptr]  = s
        self.a[self.ptr]  = a
        self.r[self.ptr]  = r
        self.s2[self.ptr] = s2
        self.d[self.ptr]  = d
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs, device):
        idx = np.random.randint(0, self.size, size=bs)
        t = lambda x: torch.from_numpy(x[idx]).to(device)
        return t(self.s), t(self.a), t(self.r), t(self.s2), t(self.d)


# ======================================================================
# BESS 仿真环境 (用于训练)
# ======================================================================
class BESSDispatchEnv:
    """
    简化港口储能调度环境, 用于 SAC 训练。
    每步: 给定 BESS 动作 a ∈ [-1, 1], 计算 grid import、cost、CO2, 更新 SoC。
    """
    def __init__(self, data_df, predictions_by_index, price_arr, carbon_ef_arr,
                 capacity_kwh, power_kw, bess_params,
                 demand_charge_rate=0.0,
                 carbon_penalty=0.005,   # CNY / kg CO2 内部化
                 peak_penalty=1.0,       # CNY / kW excess / hour
                 rolling_peak_window=720,  # 近似月度需量窗口
                 ):
        self.df = data_df.reset_index(drop=True)
        self.preds = np.asarray(predictions_by_index, dtype=float)
        self.price = np.asarray(price_arr, dtype=float)
        self.ef    = np.asarray(carbon_ef_arr, dtype=float)
        self.cap   = float(capacity_kwh)
        self.pwr   = float(power_kw)
        self.bp    = bess_params   # dict: min_soc, max_soc, eff_c, eff_d
        self.dc_rate = float(demand_charge_rate)  # CNY/kW/month
        self.carbon_penalty = float(carbon_penalty)
        self.peak_penalty   = float(peak_penalty)
        self.rolling_peak_window = int(rolling_peak_window)

        # 归一化常数
        self._price_max = float(np.max(np.abs(self.price))) or 1.0
        self._ef_max    = float(np.max(np.abs(self.ef))) or 1.0
        self._demand_max = float(np.max(self.df['E_total'].values
                                         if 'E_total' in self.df.columns
                                         else self.df['E_grid'].values)) or 1.0
        # Bug 4 修复: 计算典型每小时成本作为 reward scale, 让 reward 量级 ~ O(1)
        # 典型 = 平均需求 × 平均电价
        avg_demand = float(np.mean(self.df['E_total'].values
                                   if 'E_total' in self.df.columns
                                   else self.df['E_grid'].values))
        avg_price  = float(np.mean(np.abs(self.price)))
        self._reward_scale = max(1.0, avg_demand * avg_price)
        self.T_total = min(len(self.df), len(self.preds)) - 1
        self.reset()

    # ------------------------------------------------------------------
    def state_dim(self):
        return 9

    def action_dim(self):
        return 1

    def _obs(self):
        t = self.t
        re_t = float(self.df.iloc[t].get('E_PV_kWh', 0.0)) + \
               float(self.df.iloc[t].get('E_wind_kWh', 0.0))
        obs = np.array([
            self.soc,                                    # 0-1
            self.preds[min(t, len(self.preds)-1)] / self._demand_max,
            self.price[t] / self._price_max,
            re_t / self._demand_max,
            self.ef[t] / self._ef_max,
            np.sin(2 * np.pi * (t % 24) / 24),
            np.cos(2 * np.pi * (t % 24) / 24),
            np.sin(2 * np.pi * ((t // 24) % 7) / 7),
            np.cos(2 * np.pi * ((t // 24) % 7) / 7),
        ], dtype=np.float32)
        return obs

    def reset(self, t0=0):
        self.t = int(t0)
        self.soc = 0.5
        self.peak_grid = 0.0
        self.recent_grid = []
        return self._obs()

    def step(self, a):
        a = float(np.clip(a, -1.0, 1.0))
        t = self.t
        # 实际需求 & 可再生
        actual_demand = float(self.df.iloc[t].get(
            'E_total', self.df.iloc[t].get('E_grid', 0.0)))
        re_t = float(self.df.iloc[t].get('E_PV_kWh', 0.0)) + \
               float(self.df.iloc[t].get('E_wind_kWh', 0.0))

        # BESS 动作: a > 0 放电, a < 0 充电
        max_p = self.pwr
        if a >= 0:
            # 放电
            avail_energy = (self.soc - self.bp['min_soc']) * self.cap
            d_cmd = a * max_p
            d_actual = min(d_cmd, max_p, max(0.0, avail_energy) * self.bp['eff_d'])
            c_actual = 0.0
            bess_signed = d_actual
            self.soc -= d_actual / max(self.bp['eff_d'], 1e-6) / self.cap
        else:
            # 充电
            room = (self.bp['max_soc'] - self.soc) * self.cap
            c_cmd = -a * max_p
            c_actual = min(c_cmd, max_p, max(0.0, room) / max(self.bp['eff_c'], 1e-6))
            d_actual = 0.0
            bess_signed = -c_actual
            self.soc += c_actual * self.bp['eff_c'] / self.cap

        self.soc = float(np.clip(self.soc, self.bp['min_soc'], self.bp['max_soc']))

        # 电网购电
        grid_t = max(0.0, actual_demand - re_t - bess_signed)
        # 滚动峰值
        self.recent_grid.append(grid_t)
        if len(self.recent_grid) > self.rolling_peak_window:
            self.recent_grid.pop(0)
        peak_now = max(self.recent_grid) if self.recent_grid else 0.0
        peak_excess = max(0.0, grid_t - self.peak_grid)
        self.peak_grid = max(self.peak_grid, grid_t)

        # Reward: 负成本 + 负 CO2 + 负 peak 超额
        # 修复 2.5: 原 demand_charge_inc 仅 peak_excess × 38/30 量级 ≈ 1.27 × peak_excess,
        # 远小于 cost_t (≈ price × Pg) ≈ 1.0 × 50000 = 50000,
        # 所以 SAC 完全 ignore 削峰目标, 在 ±10000 kW 之间乱震荡 (报告 2.5).
        # 现在加入:
        #   (a) running peak 强惩罚: 全周期峰值 × demand_charge / 30 / sim_hours
        #       (注意是绝对峰值的累计影响, 不只是 "本步是否破峰")
        #   (b) round-trip 损耗惩罚 = (1 - eff) × |bess_power|
        #   (c) 终端 SOC 软约束 (在 done=True 时给一次性惩罚)
        cost_t = grid_t * self.price[t]
        co2_t  = grid_t * self.ef[t]   # kg
        demand_charge_inc = peak_excess * self.dc_rate / 30.0

        # 强化 peak 惩罚: 用整周期 running peak 直接惩罚 (放大 5×, 论文中常用做法)
        peak_penalty_t = 5.0 * self.peak_grid * self.dc_rate / 30.0 / max(self.T_total, 1)

        # round-trip 损耗
        rt_loss = (1.0 - self.bp['eff_d']) * abs(bess_signed)
        rt_penalty = rt_loss * float(np.mean(self.price)) * 0.5

        raw_reward = (cost_t
                      + self.carbon_penalty * co2_t
                      + demand_charge_inc
                      + peak_penalty_t
                      + rt_penalty)

        # 缩放到 O(1)
        scale = self._reward_scale if hasattr(self, '_reward_scale') and self._reward_scale > 0 \
                else (self._demand_max * 1.0 + 1.0)
        reward = -raw_reward / scale

        # SOC 边界惩罚: 鼓励保持中等 SOC, 不要锁死在 0.1 或 0.9
        soc_pen = 0.0
        if self.soc >= self.bp['max_soc'] - 0.02:
            soc_pen = 0.05
        elif self.soc <= self.bp['min_soc'] + 0.02:
            soc_pen = 0.05
        reward -= soc_pen

        # 终端 SOC 软约束: episode 结束时若 SOC 远离 0.5, 一次性大惩罚
        if self.t + 1 >= self.T_total:
            terminal_pen = 5.0 * abs(self.soc - 0.5)
            reward -= terminal_pen

        self.t += 1
        done = (self.t >= self.T_total)
        return self._obs(), float(reward), done, {
            'grid': grid_t, 'cost': cost_t, 'co2_kg': co2_t,
            'bess_signed': bess_signed, 'soc': self.soc,
        }


# ======================================================================
# SAC 训练器
# ======================================================================
class SACTrainer:
    def __init__(self, state_dim, action_dim=1, hidden=128,
                 lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 buffer_cap=100000, device=None):
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor  = _ActorSAC(state_dim, hidden, action_dim).to(self.device)
        self.critic = _CriticSAC(state_dim, hidden, action_dim).to(self.device)
        self.critic_tgt = _CriticSAC(state_dim, hidden, action_dim).to(self.device)
        self.critic_tgt.load_state_dict(self.critic.state_dict())

        self.opt_a = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = _ReplayBuffer(buffer_cap, state_dim, action_dim)
        self.state_dim = state_dim

    def select(self, state, deterministic=False):
        s = torch.from_numpy(np.asarray(state, dtype=np.float32)).to(self.device).unsqueeze(0)
        with torch.no_grad():
            a, _, a_det = self.actor.sample(s)
        if deterministic:
            return a_det.cpu().numpy().flatten()
        return a.cpu().numpy().flatten()

    def update(self, batch_size=128):
        if self.buffer.size < batch_size:
            return None
        s, a, r, s2, d = self.buffer.sample(batch_size, self.device)

        # Critic
        with torch.no_grad():
            a2, lp2, _ = self.actor.sample(s2)
            q1_t, q2_t = self.critic_tgt(s2, a2)
            q_t = torch.min(q1_t, q2_t) - self.alpha * lp2
            target = r + self.gamma * (1 - d) * q_t

        q1, q2 = self.critic(s, a)
        loss_c = F.mse_loss(q1, target) + F.mse_loss(q2, target)
        self.opt_c.zero_grad()
        loss_c.backward()
        self.opt_c.step()

        # Actor
        a_pi, lp_pi, _ = self.actor.sample(s)
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi = torch.min(q1_pi, q2_pi)
        loss_a = (self.alpha * lp_pi - q_pi).mean()
        self.opt_a.zero_grad()
        loss_a.backward()
        self.opt_a.step()

        # Soft update target
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_tgt.parameters()):
                pt.data.mul_(1 - self.tau)
                pt.data.add_(self.tau * p.data)

        return float(loss_c.item()), float(loss_a.item())

    def train(self, env, total_steps=5000, warm_up=200, updates_per_step=1,
              batch_size=128, log_every=500, verbose=True):
        s = env.reset()
        ep_ret, ep_len = 0.0, 0
        for step in range(total_steps):
            # Warm-up 期随机动作
            if step < warm_up:
                a = np.random.uniform(-1, 1, size=1).astype(np.float32)
            else:
                a = self.select(s)

            s2, r, done, _ = env.step(a[0])
            self.buffer.add(s, a, r, s2, float(done))
            s = s2
            ep_ret += r
            ep_len += 1

            if step >= warm_up:
                for _ in range(updates_per_step):
                    self.update(batch_size=batch_size)

            if done:
                if verbose and step % log_every == 0:
                    print(f"    [SAC] step={step}, episode_return={ep_ret:.0f}, "
                          f"len={ep_len}")
                s = env.reset()
                ep_ret, ep_len = 0.0, 0

            if verbose and step > 0 and step % log_every == 0:
                print(f"    [SAC] step={step}/{total_steps}, "
                      f"buffer={self.buffer.size}, last_return={ep_ret:.0f}")


# ======================================================================
# 策略包装 (兼容 BaseStrategy 接口)
# ======================================================================
class DRLSACStrategy:
    """
    用训练好的 SAC actor 做滚动时域调度。
    每个 MPC step 只取第 1 步 BESS 动作 (与 MPC 一致), 其余用填充占位。
    """
    name = "DRL-SAC"
    description = "Soft Actor-Critic (continuous action on BESS power)"

    def __init__(self, trainer: SACTrainer, env_template: BESSDispatchEnv,
                 horizon: int = 24):
        self.trainer = trainer
        self.env_ref = env_template
        self.horizon = horizon
        self._t = 0   # 当前时间索引

    def reset(self):
        self._t = 0

    def optimize(self, bess, pred_demand, renewable_gen, grid_prices,
                 carbon_intensity=None, pred_upper=None):
        """返回与 MPC 策略相同格式的 schedule dict"""
        import numpy as np
        H = len(pred_demand)
        charge    = np.zeros(H)
        discharge = np.zeros(H)

        # 构造当前 state
        soc = bess.get_soc()
        t   = self._t
        # 简化观测 (与训练 env 对齐)
        demand_norm = float(pred_demand[0]) / (self.env_ref._demand_max + 1e-6)
        price_norm  = float(grid_prices[0]) / (self.env_ref._price_max + 1e-6)
        re_norm     = float(renewable_gen[0]) / (self.env_ref._demand_max + 1e-6)
        ci_t = float(carbon_intensity[0]) if carbon_intensity is not None else 0.0
        ci_norm = ci_t / (self.env_ref._ef_max + 1e-6)

        state = np.array([
            soc, demand_norm, price_norm, re_norm, ci_norm,
            np.sin(2 * np.pi * (t % 24) / 24),
            np.cos(2 * np.pi * (t % 24) / 24),
            np.sin(2 * np.pi * ((t // 24) % 7) / 7),
            np.cos(2 * np.pi * ((t // 24) % 7) / 7),
        ], dtype=np.float32)

        a = float(self.trainer.select(state, deterministic=True)[0])
        if a >= 0:
            discharge[0] = a * bess.max_power
        else:
            charge[0] = -a * bess.max_power

        self._t += 1
        return {
            'bess_charge':    charge,
            'bess_discharge': discharge,
            'soc_profile':    np.full(H + 1, soc),
            'objective_value': 0.0,
        }
