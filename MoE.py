import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import mlflow
import tiktoken
from tqdm import tqdm
import numpy as np
import collections
import itertools
import random
import math
import pronouncing
import re

from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载预训练模型
model_name = "gpt2"
pretrained_lm = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

#------------------------------------------------------------------------------
# 超参数设置
n_embd = 192 # 嵌入维度
n_head = 6 # 注意力头的数量
n_layer = 8 # Transformer 层的数量
head_size = n_embd // n_head # 每个注意力头的大小
dropout = 0.3 # Dropout 比例
block_size = 128 # 模型处理的最大序列长度
num_experts = 6 # MoE 中专家的数量
top_k = 2 # 在 MoE 中，每个输入选择的专家数量
vocab_size = 50257 # 词汇表大小，表示模型可以处理的单词数量
batch_size = 64  # 每个批次的样本数量
max_iters = 6200  # 最大训练迭代次数
eval_interval = 100  # 每隔多少次迭代进行一次评估
eval_iters = 100  # 评估时使用的迭代次数
learning_rate = 5e-5  # 学习率
device = 'cuda' if torch.cuda.is_available() else 'cpu' # 训练设备

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

enc = tiktoken.get_encoding('gpt2') # 使用 GPT-2 的编解码器
tokens = enc.encode(text)
tokens = torch.tensor(tokens)
n = int(0.9*len(tokens))
train_data = tokens[:n]
val_data = tokens[n:]


# 注意力头模块
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        # 三个线性变换层（分别为键、查询、值）
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        # 注册一个下三角矩阵 tril，用于实现因果注意力
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        # 定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape # 获取输入张量 x 的形状(批量大小、时间步长、特征维度)
        k = self.key(x) # 通过键变换层，得到键张量 k
        q = self.query(x) # 通过查询变换层，得到查询张量 q
        wei = q @ k.transpose(-2,-1) * C**-0.5 # 计算注意力权重矩阵 wei，即查询和键的点积，并乘以缩放因子 C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # 将上三角部分的权重设置为负无穷
        wei = F.softmax(wei, dim=-1) # 对权重矩阵进行 Softmax 归一化
        wei = self.dropout(wei) # 应用 Dropout，防止过拟合
        v = self.value(x) # 通过值变换层，得到值张量 v
        out = wei @ v # 根据权重矩阵 wei 对值张量 v 进行加权求和
        return out
    
# 多头注意力模块
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        # 模块列表，每个头独立计算注意力
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # 线性变换层，将多个头的输出拼接后投影到目标维度
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        # 将拼接后的输出通过投影层，将其维度转换为 (B, T, n_embd)并应用dropout
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
# Transformer块
class Block(nn.Module):
    def __init__(self, n_embd, n_head, num_experts, top_k):
        super().__init__()
        # 确保初始top_k有效
        valid_top_k = min(top_k, num_experts)
        self.smoe = SparseMoE(n_embd, num_experts, valid_top_k)
        head_size = n_embd // n_head # 计算每个注意力头的大小
        self.sa = MultiHeadAttention(n_head, head_size) # 多头注意力模块
        self.smoe = SparseMoE(n_embd, num_experts, top_k) # 稀疏混合专家模块
        # 定义两个层归一化（LayerNorm）模块，用于稳定训练
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        # 随机深度概率
        self.drop_path_prob = 0.1

    def forward(self, x):
        # 对SA和MoE应用随机深度
        if self.training and torch.rand(1) < self.drop_path_prob:
            x = x + 0.7 * self.sa(self.ln1(x))
        else:
            x = x + self.sa(self.ln1(x))
        if self.training and torch.rand(1) < self.drop_path_prob:
            x = x + 0.7 * self.smoe(self.ln2(x))
        else:
            x = x + self.smoe(self.ln2(x))
        return x

    
# 专家模块
class Expert(nn.Module):
    def __init__(self, n_embd, expert_type='simple'):
        super().__init__()
        # 一个简单的前馈网络
        if expert_type == 'deep': # 复杂专家
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Linear(4 * n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(0.4)
            )
        else: # 简单专家
            self.net = nn.Sequential(
                nn.Linear(n_embd, 4 * n_embd),
                nn.GELU(),
                nn.Dropout(0.4),
                nn.Linear(4 * n_embd, n_embd),
                nn.Dropout(0.4)
            )
        self.layer_norm = nn.LayerNorm(n_embd)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))
    
# 路由模块
class TopkRouter(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k # 选择的专家数量
        self.linear = nn.Linear(n_embd, num_experts) # 线性变换层，将输入映射到专家数量的维度

    def forward(self, mh_output):
        logits = self.linear(mh_output) # 得到每个输入对每个专家的 logits
        top_k_logits, indices = logits.topk(self.top_k, dim=-1) # 选择 top-k 个专家的 logits 和对应的索引
        zeros = torch.full_like(logits, float('-inf')) # 创建一个与 logits 形状相同的张量
        sparse_logits = zeros.scatter(-1, indices, top_k_logits) # 将 top-k 的 logits 填充到负无穷张量中，其余位置保持负无穷
        router_output = F.softmax(sparse_logits, dim=-1) # 对稀疏 logits 应用 Softmax，得到每个输入对每个专家的权重
        return router_output, indices # 返回路由权重和选择的专家索引

# 定义噪声路由模块
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k
        self.topkroute_linear = nn.Linear(n_embd, num_experts) # 用于计算路由
        self.noise_linear = nn.Linear(n_embd, num_experts) # 用于生成噪声
        self.temperature = nn.Parameter(torch.tensor(0.8))  # 改为可学习参数
        self.temperature_upper = 2.0  # 温度上限
        self.temperature_lower = 0.5  # 温度下限
        # 使用更小的初始化方差
        init.normal_(self.topkroute_linear.weight, mean=0.0, std=0.02)
        init.constant_(self.topkroute_linear.bias, -3.0)  # 偏置初始化
        init.normal_(self.noise_linear.weight, mean=0.0, std=0.02)
        
    def update_top_k(self, new_top_k):
        self.top_k = new_top_k  # 新增方法用于更新top_k
        
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)  # 综合路由分数
        noise_logits = self.noise_linear(mh_output)
        noise = torch.randn_like(logits) * F.softplus(noise_logits) # 从正态分布中随机生成噪声并与正数的缩放因子相乘
        # 在噪声计算中加入温度退火
        # 动态温度约束
        clamped_temp = torch.clamp(
            self.temperature, 
            self.temperature_lower, 
            self.temperature_upper
        )
        noise_logits = logits + clamped_temp * noise # 将噪声添加到路由 logits 中

        top_k_logits, indices = noise_logits.topk(self.top_k, dim=-1) # 对带噪声的 logits 选择 top-k 个专家
        zeros = torch.full_like(noise_logits, float('-inf')) # 创建一个与 logits 形状相同的张量
        sparse_logits = zeros.scatter(-1, indices, top_k_logits) # 将 top-k 的 logits 填充到负无穷张量中，其余位置保持负无穷
        router_output = F.softmax(sparse_logits, dim=-1) # 对 sparse_logits 应用 Softmax
        return router_output, indices # 返回路由权重和选择的专家索引
    
# 稀疏混合专家模块
class SparseMoE(nn.Module):
    def __init__(self, n_embd, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embd, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embd, 'deep' if i < 2 else 'simple')
                                       for i in range(num_experts)])
        self.top_k = top_k
        self.num_experts = num_experts
        self.aux_loss_weight = 0.2 # 辅助损失权重
        self.aux_loss = torch.tensor(0.0, device=device) # 辅助损失
        # 新增负载下限约束
        self.min_expert_ratio = 0.05  # 每个专家至少处理5%的Token
        # 将累计统计改为滑动窗口
        self.window_size = 500
        self.register_buffer('count_buffer', torch.zeros(self.window_size, num_experts))
        self.register_buffer('window_tokens', torch.zeros(()))
        self.register_buffer('pointer', torch.tensor(0))

    def adjust_top_k(self, new_top_k):
        # 确保新top_k有效
        new_top_k = max(1, min(new_top_k, self.num_experts))
        # 检查元素总数是否可整除
        total_elements = self.count_buffer.numel()
        if total_elements % new_top_k != 0:
            new_top_k = self._find_valid_top_k(total_elements)
        if self.top_k != new_top_k:
            # 同步更新
            self.top_k = new_top_k
            self.router.top_k = new_top_k 

    def _find_valid_top_k(self, total):
        # 寻找能整除total的最大合法top_k
        for k in range(min(self.num_experts, total), 0, -1):
            if total % k == 0:
                return k
        return 1

    def forward(self, x):
        B, T, C = x.shape
        gating, indices = self.router(x)  # gating形状 [B, T, num_experts], indices形状 [B, T, top_k]
        # 动态计算reshape参数
        total_elements = indices.numel()
        new_top_k = self.top_k  # 获取当前动态调整后的top_k

        # 确保元素总数可被top_k整除
        if total_elements % new_top_k != 0:
            # 自动调整到最近的合法值
            new_top_k = self._find_valid_top_k(total_elements)
            self.top_k = new_top_k  # 更新当前top_k
            self.router.top_k = new_top_k  # 同步路由模块


        # 展平处理
        gating_flat = gating.view(-1, self.num_experts)  # [B*T, num_experts]
        indices_flat = indices.view(-1, self.top_k)    # [B*T, top_k]

        # 构造专家掩码 [B*T, num_experts]
        # 确保索引不超过num_experts-1
        indices_flat = torch.clamp(indices_flat, 0, self.num_experts-1)
        expert_mask = torch.zeros(B*T, self.num_experts, device=x.device)
        expert_mask.scatter_(1, indices_flat, 1)  # 仅在top_k位置标记为1

        # 滑动窗口更新逻辑
        current_counts = torch.bincount(indices.view(-1), minlength=self.num_experts)
        self.count_buffer[self.pointer] = current_counts
        self.pointer = (self.pointer + 1) % self.window_size
        self.window_tokens += B * T

        # 计算窗口内专家利用率
        window_denominator = (self.window_size * (B*T) / self.num_experts) + 1e-6
        expert_ratio = self.count_buffer.sum(dim=0) / window_denominator
        self.expert_usage = (expert_ratio > 0.1).float().mean()
    

        # 计算负载均衡损失（仅用KL散度）
        window_counts = self.count_buffer.sum(dim=0)
        expert_probs = (window_counts + 1e-3) / (window_counts.sum() + 1e-3)
        expert_probs = torch.clamp(expert_probs, min=self.min_expert_ratio)  # 添加下限约束
        expert_probs = torch.sqrt(expert_probs)  # 平方根平滑分布
        # 改为弹性分布
        load_balance_factor = torch.sigmoid(5*(expert_probs - 0.1))  # 0.1为理想负载
        expert_probs = expert_probs * load_balance_factor
        # 重新归一化
        expert_probs = expert_probs / expert_probs.sum()

        uniform_dist = torch.ones_like(expert_probs) / self.num_experts
        self.aux_loss = F.kl_div(
            torch.log(expert_probs + 1e-10), 
            uniform_dist, 
            reduction='batchmean'
        ) * self.aux_loss_weight
        
        # 仅保留top_k专家的权重，其余置零
        masked_gating = gating_flat * expert_mask  # [B*T, num_experts]

        # 展平输入
        flat_x = x.view(-1, C)

        # 并行计算所有专家输出 [B*T, num_experts, C]
        flat_x = x.view(-1, C)
        expert_outputs = torch.stack([expert(flat_x) for expert in self.experts], dim=1)

        # 加权求和（仅激活top_k专家）
        final_output = (expert_outputs * masked_gating.unsqueeze(-1)).sum(dim=1)  # [B*T, C]
        
        # 恢复形状
        return final_output.view(B, T, C)
            
# 语言模型
class SparseMoELanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # 词嵌入表，将单词索引映射到嵌入向量
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # 位置嵌入表，为每个位置添加位置信息
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, num_experts=num_experts,top_k=top_k) for _ in range(n_layer)]) # 模块序列
        self.ln_f = nn.LayerNorm(n_embd) # 最终的层归一化模块
        self.lm_head = nn.Linear(n_embd, vocab_size) # 线性变换层，将嵌入向量映射到词汇表大小的维度，用于生成下一个单词的概率分布

        # 价值网络
        self.value_network = nn.Sequential(
            nn.Linear(n_embd, 512),
            nn.ReLU(),
            nn.LayerNorm(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # 将输入索引通过词嵌入表，得到词嵌入
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # 为每个位置生成位置嵌入
        x = tok_emb + pos_emb # 将词嵌入和位置嵌入相加，得到输入张量
        aux_loss_total = 0.0
        for block in self.blocks:
            x = block(x)
            aux_loss_total += block.smoe.aux_loss * block.smoe.aux_loss_weight
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C) # 将 logits 展平为 (B * T, C)
            targets = targets.view(B*T) # 将目标标签展平为 (B * T)
            loss = F.cross_entropy(logits, targets, label_smoothing=0.1) # 计算交叉熵损失

        # PPO
        with torch.no_grad(): # 冻结原有模型参数
            x = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(idx.size(1), device=device))
            x += pos_emb
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
        
        # 仅训练价值网络
        values = self.value_network(x).squeeze(-1) # [B, T]
        return logits, loss, aux_loss_total, values
        
    def generate(self, idx, max_new_tokens, top_p=0.95, temperature=0.8):
         # 新增温度退火逻辑
        base_temp = temperature
        for i in tqdm(range(max_new_tokens), desc='Generate Text', unit='iter'):
            # 余弦退火温度：
            current_temp = base_temp * (0.5 + 0.5 * math.cos(math.pi * i / max_new_tokens))
            # 裁剪输入序列，确保不超过 block_size
            idx_cond = idx[:, -block_size:] if idx.size(1) > block_size else idx
            
            # 获取模型输出
            logits, _, _, _ = self(idx_cond)
            
            # 取最后一个时间步的 logits
            logits = logits[:, -1, :] / current_temp  # 应用温度参数

            # 计算概率分布
            probs = F.softmax(logits, dim=-1)
            
            # Top-p 采样
            sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            # 将不需要的 token 的概率设为 0
            indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            
            # 重新归一化概率分布
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # 添加多样性奖励
            if len(idx[0]) > 10:
                last_tokens = idx[0,-10:].tolist()
                diversity = len(set(last_tokens))/10
                probs *= (0.9 + 0.1*diversity)  # 多样性奖励

            # 从剩余 token 中采样
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # 将生成的 token 添加到输入序列中
            idx = torch.cat([idx, idx_next], dim=-1)

        return idx
    

"""----------------------------------------------------------------------------------"""

class PPOTuner:
    def __init__(self, model, ppo_params):
        self.model = model
        self.optimizer = torch.optim.AdamW(
            [
                {'params': model.value_network.parameters(), 'lr':3e-6},
                {'params': model.lm_head.parameters(), 'lr':2e-5}, # 仅微调最后一层
                {'params': model.blocks.parameters(), 'lr': 1e-5}  # 新增block层微调
            ],
            lr=ppo_params['lr'],
            weight_decay=0.01  # 添加权重衰减
        )
        self.scheduler = torch.optim.lr_scheduler.CyclicLR(
                                                    self.optimizer, 
                                                    base_lr=1e-6, 
                                                    max_lr=3e-6,
                                                    step_size_up=200,
                                                    cycle_momentum=False
                                                )
        device = model.lm_head.weight.device  # 获取模型所在设备
        self.clip_epsilon = ppo_params['clip_epsilon'] # PPO 的裁剪范围（，用于限制策略更新的幅度
        self.gamma = ppo_params['gamma'] # 折扣因子，用于计算未来奖励的折现值
        self.lam = ppo_params['lambda'] # 用于计算广义优势估计(GAE)的参数
        self.entropy_coef = ppo_params['entropy_coef'] # 熵正则化项的权重，用于鼓励策略的探索性
        self.kl_coef = nn.Parameter(torch.tensor(ppo_params['kl_coef'], device=device))  # KL 散度的权重
        self.ppo_epochs = ppo_params['ppo_epochs'] #  每次更新时的训练轮数
        # 新增KL动态调整机制
        self.kl_target = torch.tensor(0.02, device=device)  # kl目标值
        # 新增奖励平滑参数
        self.reward_ema = 0.0
        self.reward_alpha = 0.95  # 平滑系数
    
    # 计算广义优势估计
    def compute_advantages(self, rewards, values):
        # 应用EMA平滑
        self.reward_ema = self.reward_alpha * self.reward_ema + (1 - self.reward_alpha) * rewards.mean()
        stabilized_rewards = rewards - (rewards.mean() - self.reward_ema)
        batch_size, seq_len = rewards.size()
        advantages = torch.zeros_like(stabilized_rewards)
        last_gae = 0

        # 并行化GAE计算
        deltas = rewards[:, :-1] + self.gamma * values[:, 1:] - values[:, :-1] # 每个时间步的即时优势，即实际奖励与估计价值的差值
        for t in reversed(range(seq_len-1)): # 从最后一个时间步向前计算
            # 当前时间步的即时优势加上考虑未来时间步的累积优势
            last_gae = deltas[:, t] + self.gamma * self.lam * last_gae
            advantages[:, t] = last_gae # 存储每个时间步的最终优势估计
        return advantages
    
    # PPO微调训练器
    def ppo_step(self, old_logprobs, states, actions, rewards):
        for _ in range(self.ppo_epochs):
            # 生成新策略
            logits, _, _, values = self.model(states)  # logits: [B, T, vocab_size]
            B, T = actions.shape

            # 调整动作索引维度
            action_indices = actions.unsqueeze(-1)  # [B, T] -> [B, T, 1]

            # 计算新策略的概率
            new_logprobs = torch.log_softmax(logits, dim=-1)  # [B, T, vocab_size]
            # 获取动作对应的对数概率（仅用于策略损失）
            action_logprobs = new_logprobs.gather(-1, action_indices).squeeze(-1)  # [B, T]

            # 熵正则化(用于鼓励探索性)
            entropy = - (new_logprobs.exp() * new_logprobs).sum(-1).mean() # sum(-1)沿动作维度求和

            old_action_logprobs = old_logprobs.gather(-1, actions.unsqueeze(-1)).squeeze() # [B, T]
            # 计算概率比值
            log_ratio = action_logprobs - old_action_logprobs  # [B, T]
            ratio = log_ratio.exp()

            # 应用EMA平滑
            self.reward_ema = self.reward_alpha * self.reward_ema + (1 - self.reward_alpha) * rewards.mean()
            stabilized_rewards = rewards - (rewards.mean() - self.reward_ema)
            # 计算优势估计
            advantages = self.compute_advantages(stabilized_rewards, values) 
            # 策略损失
            surr1 = ratio * advantages # 无裁剪的目标
            # 裁剪后的目标(防止策略更新过快)
            surr2 = torch.clamp(ratio, 1-self.clip_epsilon, 1+self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() # 取负因为优化器是梯度下降

            # 时间差分误差
            value_pred = values[:, :-1]
            value_target = rewards[:, :-1] + self.gamma * values[:, 1:].detach()
            value_loss = F.mse_loss(value_pred, value_target)

            # 增加KL散度惩罚
            kl_div = (old_logprobs.exp() * (old_logprobs - new_logprobs)).sum(-1).mean()

            # 动态调整KL系数
            kl_error = (kl_div.detach() - self.kl_target)
            self.kl_coef.data += 0.02 * torch.sigmoid(kl_error/0.1) - 0.01  # S型自适应调整
            self.kl_coef.data.clamp_(0.005, 0.25) # 限制系数范围
            
            kl_penalty = self.kl_coef * kl_div

            # 总损失
            total_loss = (
                        policy_loss 
                        + 0.5 * value_loss 
                        - self.entropy_coef * entropy 
                        + kl_penalty
                    )

            # 反向传播
            self.optimizer.zero_grad() # 清空之前的梯度
            total_loss.backward() # 对总损失进行反向传播，计算梯度
            # 对梯度进行裁剪，防止梯度爆炸
            for name, param in model.named_parameters():
                if 'expert' in name:
                    torch.nn.utils.clip_grad_norm_(param, 2.0)  # 专家层宽松限制
                elif 'router' in name:
                    torch.nn.utils.clip_grad_norm_(param, 0.5)  # 路由层严格限制
                else:
                    torch.nn.utils.clip_grad_norm_(param, 1.0)

            self.optimizer.step() # 更新模型参数
            self.scheduler.step() # 更新调度器

        return total_loss.item()
    
# 微调流程实现
def ppo_finetune(model, checkpoint_path):
    # 加载预训练权重
    pretrained_model.load_state_dict(torch.load(checkpoint_path, weights_only=False), strict=True)  # 确保加载参数
    
    # 冻结不需要训练的层
    for param in model.parameters():
        param.requires_grad = False

    # 解冻关键组件
    components_to_unfreeze = [
        model.value_network,  # 价值网络
        model.lm_head,        # 注意力层
        model.blocks,          # 所有MoE层
        model.position_embedding_table,  # 新增位置编码
        model.token_embedding_table  # 词嵌入
    ]
    for comp in components_to_unfreeze:
        for param in comp.parameters():
            param.requires_grad = True

    # PPO参数配置
    train_epoch = 30
    plt_loss=[]
    plt_reward=[]
    ppo_params = {
        'clip_epsilon': 0.18, # PPO 的裁剪范围，用于限制策略更新的幅度
        'gamma': 0.95, # 折扣因子，用于计算未来奖励的折现值
        'lambda': 0.92, # 用于计算广义优势估计(GAE)的参数
        'entropy_coef': 0.12, # 熵正则化项的权重，用于鼓励策略的探索性
        'kl_coef': 0.03, # KL 散度的权重
        'ppo_epochs': 3, # 每次更新时的训练轮数
        'lr': 3e-6, # 学习率
    }

    tuner = PPOTuner(model, ppo_params) # 初始化PPO调优器
    # 微调循环
    for epoch in range(train_epoch):
        # 随训练降低 KL 权重
        ppo_params['kl_coef'] = max(0.01, 0.05 * (1 - epoch/train_epoch))  
        # 课程学习
        current_max_len = min(
            block_size, 
            48 + epoch * 4 if epoch < 10 else  # 前10个epoch缓慢增长
            88 + (epoch-10) * 2  # 后续更慢增长
        )
        # 生成样本
        with torch.no_grad(): # 禁用梯度计算
            num_candidates = 2 # 每次生成4个候选
            generated_list = []
            for _ in range(num_candidates):
                states = torch.zeros(1,1).long().to(device)
                generated = model.generate(states, max_new_tokens=current_max_len, top_p=0.95, temperature=1.0)
                generated_list.append(generated)  # 直接保存张量
            # 选择奖励最高的样本
            rewards = [style_reward(decode(g[0].tolist())) for g in generated_list]
            best_idx = np.argmax(rewards)
            best_generated = generated_list[best_idx]
            generated_text = decode(best_generated[0].tolist())
        
        # 计算奖励(莎士比亚风格得分)
        reward = style_reward(generated_text) # [0-1]范围
        rewards = torch.full((1,current_max_len), reward).to(device)

        # 准备数据
        states = best_generated[:, :-1]
        actions = best_generated[:, 1:]

        # 获取旧策略log概率
        with torch.no_grad():
            logits, _, _, _ = model(states)
            old_probs = F.softmax(logits, dim=-1)
            old_logprobs = torch.log(old_probs + 1e-10)
        
        # PPO更新
        loss = tuner.ppo_step(
            old_logprobs,
            states,
            actions,
            rewards
        )

        print(f"Epoch {epoch}: Loss={loss:.4f}, Reward={reward:.4f}")
        plt_loss.append(loss)
        plt_reward.append(reward)

        # 在PPO训练循环中添加：
        if epoch % 2 == 0:
            # 可视化专家激活模式
            expert_activations = torch.stack(
                [block.smoe.count_buffer.float().mean(dim=0) 
                for block in model.blocks]
            ).cpu().numpy()
            
            plt.figure(figsize=(12,6))
            sns.heatmap(expert_activations, annot=True, fmt=".1f")
            plt.savefig(f"PPO_expert_heatmap_epoch{epoch}.png")
            plt.close()
            
            # 记录梯度统计
            grad_norms = [p.grad.norm().item() 
                        for p in model.parameters() if p.grad is not None]
            mlflow.log_metric("max_grad_norm", max(grad_norms), step=epoch)

        # 保存微调后的模型
        if epoch % 10 == 0:
            torch.save(model.state_dict(), "ppo_finetuned.pth")
        
    return plt_loss, plt_reward

"""----------------------------------------------------------------------------------"""


def _init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)



def get_batch(data, block_size, batch_size, device, iter):
    # 检查数据长度是否足够
    assert len(data) > block_size, f"数据长度不足: len(data)={len(data)}, block_size={block_size}"
    
    # 随机生成 batch_size 个起始索引
    idxs = torch.randint(0, len(data) - block_size, (batch_size,))
    
    # 提取输入序列和目标序列
    xb = torch.stack([data[i:i+block_size] for i in idxs])
    yb = torch.stack([data[i+1:i+block_size+1] for i in idxs])
    
    # 移动到设备
    xb = xb.to(device)
    yb = yb.to(device)

    # 动态调整dropping概率
    drop_prob = max(0.05, 0.2 - (iter/max_iters)*0.15)  # 从20%逐渐降至5%
    drop_mask = torch.rand_like(xb.float()) < drop_prob
    xb[drop_mask] = enc.encode(
        "<|endoftext|>", 
        allowed_special={"<|endoftext|>"}  # 允许使用特殊token
    )[0]  # 用特殊标记替换

    
    return xb, yb

# 损失估计函数
def estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size, get_batch, iter):
    model.eval() # 将模型切换到评估模式
    train_losses = [] # 训练集损失
    val_losses = [] # 验证集损失
    with torch.no_grad():  # 确保在评估时不计算梯度
        for _ in range(eval_iters):
            xb, yb = get_batch(train_data, block_size, batch_size, device, iter=iter)
            logits, loss, aux_loss, _ = model(xb, yb) # 将输入传递给模型，计算 logits 和损失
            train_losses.append(loss.item())
        for _ in range(eval_iters):
            xb, yb = get_batch(val_data, block_size, batch_size, device, iter=iter)
            logits, loss, aux_loss, _ = model(xb, yb)
            val_losses.append(loss.item())
    model.train() # 将模型切换回训练模式
    return {"train": torch.tensor(train_losses).mean().item(), "val": torch.tensor(val_losses).mean().item()}

# 解码函数
def decode(ids):
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()  # 如果 ids 是张量，转换为列表
    return enc.decode(ids)  # 使用 tiktoken 解码

def clean_text(text):
    # 新增净化逻辑
    text = re.sub(r'[^\w\s\',.;:!?\-]', '', text)  # 移除非标准标点
    text = re.sub(r'\s+', ' ', text).strip()  # 压缩多余空格
    return text[:2000]  # 限制最大长度

# 语法评分函数
def score(text):
    score = 1 / (calculate_perplexity(text) + 1e-6)
    return score

# 连贯性评分函数
def calculate_perplexity(text):
    inputs = tokenizer(text, return_tensors="pt")
    labels = inputs["input_ids"].clone()
    outputs = pretrained_lm(**inputs, labels=labels)
    loss = outputs.loss
    perplexity = torch.exp(loss).item()
    return perplexity

# 抑扬格检测函数
def check_iambic_pentameter(line):
    # 使用更精确的音节模式检测
    stress_pattern = []
    for word in line.split():
        stresses = pronouncing.stresses_for_word(word)  # 需要安装pronouncing库
        if stresses:
            stress_pattern.extend(stresses[0].replace('2','1'))
    return '1010101010' in ''.join(stress_pattern)  # 匹配五音步

# 莎士比亚风格奖励函数
def style_reward(text):
    text = clean_text(text)  # 先净化文本
    # 增加现代词汇惩罚项
    modern_terms = ['internet', 'computer', 'phone', 'AI']  # 示例禁用词汇
    penalty = sum(text.lower().count(term)*0.05 for term in modern_terms)  # 每个禁用词扣0.2分
    # 风格关键词检索
    shakespeare_terms = [
        'thy', 'thou', 'doth', 'hark', 'wherefore', 'tis',
        'thee', 'hath', 'doth', 'ere', 'forsooth', 'prithee',
        'zounds', 'gramercy', 'marry', 'odds'
    ]
    # 增加韵律权重
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    # 动态权重调整
    line_count = max(len(lines), 1)
    term_weight = min(0.3, 0.1 * line_count)  # 行数越多，关键词权重越低

    # 风格关键词评分
    term_count = sum(text.lower().count(term) for term in shakespeare_terms)

    # 调用语法评分函数
    syntax_score = score(text)  

    # 押韵检测（检查押韵模式ABAB/AABB）
    rhyme_score = 0
    rhyme_patterns = {
        'ABAB': [(0,2), (1,3)], 
        'AABB': [(0,1), (2,3)]
    }
    for pattern_name, pairs in rhyme_patterns.items():
        if len(lines) >= 4:
            pattern_score = 0 
            for i,j in pairs:
                w1 = lines[i].split()[-1].lower()
                w2 = lines[j].split()[-1].lower()
                if len(pronouncing.rhymes(w1))>0 and w2 in pronouncing.rhymes(w1):
                    pattern_score  += 2  # 模式匹配奖励加倍
            rhyme_score += pattern_score * 0.5  # 按模式权重加成

    # 增加重复惩罚项（防止重复短语）
    unique_phrases = len(set([line[:20] for line in lines]))
    repetition_penalty = max(0, 0.1*(len(lines)-unique_phrases))

    # 五音步检测
    iambic_score = sum(
        2.0 if check_iambic_pentameter(line) else 0.5 
        for line in lines
    ) / line_count

    # 句子长度奖励
    avg_len = sum(len(line.split()) for line in lines) / line_count
    length_reward = 1 - abs(avg_len-8)/8  # 鼓励每行8词左右

    # 调整综合评分权重
    return (
        0.35 * (1/calculate_perplexity(text)) +  # 连贯性
        0.35 * (rhyme_score/line_count) +        # 押韵（权重提升）
        0.15 * syntax_score +                    # 语法
        0.15 * iambic_score +                   # 韵律
        0.1 * term_weight * (term_count/line_count) +  # 关键词
        0.05 * length_reward - # 句子长度
        penalty * 2 - # 现代词汇
        repetition_penalty * 3 # 重复惩罚
    )

# 早停类
class EarlyStopping:
    def __init__(self, patience=6, delta=0.02):
        self.patience = patience
        self.delta = delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
"""-------------------------------------------------------------------------------------------
----------------------------------------------------------------------------------------------"""

model = SparseMoELanguageModel()
model.apply(_init_weights) 

m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters') #  打印模型的参数数量

max_lr = learning_rate # 最大学习率
min_lr = max_lr * 0.1 # 最小学习率
warmup_steps = max_iters*0.1 # 学习率预热步数

# 使用 AdamW 优化器初始化模型参数
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=max_lr, 
    betas=(0.9, 0.95), 
    eps=1e-8,
    weight_decay=0.1  # 新增权重衰减
)

first_step = 2000
# 改用余弦退火+热重启调度器
scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=first_step, eta_min=1e-4),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters-first_step, eta_min=1e-5)
    ],
    milestones=[2000] # 在2000步切换
)
checkpoint_path = "model_checkpoint.pth" # 定义模型检查点的保存路径
import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description="Sparse MoE Language Model")
parser.add_argument("--train", action="store_true", help="Train the model")
parser.add_argument("--generate", action="store_true", help="Generate text using the trained model")
parser.add_argument("--rlhf", action="store_true", help="make PPO fine-tuning")
parser.add_argument("--ftgenerate", action="store_true", help="Generate text using the PPO fine-tuning model ")
args = parser.parse_args()

import matplotlib.pyplot as plt
import seaborn as sns

# 添加损失值记录
train_losses = []  # 记录训练集损失
val_losses = []    # 记录验证集损失
from torch.amp import GradScaler
scaler = GradScaler()
lr_history = []
ready_iter = 0
# 训练逻辑
if args.train:
    print("Starting training...")
    # 先冻结所有参数
    for param in model.parameters():
        param.requires_grad = False
    # 仅解冻非 value_network
    for block in model.blocks:
        for param in block.parameters():
            param.requires_grad = True
    for param in model.token_embedding_table.parameters():
        param.requires_grad = True
    for param in model.position_embedding_table.parameters():
        param.requires_grad = True
    for param in model.lm_head.parameters():
        param.requires_grad = True

    # 早停机制
    early_stopping = EarlyStopping(patience=5, delta=0.01)

    with mlflow.start_run():
        params = {
            "batch_size": batch_size,
            "block_size": block_size,
            "max_iters": max_iters,
            "eval_interval": eval_interval,
            "learning_rate": learning_rate,
            "device": device,
            "eval_iters": eval_iters,
            "dropout": dropout,
            "num_experts": num_experts,
            "top_k": top_k
        }
        mlflow.log_params(params)  # 将超参数记录到 mlflow
        for iter in tqdm(range(max_iters), desc="Training", unit="iter"):
            current_lr = optimizer.param_groups[0]['lr']
            lr_history.append(current_lr) # 记录学习率

            optimizer.zero_grad() # 清空梯度
            xb, yb = get_batch(train_data, block_size, batch_size, device, iter=iter)  # 从训练集中采样一批数据
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits, loss, aux_loss, values = model(xb, yb)
                total_loss = loss + aux_loss * model.blocks[0].smoe.aux_loss_weight  # 动态权重
                total_loss = total_loss # 添加损失缩放因子
            if total_loss.dim() > 0:
                total_loss = total_loss.mean()  
            scaler.scale(total_loss).backward()  # 反向传播，计算梯度

            scaler.unscale_(optimizer)
            if iter < max_iters//2:  # 分阶段裁剪
                max_norm = 1.0 + (iter/(max_iters//2))*1.0  # 前半段1.0->2.0
            else:
                max_norm = 2.0 + (iter/(max_iters//2)-1)*0.5  # 后半段2.0->2.5
            torch.nn.utils.clip_grad_norm_(
                                            model.parameters(), 
                                            max_norm=max_norm,
                                            norm_type=2.0  # 使用L2范数更稳定
                                        )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5

            # 评测模型
            if iter % eval_interval == 0 or iter == max_iters - 1:
                ready_iter += 1  
                # 计算loss值
                losses = estimate_loss(model, eval_iters, device, train_data, val_data, block_size, batch_size, get_batch, iter)
                print(f"Iter {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # 打印专家利用率
                expert_usage = torch.stack([block.smoe.expert_usage for block in model.blocks])
                avg_usage = expert_usage.mean()
                print(f"Expert Usage: {avg_usage:.2%}")

                # 打印梯度范数
                print(f"Gradient Norm: {total_norm:.2f}")
                # 记录损失值
                train_losses.append(losses['train'])
                val_losses.append(losses['val'])

                # 打印当前学习率
                print(f"Iter {iter}: Current LR = {optimizer.param_groups[0]['lr']}")

                # 记录loss值
                mlflow.log_metric("train_loss", losses['train'], step=iter)
                mlflow.log_metric("val_loss", losses['val'], step=iter)

                # 早停检查
                early_stopping(losses['val'], model)
                if early_stopping.early_stop:
                    torch.save(model.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved to {checkpoint_path}")
                    print("Early stopping triggered.")
                    break


            # 绘制热力图
            if iter % (eval_interval*3) == 0:  
                # 确保expert_counts是二维数组
                expert_counts = torch.stack(
                    [block.smoe.count_buffer.sum(dim=0).cpu() for block in model.blocks]  # 使用滑动窗口累计值
                ).float().numpy()
                
                # 检查形状是否为 (n_layer, num_experts)
                if expert_counts.ndim == 1:
                    expert_counts = expert_counts.reshape(-1, 1)  # 若为一维则转为二维
                
                plt.figure(figsize=(10,6))
                sns.heatmap(expert_counts, annot=True, fmt='.1f')  # 添加fmt参数确保数值格式
                plt.savefig(f"expert_heatmap_{iter}.png")
                plt.close()
                
            expert_usage = torch.stack([block.smoe.expert_usage for block in model.blocks])
            avg_usage = expert_usage.mean()
            # 动态调整top_k 
            if avg_usage > 0.95:
                for block in model.blocks:
                    block.smoe.adjust_top_k(block.smoe.top_k - 1)
            elif avg_usage < 0.8:
                for block in model.blocks:
                    block.smoe.adjust_top_k(block.smoe.top_k + 1)

            if iter % (eval_interval * 10) == 0 or iter == max_iters - 1:
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        
        # 训练完成后绘制损失曲线
        iterations = [i for i in range(0,ready_iter*eval_interval,eval_interval)]
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, train_losses, label="Train Loss", color="blue")
        plt.plot(iterations, val_losses, label="Validation Loss", color="red")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.savefig("loss_curve.png")  # 保存图像
        plt.show()  # 显示图像

        # 训练完成后绘制学习率曲线
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(lr_history)), lr_history, label="Learning Rate", color="blue")
        plt.xlabel("Iterations")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.legend()
        plt.grid(True)
        plt.savefig("lr_curve.png")  # 保存图像
        plt.show()  # 显示图像
# 推理逻辑
if args.generate:
    print("Loading model weights and generating text...")
    # 加载权重
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式

    # 生成文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=250, top_p=0.9, temperature=0.8)[0]
    generated_text = decode(generated_tokens)
    print(generated_text)
    output_file = "generated_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")


# 微调逻辑
if args.rlhf:
    print("Loading model weights and make RLHF fine-tuning.")
    # 加载预训练模型
    pretrained_model = SparseMoELanguageModel().to(device)
    
    # 执行PPO微调
    plt_loss, plt_reward = ppo_finetune(pretrained_model, checkpoint_path)

    # 生成测试
    context = torch.zeros((1,1), dtype=torch.long, device=device)
    generated = pretrained_model.generate(context, 250)
    generated_text = decode(generated[0].tolist())
    print(generated_text)
    output_file = "generated_RLHF_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text with RLHF saved to {output_file}")

    # 绘制 loss 和 reward 曲线
    plt.figure(figsize=(12, 6))
    
    # 绘制 loss 曲线
    plt.subplot(1, 2, 1)
    plt.plot(plt_loss, label='Loss', color='blue', linestyle='-')
    plt.title('Training Loss', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    # 绘制 reward 曲线
    plt.subplot(1, 2, 2)
    plt.plot(plt_reward, label='Reward', color='green', linestyle='-')
    plt.title('Reward', fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    # 调整布局
    plt.tight_layout()
    # 保存图表
    plt.savefig('ppo_finetuning_results.png')
    print("Training results plot saved to 'ppo_finetuning_results.png'")
    # 显示图表
    plt.show()

if args.ftgenerate:
    ppo_checkpoint = 'ppo_finetuned.pth'
    print("Loading PPO-Fine tuning model weights and generating text...")

    model.load_state_dict(torch.load(ppo_checkpoint, map_location=device, weights_only=True))
    model.eval()  # 切换到评估模式
    # 生成文本
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated_tokens = model.generate(context, max_new_tokens=250, top_p=0.9, temperature=0.8)[0]
    generated_text = decode(generated_tokens)
    print(generated_text)
    output_file = "generated_text.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text)
    print(f"Generated text saved to {output_file}")
