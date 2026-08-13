import os
import time
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import multiprocessing as mp
from tqdm import tqdm

from model import QuoridorActorCritic
from game_env import QuoridorEnv

# =====================================================================
# MULTIPROCESSING WRAPPER (ОБХОД ПИТОНОВСКОГО GIL)
# =====================================================================
def worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset())
            elif cmd == 'close':
                remote.close()
                break
        except Exception as e:
            print(f"Worker process terminated: {e}")
            break

class MultiprocessEnv:
    def __init__(self, env_fns):
        self.num_envs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[mp.Pipe() for _ in env_fns])
        self.ps = [mp.Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()

    def step(self, actions, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for idx, action in zip(indices, actions):
            self.remotes[idx].send(('step', action))
        # Ожидаем результаты параллельного выполнения
        results = [self.remotes[idx].recv() for idx in indices]
        return zip(*results)

    def reset(self, indices=None):
        if indices is None:
            indices = range(self.num_envs)
        for idx in indices:
            self.remotes[idx].send(('reset', None))
        results = [self.remotes[idx].recv() for idx in indices]
        return results

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
# =====================================================================

def train_ppo():
    # macOS использует 'spawn' по умолчанию, что безопасно для PyTorch CUDA/MPS
    if mp.get_start_method(allow_none=True) != 'spawn':
        mp.set_start_method('spawn', force=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Устройство: {device} | Режим: MULTIPROCESSING CPU + NUMBA")

    # Сбалансированные параметры для максимального FPS
    num_envs = 32
    num_steps = 256
    batch_size = num_envs * num_steps
    minibatch_size = batch_size // 4
    total_timesteps = 10_000_000

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_model.pt"

    # Инициализация параллельных сред
    print("Запуск 32 параллельных процессов...")
    envs = MultiprocessEnv([QuoridorEnv for _ in range(num_envs)])

    agent = QuoridorActorCritic(num_actions=136).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=2.5e-5, eps=1e-5)
    
    opponent_agent = QuoridorActorCritic(num_actions=136).to(device)
    opponent_agent.eval()
    
    opponent_pool = []
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        opponent_pool = checkpoint.get('opponent_pool', [])
        print("--> Загружен существующий чекпоинт.")
    
    if not opponent_pool:
        opponent_pool = [{k: v.cpu().clone() for k, v in agent.state_dict().items()}]

    obs_b = torch.zeros((num_steps, num_envs, 6, 9, 9), device=device)
    actions_b = torch.zeros((num_steps, num_envs), device=device, dtype=torch.long)
    logprobs_b = torch.zeros((num_steps, num_envs), device=device)
    rewards_b = torch.zeros((num_steps, num_envs), device=device)
    dones_b = torch.zeros((num_steps, num_envs), device=device)
    values_b = torch.zeros((num_steps, num_envs), device=device)
    masks_b = torch.zeros((num_steps, num_envs, 136), device=device, dtype=torch.bool)

    # Буферы для следующего шага
    next_obs_arr = np.zeros((num_envs, 6, 9, 9), dtype=np.float32)
    next_mask_arr = np.zeros((num_envs, 136), dtype=np.bool_)
    
    # ИСПРАВЛЕННЫЙ БЛОК: итерируемся напрямую по результатам
    for i, (res_obs, res_info) in enumerate(envs.reset()):
        next_obs_arr[i] = res_obs
        next_mask_arr[i] = res_info['valid_actions_mask']
        
    next_obs = torch.from_numpy(next_obs_arr).float().to(device)
    next_masks = torch.from_numpy(next_mask_arr).to(device)
    next_done = torch.zeros(num_envs, dtype=torch.float32, device=device)

    num_updates = total_timesteps // batch_size
    autocast_dtype = torch.float16 if device.type == 'mps' else torch.float32

    pbar = tqdm(total=num_updates, desc="Обучение PPO", unit="upd", dynamic_ncols=True)

    for update in range(1, num_updates + 1):
        step_start_time = time.time()
        
        progress = (update - 1) / num_updates
        lr_now = 2.5e-5 * (1.0 - progress)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_now

        # ПАТЧ: Сэмплирование исторического оппонента для Self-Play
        if update % 20 == 0:
            opponent_pool.append({k: v.cpu().clone() for k, v in agent.state_dict().items()})
            if len(opponent_pool) > 20: opponent_pool.pop(0)

        if opponent_pool:
            opp_weights = random.choice(opponent_pool)
            opponent_agent.load_state_dict({k: v.to(device) for k, v in opp_weights.items()})

        agent.eval()
        for step in range(num_steps):
            obs_b[step] = next_obs
            masks_b[step] = next_masks
            dones_b[step] = next_done

            with torch.no_grad():
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    action, logprob, _, value = agent.get_action_and_value(next_obs, next_masks)

            values_b[step] = value.flatten()
            actions_b[step] = action
            logprobs_b[step] = logprob

            # 1. Шаг активного агента (Параллельно через Multiprocessing)
            actions_cpu = action.cpu().numpy()
            obs_tup, r_tup, term_tup, trunc_tup, info_tup = envs.step(actions_cpu)
            
            rewards_arr = np.array(r_tup, dtype=np.float32)
            dones_arr = np.array([t1 or t2 for t1, t2 in zip(term_tup, trunc_tup)], dtype=np.bool_)
            
            opp_obs_list = []
            opp_mask_list = []
            opp_indices = []
            
            envs_to_reset = []
            for i in range(num_envs):
                rewards_arr[i] -= 0.002 # Shaping штраф за время
                if dones_arr[i]:
                    envs_to_reset.append(i)
                else:
                    opp_obs_list.append(obs_tup[i])
                    opp_mask_list.append(info_tup[i]['valid_actions_mask'])
                    opp_indices.append(i)
                    
            if envs_to_reset:
                reset_results = envs.reset(envs_to_reset)
                for idx, (res_obs, res_info) in zip(envs_to_reset, reset_results):
                    next_obs_arr[idx] = res_obs
                    next_mask_arr[idx] = res_info['valid_actions_mask']
                    
            for list_idx, env_i in enumerate(opp_indices):
                next_obs_arr[env_i] = opp_obs_list[list_idx]
                next_mask_arr[env_i] = opp_mask_list[list_idx]

            # 2. Шаг исторического оппонента (Self-Play)
            if opp_indices:
                opp_obs_t = torch.from_numpy(np.stack(opp_obs_list)).float().to(device)
                opp_mask_t = torch.from_numpy(np.stack(opp_mask_list)).to(device)
                
                with torch.no_grad():
                    with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                        opp_actions, _, _, _ = opponent_agent.get_action_and_value(opp_obs_t, opp_mask_t)
                
                opp_actions_cpu = opp_actions.cpu().numpy()
                
                # Параллельный шаг оппонента!
                opp_obs_tup, opp_r_tup, opp_term_tup, opp_trunc_tup, opp_info_tup = envs.step(opp_actions_cpu, indices=opp_indices)
                
                opp_envs_to_reset = []
                for list_idx, env_i in enumerate(opp_indices):
                    r = opp_r_tup[list_idx]
                    term = opp_term_tup[list_idx]
                    trunc = opp_trunc_tup[list_idx]
                    
                    if term or trunc:
                        rewards_arr[env_i] -= r
                        dones_arr[env_i] = True
                        opp_envs_to_reset.append(env_i)
                    else:
                        rewards_arr[env_i] += (-r - 0.02)
                        next_obs_arr[env_i] = opp_obs_tup[list_idx]
                        next_mask_arr[env_i] = opp_info_tup[list_idx]['valid_actions_mask']
                        
                if opp_envs_to_reset:
                    reset_results = envs.reset(opp_envs_to_reset)
                    for idx, (res_obs, res_info) in zip(opp_envs_to_reset, reset_results):
                        next_obs_arr[idx] = res_obs
                        next_mask_arr[idx] = res_info['valid_actions_mask']

            rewards_b[step] = torch.from_numpy(rewards_arr).float().to(device)
            next_obs = torch.from_numpy(next_obs_arr).float().to(device)
            next_masks = torch.from_numpy(next_mask_arr).to(device)
            next_done = torch.from_numpy(dones_arr.astype(np.float32)).to(device)

        with torch.no_grad():
            _, _, _, next_value = agent.get_action_and_value(next_obs, next_masks)
            advantages = torch.zeros_like(rewards_b, device=device)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextval = next_value.flatten()
                else:
                    nextnonterminal = 1.0 - dones_b[t + 1]
                    nextval = values_b[t + 1]
                delta = rewards_b[t] + 0.99 * nextval * nextnonterminal - values_b[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_b

        b_obs = obs_b.reshape((-1, 6, 9, 9)).detach()
        b_logprobs = logprobs_b.reshape(-1).detach()
        b_actions = actions_b.reshape(-1).detach()
        b_advantages = ((advantages.reshape(-1) - advantages.mean()) / (advantages.std() + 1e-8)).detach()
        b_returns = returns.reshape(-1).detach()
        b_masks = masks_b.reshape((-1, 136)).detach()

        agent.train()
        b_inds = np.arange(batch_size)
        
        pg_loss_epoch = 0.0
        v_loss_epoch = 0.0
        
        for _ in range(4):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb = b_inds[start:end]

                _, newlogprob, entropy, newval = agent.get_action_and_value(b_obs[mb], b_masks[mb], b_actions[mb])
                ratio = (newlogprob - b_logprobs[mb]).exp()
                pg_loss = torch.max(-b_advantages[mb] * ratio, -b_advantages[mb] * torch.clamp(ratio, 0.9, 1.1)).mean()
                v_loss = nn.functional.smooth_l1_loss(newval.view(-1), b_returns[mb])
                loss = pg_loss - 0.01 * entropy.mean() + 0.5 * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()
                
                pg_loss_epoch += pg_loss.item()
                v_loss_epoch += v_loss.item()

        fps = int(batch_size / (time.time() - step_start_time))
        mean_reward = rewards_b.mean().item()
        avg_pg_loss = pg_loss_epoch / (4 * (batch_size // minibatch_size))
        avg_v_loss = v_loss_epoch / (4 * (batch_size // minibatch_size))

        pbar.set_postfix({
            'FPS': fps, 
            'Rew': f"{mean_reward:.2f}",
            'PLoss': f"{avg_pg_loss:.3f}",
            'VLoss': f"{avg_v_loss:.3f}"
        })
        pbar.update(1)

        if update % 25 == 0:
            torch.save({
                'model_state_dict': agent.state_dict(), 
                'optimizer_state_dict': optimizer.state_dict(),
                'opponent_pool': opponent_pool
            }, checkpoint_path)

    envs.close()
    pbar.close()

if __name__ == "__main__":
    train_ppo()