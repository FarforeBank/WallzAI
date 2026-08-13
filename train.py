"""
PPO Self-Play для Quoridor — GPU-native версия (итерация 2 рефакторинга).

Ключевые изменения относительно multiprocessing-версии:
  [SPEED-1] Среда QuoridorBatchedTensorEnv живёт на MPS: ни IPC, ни копирований
            obs CPU<->GPU на каждом шаге. Rollout полностью на устройстве.
  [SPEED-2] num_envs=512 вместо 32 процессов — батч-инференс реально загружает GPU.
  [FIX-1]   lr 2.5e-4 (было 2.5e-5 — опечатка на порядок, обучение ползло).
  [FIX-2]   Пул оппонентов сохраняется отдельно (checkpoints/opponent_pool.pt).
            Раньше пул из 20 state-dict'ов (~44 МБ каждый) писался в best_model.pt
            каждые 25 апдейтов -> чекпоинт ~900 МБ и секунды на каждом save/load.
  [FIX-3]   Убраны дублирующиеся time-penalty (0.002 / 0.02) поверх шейпинга среды
            (среда уже даёт -0.01 за полуход) — награды теперь консистентны.
  [LOGIC]   Self-play: обучающийся агент всегда играет за Игрока 1, замороженный
            оппонент из исторического пула — за Игрока 2. Так как тензорная среда
            шагает все среды залочено, переходы агента собираются через per-env
            slot-бухгалтерию (см. комментарии в rollout). GAE считается по
            "шагам агента", а не по raw-шагам среды — это корректный MDP с точки
            зрения агента: награда перехода = r(ход агента) - r(ответ оппонента).
"""

import os
import time
import random
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from model import QuoridorActorCritic
from tensor_env import QuoridorBatchedTensorEnv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=512,
                   help="Параллельных сред на GPU")
    p.add_argument("--num-steps", type=int, default=64,
                   help="Переходов АГЕНТА на среду за rollout (T)")
    p.add_argument("--total-timesteps", type=int, default=50_000_000,
                   help="Лимит по raw-шагам среды (полуходы)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--num-minibatches", type=int, default=4)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--num-blocks", type=int, default=6)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.1)
    p.add_argument("--ent-coef-start", type=float, default=0.05, help="Начальная энтропия")
    p.add_argument("--ent-coef-end", type=float, default=0.005, help="Конечная энтропия")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--pool-size", type=int, default=10)
    p.add_argument("--pool-snapshot-every", type=int, default=20,
                   help="Снапшот агента в пул оппонентов каждые N rollout'ов")
    p.add_argument("--save-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    use_amp = device.type == "mps"
    print(f"🚀 Устройство: {device} | GPU-NATIVE TENSOR ENV | envs={args.num_envs}, T={args.num_steps}")

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = "checkpoints/best_model.pt"
    pool_path = "checkpoints/opponent_pool.pt"

    # ------------------------------------------------------------------
    # Среда и агенты
    # ------------------------------------------------------------------
    envs = QuoridorBatchedTensorEnv(num_envs=args.num_envs, device=device)
    B = args.num_envs
    T = args.num_steps

    agent = QuoridorActorCritic(num_actions=136, hidden_dim=args.hidden_dim,
                                num_blocks=args.num_blocks).to(device)
    optimizer = optim.AdamW(agent.parameters(), lr=args.lr, eps=1e-5)

    opponent = QuoridorActorCritic(num_actions=136, hidden_dim=args.hidden_dim,
                                   num_blocks=args.num_blocks).to(device)
    opponent.eval()

    # Подготовка переменной для старта
    start_update = 1

    # Загрузка агента и пула оппонентов
    opponent_pool = []
    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        try:
            agent.load_state_dict(ckpt["model_state_dict"])
            if "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            
            # ВАЖНО: Восстанавливаем шаг для расчета энтропии и LR
            if "update" in ckpt:
                start_update = ckpt["update"] + 1
                
            print(f"--> Загружен чекпоинт агента. Продолжаем с апдейта {start_update}.")
        except RuntimeError as e:
            print(f"[!] Архитектура чекпоинта не совпадает с текущей моделью —")
            print(f"    стартуем с нуля (пул оппонентов сохраним, если есть).")
        # Обратная совместимость: пул из чекпоинта старого формата
        if not os.path.exists(pool_path) and ckpt.get("opponent_pool"):
            opponent_pool = ckpt["opponent_pool"]
            print("--> Пул оппонентов перенесён из чекпоинта старого формата.")
    
    if os.path.exists(pool_path):
        opponent_pool = torch.load(pool_path, map_location="cpu")
        print(f"--> Загружен пул оппонентов ({len(opponent_pool)} снапшотов).")
    
    if not opponent_pool:
        opponent_pool = [{k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}]
    
    # Отбрасываем снапшоты пула, несовместимые с текущей архитектурой
    model_keys = set(agent.state_dict().keys())
    opponent_pool = [sd for sd in opponent_pool if set(sd.keys()) == model_keys]
    if not opponent_pool:
        opponent_pool = [{k: v.detach().cpu().clone() for k, v in agent.state_dict().items()}]
    current_opp_idx = -1

    # ------------------------------------------------------------------
    # Буферы rollout. Индекс слота = номер перехода АГЕНТА в данной среде.
    # CAP — верхняя граница числа ходов агента за rollout (raw-шагов всего
    # 2*(T+1), и каждый raw-шаг — максимум один ход агента в среде).
    # ------------------------------------------------------------------
    CAP = 2 * (T + 1)
    obs_b      = torch.zeros((CAP, B, 6, 9, 9), device=device)
    masks_b    = torch.zeros((CAP, B, 136), dtype=torch.bool, device=device)
    actions_b  = torch.zeros((CAP, B), dtype=torch.long, device=device)
    logprobs_b = torch.zeros((CAP, B), device=device)
    values_b   = torch.zeros((CAP, B), device=device)
    rewards_b  = torch.zeros((CAP, B), device=device)
    dones_b    = torch.zeros((CAP, B), device=device)

    slot_ctr       = torch.zeros(B, dtype=torch.long, device=device)   # сколько переходов записано
    pending_slot   = torch.zeros(B, dtype=torch.long, device=device)   # слот незакрытого перехода
    pending_rew    = torch.zeros(B, device=device)                     # накопленная награда перехода
    pending_active = torch.zeros(B, dtype=torch.bool, device=device)   # ждёт ответа оппонента

    raw_steps = 2 * (T + 1)
    batch_size = B * T
    minibatch_size = batch_size // args.num_minibatches
    steps_per_rollout = B * raw_steps
    num_updates = max(1, args.total_timesteps // steps_per_rollout)

    obs, info = envs.reset()

    pbar = tqdm(total=num_updates, initial=start_update-1, desc="PPO GPU-native", unit="upd", dynamic_ncols=True)

    for update in range(start_update, num_updates + 1):
        t0 = time.time()

        # Линейный отжиг lr и коэффициента энтропии
        frac = 1.0 - (update - 1.0) / num_updates
        current_ent_coef = args.ent_coef_end + (args.ent_coef_start - args.ent_coef_end) * frac
        
        for g in optimizer.param_groups:
            g["lr"] = args.lr * frac

        # Выбираем оппонента на текущий rollout
        opp_idx = random.randrange(len(opponent_pool))
        if opp_idx != current_opp_idx:
            opponent.load_state_dict({k: v.to(device) for k, v in opponent_pool[opp_idx].items()})
            current_opp_idx = opp_idx

        # Сброс бухгалтерии rollout'а
        slot_ctr.zero_()
        pending_active.zero_()
        pending_rew.zero_()
        dones_b.zero_()

        wins = torch.zeros(B, device=device)
        losses = torch.zeros(B, device=device)
        finishes = torch.zeros(B, device=device)

        agent.eval()
        for _raw in range(raw_steps):
            obs_dec = obs
            mask_dec = info["valid_actions_mask"]
            is_p1 = envs.turn == 1
            agent_idx = envs.b_idx[is_p1]
            opp_env_idx = envs.b_idx[~is_p1]

            actions = torch.empty(B, dtype=torch.long, device=device)
            a_act = a_logp = a_val = None

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                if agent_idx.numel() > 0:
                    a_act, a_logp, _, a_val = agent.get_action_and_value(obs_dec[agent_idx], mask_dec[agent_idx])
                    actions[agent_idx] = a_act
                if opp_env_idx.numel() > 0:
                    o_act, _, _, _ = opponent.get_action_and_value(obs_dec[opp_env_idx], mask_dec[opp_env_idx])
                    actions[opp_env_idx] = o_act

            obs, r, dones, _trunc, info = envs.step(actions)

            # ---- Бухгалтерия переходов агента (Игрок 1) ----
            if agent_idx.numel() > 0:
                has_p = pending_active[agent_idx]
                cids = agent_idx[has_p]
                if cids.numel() > 0:
                    rewards_b[pending_slot[cids], cids] = pending_rew[cids]
                    pending_active[cids] = False

                s = slot_ctr[agent_idx]
                obs_b[s, agent_idx] = obs_dec[agent_idx]
                masks_b[s, agent_idx] = mask_dec[agent_idx]
                actions_b[s, agent_idx] = a_act
                logprobs_b[s, agent_idx] = a_logp.float()
                values_b[s, agent_idx] = a_val.float().reshape(-1)
                pending_slot[agent_idx] = s
                pending_rew[agent_idx] = r[agent_idx]

                d_ag = dones[agent_idx]
                dids = agent_idx[d_ag]
                if dids.numel() > 0:
                    rewards_b[slot_ctr[dids], dids] = r[dids]
                    dones_b[slot_ctr[dids], dids] = True
                    wins[dids] += (r[dids] > 0.5).float()
                    losses[dids] += (r[dids] < -0.5).float()
                    finishes[dids] += 1.0
                pending_active[agent_idx] = ~d_ag
                slot_ctr[agent_idx] += 1

            # ---- Ответ оппонента (Игрок 2): минусуем его награду ----
            if opp_env_idx.numel() > 0:
                has_p = pending_active[opp_env_idx]
                pids = opp_env_idx[has_p]
                if pids.numel() > 0:
                    # ВАЖНО: Фикс аннигиляции time penalty! (компенсируем -0.01 за ожидание)
                    pending_rew[pids] -= (r[pids] + 0.02)
                    d_op = dones[pids]
                    dids = pids[d_op]
                    if dids.numel() > 0:
                        rewards_b[pending_slot[dids], dids] = pending_rew[dids]
                        dones_b[pending_slot[dids], dids] = True
                        pending_active[dids] = False
                        wins[dids] += (r[dids] < -0.5).float()   # оппонент получил < -0.5 (поражение) => мы выиграли
                        losses[dids] += (r[dids] > 0.5).float()  # оппонент получил > 0.5 (победа) => мы проиграли
                        finishes[dids] += 1.0

        min_slots = int(slot_ctr.min().item())
        assert min_slots >= T + 1, f"Нарушена инварианта буфера: {min_slots} < {T + 1}"

        # ------------------------------------------------------------------
        # GAE по шагам агента. dones_b[t]=1 => переход t завершил партию
        # ------------------------------------------------------------------
        with torch.no_grad():
            advantages = torch.zeros((T, B), device=device)
            lastgaelam = torch.zeros(B, device=device)
            for t in reversed(range(T)):
                nextnonterminal = 1.0 - dones_b[t]
                nextval = values_b[t + 1]
                delta = rewards_b[t] + args.gamma * nextval * nextnonterminal - values_b[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_b[:T]

        b_obs = obs_b[:T].reshape(-1, 6, 9, 9)
        b_masks = masks_b[:T].reshape(-1, 136)
        b_actions = actions_b[:T].reshape(-1)
        b_logprobs = logprobs_b[:T].reshape(-1)
        b_returns = returns.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        # ------------------------------------------------------------------
        # PPO update
        # ------------------------------------------------------------------
        agent.train()
        pg_loss_acc = v_loss_acc = kl_acc = 0.0
        n_mb = 0

        for _epoch in range(args.update_epochs):
            perm = torch.randperm(batch_size, device=device)
            for start in range(0, batch_size, minibatch_size):
                mb = perm[start:start + minibatch_size]

                _, newlogprob, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_masks[mb], b_actions[mb])
                logratio = newlogprob - b_logprobs[mb]
                ratio = logratio.exp()

                pg_loss = torch.max(
                    -b_advantages[mb] * ratio,
                    -b_advantages[mb] * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef),
                ).mean()
                v_loss = nn.functional.smooth_l1_loss(newval.reshape(-1), b_returns[mb])
                
                loss = pg_loss - current_ent_coef * entropy.mean() + args.vf_coef * v_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                pg_loss_acc += pg_loss.item()
                v_loss_acc += v_loss.item()
                kl_acc += approx_kl.item()
                n_mb += 1

        # ------------------------------------------------------------------
        # Логирование и Curriculum
        # ------------------------------------------------------------------
        dt = time.time() - t0
        fps = int(steps_per_rollout / dt)
        n_fin = finishes.sum().item()
        winrate = wins.sum().item() / n_fin if n_fin > 0 else 0.0
        lossrate = losses.sum().item() / n_fin if n_fin > 0 else 0.0
        drawrate = (n_fin - wins.sum().item() - losses.sum().item()) / n_fin if n_fin > 0 else 0.0

        pbar.set_postfix({
            "FPS": fps,
            "Win%": f"{100.0 * winrate:.1f}",
            "Draw%": f"{100.0 * drawrate:.1f}",
            "Loss%": f"{100.0 * lossrate:.1f}",
            "Ent": f"{current_ent_coef:.4f}",
            "VLoss": f"{v_loss_acc / n_mb:.4f}",
            "KL": f"{kl_acc / n_mb:.4f}",
        })
        pbar.update(1)

        # Smart Curriculum: Сохраняем в пул только если агент уверенно обыгрывает старые версии
        if update % args.pool_snapshot_every == 0:
            if winrate > 0.55:
                opponent_pool.append({k: v.detach().cpu().clone() for k, v in agent.state_dict().items()})
                if len(opponent_pool) > args.pool_size:
                    opponent_pool.pop(0)
            else:
                pass # Агент еще не готов стать новым бейзлайном

        if update % args.save_every == 0 or update == num_updates:
            torch.save({
                "model_state_dict": agent.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "update": update,
            }, checkpoint_path)
            torch.save(opponent_pool, pool_path)

    pbar.close()
    print("Обучение завершено. Финальный чекпоинт:", checkpoint_path)


if __name__ == "__main__":
    main()