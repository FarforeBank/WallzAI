import os
import torch
import numpy as np
from model import QuoridorActorCritic
from game_env import QuoridorEnv
from npu_inference import QuoridorNPUInference

def load_agent(checkpoint_path, device):
    agent = QuoridorActorCritic(num_actions=136).to(device)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        agent.load_state_dict(checkpoint['model_state_dict'])
        print(f"[+] Успешно загружена модель: {checkpoint_path}")
    else:
        print(f"[!] Чекпоинт {checkpoint_path} не найден! Используются случайные веса.")
    agent.eval()
    return agent

def get_greedy_action(agent, obs_np, mask_np, device):
    # Если это Core ML / NPU модель (имеет метод predict_action)
    if hasattr(agent, 'predict_action'):
        return agent.predict_action(obs_np, mask_np)
    
    # Иначе это PyTorch модель
    obs_t = torch.from_numpy(obs_np).unsqueeze(0).float().to(device)
    mask_t = torch.from_numpy(mask_np).unsqueeze(0).bool().to(device)

    with torch.no_grad():
        with torch.autocast(device_type='mps' if device.type == 'mps' else 'cpu', dtype=torch.float16):
            logits, _ = agent(obs_t, mask_t)
            action = torch.argmax(logits, dim=-1).item()
    return action

def run_tournament(agent_a, agent_b, device, num_games=100):
    env = QuoridorEnv()

    wins_a = 0
    wins_b = 0
    draws = 0
    total_steps = 0

    print(f"\nНачинаем турнир из {num_games} партий...")
    print("Первые 50% игр Агент A играет за Игрока 1 (Снизу).")

    for game in range(num_games):
        obs, info = env.reset()
        done = False

        if game < num_games // 2:
            p1_agent, p2_agent = agent_a, agent_b
        else:
            p1_agent, p2_agent = agent_b, agent_a

        if game == num_games // 2:
            print("\nСмена сторон! Теперь Агент A играет за Игрока 2 (Сверху).")

        steps = 0
        while not done:
            mask = info['valid_actions_mask']
            active_agent = p1_agent if env.turn == 1 else p2_agent
            action = get_greedy_action(active_agent, obs, mask, device)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        total_steps += steps

        if truncated:
            draws += 1
            winner_str = "Ничья (лимит шагов)"
        else:
            winner_turn = 2 if env.turn == 1 else 1

            if game < num_games // 2:
                if winner_turn == 1:
                    wins_a += 1
                    winner_str = "Агент A"
                else:
                    wins_b += 1
                    winner_str = "Агент B"
            else:
                if winner_turn == 1:
                    wins_b += 1
                    winner_str = "Агент B"
                else:
                    wins_a += 1
                    winner_str = "Агент A"

        if (game + 1) % 10 == 0:
            print(f"Партия {game + 1}/{num_games} завершена. Победитель: {winner_str} ({steps} шагов)")

    print("\n" + "="*30)
    print("ИТОГИ ТУРНИРА")
    print("="*30)
    print(f"Победы Агента A : {wins_a} ({(wins_a/num_games)*100:.1f}%)")
    print(f"Победы Агента B : {wins_b} ({(wins_b/num_games)*100:.1f}%)")
    print(f"Ничьи (Таймаут) : {draws} ({(draws/num_games)*100:.1f}%)")
    print(f"Ср. длина игры  : {total_steps / num_games:.1f} шагов")
    print("="*30)

# Обнови инициализацию для турнира:
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    print("=== Инициализация Агента A (PyTorch MPS) ===")
    agent_a = load_agent("checkpoints/best_model.pt", device)

    print("\n=== Инициализация Агента B (Apple Neural Engine) ===")
    # Заменяем PyTorch-инференс бота на аппаратно-ускоренный Core ML
    try:
        agent_b = QuoridorNPUInference("checkpoints/QuoridorNPU.mlpackage")
    except Exception as e:
        print(f"Ошибка загрузки NPU модели: {e}. Используем fallback PyTorch.")
        agent_b = load_agent("quoridor_ppo_checkpoint.pt", device)
        
    # В функции get_greedy_action нужно добавить ветвление:
    # если агент имеет тип QuoridorNPUInference, вызываем agent.predict_action(obs_np, mask_np)