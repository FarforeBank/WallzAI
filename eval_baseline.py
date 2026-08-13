"""Диагностика силы текущего чекпоинта: матч против скриптованного «вокера».

Вокер — наивный бейзлайн: всегда идёт пешкой к цели (action 0 — «вперёд» в
канонических координатах), никогда не ставит стены. Любой минимально обучившийся
агент обязан выигрывать у него ~100%. Матч со сменой сторон. Также считаем долю
ходов-стен у модели (симптом «wall-spam» фазы).
"""
import sys
import numpy as np
import torch

from model import QuoridorActorCritic
from game_env import QuoridorEnv

CKPT = "checkpoints/best_model.pt"
MLPKG = "checkpoints/QuoridorNPU.mlpackage"
GAMES = 40


def walker_action(mask):
    for a in (0, 2, 3, 4, 5, 6, 7, 1):  # вперёд, вбок, диагонали, назад
        if mask[a]:
            return a
    return int(np.flatnonzero(mask)[0])


def model_action(agent, obs, mask, device):
    obs_t = torch.from_numpy(obs).unsqueeze(0).to(device)
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(device)
    with torch.no_grad():
        logits, _ = agent(obs_t, mask_t)
    return int(torch.argmax(logits, dim=-1).item())


def main():
    use_npu = "--npu" in sys.argv
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    if use_npu:
        from npu_inference import QuoridorNPUInference
        npu = QuoridorNPUInference(MLPKG)

        def pick(obs, mask):
            return npu.predict_action(obs, mask)
        print(f"Агент: Core ML / ANE ({MLPKG})")
    else:
        agent = QuoridorActorCritic(num_actions=136).to(device)
        ckpt = torch.load(CKPT, map_location=device)
        agent.load_state_dict(ckpt["model_state_dict"])
        agent.eval()

        def pick(obs, mask):
            return model_action(agent, obs, mask, device)
        print(f"Агент: PyTorch ({CKPT}, update={ckpt.get('update', '?')})")

    env = QuoridorEnv()
    wins = losses = draws = 0
    wall_actions = model_actions = 0
    total_steps = 0

    for g in range(GAMES):
        obs, info = env.reset()
        model_is_p1 = g < GAMES // 2
        done = False
        steps = 0
        while not done:
            mask = info["valid_actions_mask"]
            model_turn = (env.turn == 1) == model_is_p1
            if model_turn:
                action = pick(obs, mask)
                model_actions += 1
                if action >= 8:
                    wall_actions += 1
            else:
                action = walker_action(mask)
            obs, r, term, trunc, info = env.step(action)
            done = term or trunc
            steps += 1
        total_steps += steps

        if trunc:
            draws += 1
        else:
            # turn уже переключён: победил предыдущий игрок
            winner = 2 if env.turn == 1 else 1
            model_won = (winner == 1) == model_is_p1
            if model_won:
                wins += 1
            else:
                losses += 1

    print(f"\n=== Модель vs Вокер ({GAMES} партий, со сменой сторон) ===")
    print(f"Победы модели : {wins} ({100.0 * wins / GAMES:.1f}%)")
    print(f"Поражения     : {losses}")
    print(f"Ничьи (400)   : {draws}")
    print(f"Ср. длина     : {total_steps / GAMES:.1f} полуходов")
    print(f"Доля ходов-стен у модели: {100.0 * wall_actions / max(model_actions, 1):.1f}% "
          f"({wall_actions}/{model_actions})")


if __name__ == "__main__":
    main()
