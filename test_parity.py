"""
Тест паритета: QuoridorBatchedTensorEnv (GPU) vs QuoridorEnv (CPU/numba).

Прогоняет N идентичных партий со случайными ЛЕГАЛЬНЫМИ действиями и сверяет
после каждого шага: obs, маски валидных действий, награды, флаги завершения.
Любое расхождение — AssertionError с описанием.
"""
import random

import numpy as np
import torch

from game_env import QuoridorEnv
from tensor_env import QuoridorBatchedTensorEnv

N_ENVS = 8
STEPS = 400
SEED = 123


def main():
    random.seed(SEED)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tenv = QuoridorBatchedTensorEnv(num_envs=N_ENVS, device=device)
    cenvs = [QuoridorEnv() for _ in range(N_ENVS)]

    c_obs, c_info = [], []
    for e in cenvs:
        o, i = e.reset()
        c_obs.append(o)
        c_info.append(i)

    t_obs, t_info = tenv.reset()
    t_mask = t_info["valid_actions_mask"]

    checked = 0
    for step in range(STEPS):
        actions = []
        for i in range(N_ENVS):
            m = c_info[i]["valid_actions_mask"]
            valid = np.flatnonzero(m)
            assert len(valid) > 0, f"env {i}: пустая маска в CPU-среде на шаге {step}"
            actions.append(int(random.choice(valid)))

        # Сверка obs/масок ДО шага
        for i in range(N_ENVS):
            to = t_obs[i].cpu().numpy()
            tm = t_mask[i].cpu().numpy()
            if not np.allclose(to, c_obs[i], atol=1e-6):
                bad = np.argwhere(~np.isclose(to, c_obs[i], atol=1e-6))
                raise AssertionError(
                    f"[step {step} env {i}] OBS mismatch в {len(bad)} точках, "
                    f"напр. {bad[0]}: tensor={to[tuple(bad[0])]}, cpu={c_obs[i][tuple(bad[0])]}")
            if not np.array_equal(tm, c_info[i]["valid_actions_mask"]):
                diff = np.flatnonzero(tm != c_info[i]["valid_actions_mask"])
                raise AssertionError(
                    f"[step {step} env {i}] MASK mismatch по действиям {diff}: "
                    f"tensor={tm[diff]}, cpu={c_info[i]['valid_actions_mask'][diff]}")
            checked += 1

        a_t = torch.tensor(actions, dtype=torch.long, device=device)
        t_obs, t_rew, t_done, t_trunc, t_info = tenv.step(a_t)
        t_mask = t_info["valid_actions_mask"]

        for i in range(N_ENVS):
            c_obs[i], c_rew, c_term, c_trunc, c_info[i] = cenvs[i].step(actions[i])
            c_done = c_term or c_trunc

            td = bool(t_done[i].item())
            tr = float(t_rew[i].item())
            if td != c_done:
                raise AssertionError(
                    f"[step {step} env {i}] DONE mismatch: tensor={td}, cpu={c_done} (action={actions[i]})")
            if abs(tr - c_rew) > 1e-5:
                raise AssertionError(
                    f"[step {step} env {i}] REWARD mismatch: tensor={tr}, cpu={c_rew} (action={actions[i]})")

            if c_done:
                c_obs[i], c_info[i] = cenvs[i].reset()

    print(f"✅ ПАРИТЕТ ПОДТВЕРЖДЁН: {N_ENVS} сред x {STEPS} шагов, "
          f"{checked} сверок obs/mask, {N_ENVS * STEPS} сверок reward/done — расхождений нет.")


if __name__ == "__main__":
    main()
