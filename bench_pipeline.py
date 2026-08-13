"""Профилировка узких мест GPU-пайплайна: env.step, flood fill, инференс, апдейт."""
import time
import torch

from tensor_env import QuoridorBatchedTensorEnv
from model import QuoridorActorCritic

device = "mps"
B = 64

env = QuoridorBatchedTensorEnv(num_envs=B, device=device)
agent = QuoridorActorCritic(num_actions=136).to(device)
opt = torch.optim.AdamW(agent.parameters(), lr=2.5e-4)
obs, info = env.reset()
mask = info["valid_actions_mask"]


def bench(name, fn, n=30, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    dt = (time.perf_counter() - t0) / n * 1000
    print(f"{name:38s} {dt:8.2f} ms")
    return dt


a = torch.zeros(B, dtype=torch.long, device=device)  # action 0 всегда легален в начале

bench("env.step (action=0)", lambda: env.step(a))
bench("batched_flood_fill (all)", lambda: env.batched_flood_fill())
bench("_get_obs", lambda: env._get_obs())
bench("_get_info (mask)", lambda: env._get_info())

with torch.no_grad(), torch.autocast(device_type="mps", dtype=torch.float16):
    bench("agent.forward batch=64 fp16", lambda: agent.get_action_and_value(obs, mask))

mb_obs = obs[:B]
mb_mask = mask[:B]


def upd():
    _, lp, ent, v = agent.get_action_and_value(mb_obs, mb_mask, torch.zeros(B, dtype=torch.long, device=device))
    loss = lp.mean() - 0.01 * ent.mean() + v.mean()
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


agent.train()
bench("train fwd+bwd batch=64 fp32", upd, n=10, warmup=2)

# boolean indexing / scatter как в train.py
is_p1 = env.turn == 1
bench("b_idx[is_p1] (bool index)", lambda: env.b_idx[is_p1])
s = torch.zeros(B, dtype=torch.long, device=device)
buf = torch.zeros((40, B, 6, 9, 9), device=device)
bench("obs_b[s, idx] = obs[idx] (scatter)", lambda: buf.__setitem__((s, env.b_idx), obs))
