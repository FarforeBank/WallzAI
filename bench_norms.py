"""Изоляция причины медленного forward: GroupNorm vs BatchNorm vs без нормы."""
import time
import torch
import torch.nn as nn

from model import QuoridorActorCritic, ResBlock

device = "mps"


def bench(name, fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.mps.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.mps.synchronize()
    print(f"{name:44s} {(time.perf_counter() - t0) / n * 1000:8.2f} ms")


def strip_gn(m):
    for mod in m.modules():
        if isinstance(mod, ResBlock):
            mod.gn1 = nn.Identity()
            mod.gn2 = nn.Identity()
    m.initial[1] = nn.Identity()


def swap_to_bn(m):
    for mod in m.modules():
        if isinstance(mod, ResBlock):
            mod.gn1 = nn.BatchNorm2d(256).to(device)
            mod.gn2 = nn.BatchNorm2d(256).to(device)
    m.initial[1] = nn.BatchNorm2d(256).to(device)


for B in (64, 512):
    print(f"--- batch={B} ---")
    x = torch.zeros(B, 6, 9, 9, device=device)
    mk = torch.ones(B, 136, dtype=torch.bool, device=device)

    m = QuoridorActorCritic(num_actions=136).to(device).eval()
    with torch.no_grad():
        bench("forward fp32 (GroupNorm)", lambda: m(x, mk))
        with torch.autocast(device_type="mps", dtype=torch.float16):
            bench("forward fp16-autocast (GroupNorm)", lambda: m(x, mk))

    strip_gn(m)
    with torch.no_grad():
        bench("forward fp32 (NO norm)", lambda: m(x, mk))
        with torch.autocast(device_type="mps", dtype=torch.float16):
            bench("forward fp16-autocast (NO norm)", lambda: m(x, mk))

    m2 = QuoridorActorCritic(num_actions=136).to(device).eval()
    swap_to_bn(m2)
    m2.eval()
    with torch.no_grad():
        bench("forward fp32 (BatchNorm)", lambda: m2(x, mk))
    del m, m2

# Отдельно: стоимость одной нормализации
y = torch.randn(512, 256, 9, 9, device=device)
gn = nn.GroupNorm(32, 256).to(device)
bn = nn.BatchNorm2d(256).to(device).eval()
bench("GroupNorm (512,256,9,9) fp32", lambda: gn(y))
bench("BatchNorm2d (512,256,9,9) fp32", lambda: bn(y))
conv = nn.Conv2d(256, 256, 3, padding=1).to(device)
bench("Conv2d 3x3 (512,256,9,9) fp32", lambda: conv(y))
