import torch
import torch.nn as nn
from torch.distributions import Categorical

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # [FIX-SPEED] GroupNorm -> BatchNorm2d. GroupNorm на Apple MPS выполняется
        # в разы медленнее (профилировка: forward batch=64 занимал ~129 мс, это
        # главный bottleneck обучения). В eval-режиме BN — это fused pointwise
        # scale/shift, почти бесплатно и на MPS, и на ANE (Core ML).
        # ВАЖНО: rollout-инференс обязан быть в eval()-режиме (в train.py и
        # evaluate.py уже так), иначе BN использовал бы batch-статистики.
        self.gn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.gn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = torch.relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return torch.relu(x + residual)

class QuoridorActorCritic(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128, num_actions=136, num_blocks=6):
        super().__init__()
        # [FIX-SPEED] Башня уменьшена: 256ch x 8 блоков -> 128ch x 6 блоков.
        # Профилировка MPS показала, что forward compute-bound (~1.5 GFLOP/состояние
        # для поля 9x9 — избыточно для Quoridor). Новая конфигурация ~в 4.5 раза
        # дешевле при достаточной ёмкости для этой игры. Старые значения можно
        # вернуть явно: hidden_dim=256, num_blocks=8.
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )
        self.res_blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(num_blocks)])

        self.policy_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1)
        self.policy_fc = nn.Linear(32 * 9 * 9, num_actions)

        self.value_conv = nn.Conv2d(hidden_dim, 32, kernel_size=1)
        self.value_fc1 = nn.Linear(32 * 9 * 9, 512)
        self.value_fc2 = nn.Linear(512, 1)

        self.apply(self._init_weights)
        nn.init.orthogonal_(self.policy_fc.weight, gain=0.01)
        nn.init.orthogonal_(self.value_fc2.weight, gain=1.0)

    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x, action_mask=None):
        features = self.initial(x)
        for block in self.res_blocks:
            features = block(features)

        p = torch.relu(self.policy_conv(features))
        logits = self.policy_fc(p.reshape(p.size(0), -1))

        if action_mask is not None:
            # -10000.0 исключает переполнение FP16 на Apple Neural Engine и MPS
            mask_val = -10000.0
            logits = logits + ((1.0 - action_mask.float()) * mask_val)

        v = torch.relu(self.value_conv(features))
        v = torch.relu(self.value_fc1(v.reshape(v.size(0), -1)))
        # [FIX] Без tanh: награды за победу/поражение — ±10, поэтому tanh([-1, 1])
        # делал целевые returns физически недостижимыми для критика и взрывал value-loss.
        value = self.value_fc2(v)
        return logits, value

    def get_action_and_value(self, x, action_mask=None, action=None):
        logits, value = self.forward(x, action_mask)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value