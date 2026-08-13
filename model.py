import torch
import torch.nn as nn
from torch.distributions import Categorical

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        # [FIX] Используем InstanceNorm2d(affine=True) вместо BatchNorm2d/GroupNorm.
        # В RL данные нестационарны, поэтому BN ломает обучение из-за running_stats.
        # InstanceNorm нормализует каждый пример независимо, отлично оптимизирован
        # на Apple Silicon (MPS/CoreML) и не зависит от размера батча.
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        x = torch.relu(self.in1(self.conv1(x)))
        x = self.in2(self.conv2(x))
        return torch.relu(x + residual)

class QuoridorActorCritic(nn.Module):
    def __init__(self, in_channels=6, hidden_dim=128, num_actions=136, num_blocks=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.InstanceNorm2d(hidden_dim, affine=True),
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
        # Без tanh, чтобы Value Function могла спокойно сходиться к ±10
        value = self.value_fc2(v)
        return logits, value

    def get_action_and_value(self, x, action_mask=None, action=None):
        logits, value = self.forward(x, action_mask)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), value