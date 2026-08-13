import torch

class QuoridorBatchedTensorEnv:
    def __init__(self, num_envs=1024, device="mps"):
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.num_actions = 136
        
        self.b_idx = torch.arange(num_envs, device=self.device)
        
        # Тензоры состояния
        self.p1_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.p2_pos = torch.zeros((num_envs, 2), dtype=torch.long, device=self.device)
        self.p1_walls = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self.p2_walls = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.h_walls = torch.zeros((num_envs, 9, 9), dtype=torch.bool, device=self.device)
        self.v_walls = torch.zeros((num_envs, 9, 9), dtype=torch.bool, device=self.device)
        self.centers = torch.zeros((num_envs, 8, 8), dtype=torch.bool, device=self.device)
        
        self.turn = torch.ones(num_envs, dtype=torch.long, device=self.device)
        self.step_count = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        
        self.p1_last_dist = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.p2_last_dist = torch.zeros(num_envs, dtype=torch.float32, device=self.device)

        # [ПАТЧ 4] Перестановка каноническое <-> абсолютное действие (инволюция).
        # Совпадает с QuoridorEnv._map_action_to_absolute: ходы 0<->1, 2<->3, 4<->7, 5<->6,
        # стены (r, c) -> (7-r, 7-c). Для хода игрока 2 obs канонический,
        # поэтому действия и маски тоже должны быть каноническими.
        perm = list(range(self.num_actions))
        perm[0], perm[1] = 1, 0
        perm[2], perm[3] = 3, 2
        perm[4], perm[5], perm[6], perm[7] = 7, 6, 5, 4
        for base in (8, 72):
            for i in range(64):
                wr, wc = i // 8, i % 8
                perm[base + i] = base + (7 - wr) * 8 + (7 - wc)
        self.action_perm = torch.tensor(perm, dtype=torch.long, device=self.device)

        self._reset_state(self.b_idx)

    def _reset_state(self, indices):
        """Сброс состояния сред БЕЗ пересчёта obs/mask.

        [FIX-SPEED] Раньше step() вызывал reset(), который считал _get_obs() и
        _get_info() (дорогая маска!) для ВСЕХ сред — и выбрасывал результат.
        """
        if indices.numel() == 0:
            return

        self.p1_pos[indices, 0] = 8
        self.p1_pos[indices, 1] = 4
        self.p2_pos[indices, 0] = 0
        self.p2_pos[indices, 1] = 4

        self.p1_walls[indices] = 10
        self.p2_walls[indices] = 10
        self.h_walls[indices] = False
        self.v_walls[indices] = False
        self.centers[indices] = False
        self.turn[indices] = 1
        self.step_count[indices] = 0

        # Flood fill только для сбрасываемых сред, а не для всех
        d1, d2 = self.batched_flood_fill(indices)
        self.p1_last_dist[indices] = d1
        self.p2_last_dist[indices] = d2

    def reset(self, indices=None):
        if indices is None:
            indices = self.b_idx
        self._reset_state(indices)
        return self._get_obs(), self._get_info()

    def step(self, actions):
        self.step_count += 1
        is_p1 = (self.turn == 1)

        # [ПАТЧ 4] Агент играет в канонических координатах (как в game_env):
        # для хода игрока 2 переводим действие в абсолютное.
        actions = torch.where(is_p1, actions, self.action_perm[actions])
        
        active_pos = torch.where(is_p1.unsqueeze(1), self.p1_pos, self.p2_pos)
        op_pos = torch.where(is_p1.unsqueeze(1), self.p2_pos, self.p1_pos)
        
        r, c = active_pos[:, 0], active_pos[:, 1]
        op_r, op_c = op_pos[:, 0], op_pos[:, 1]
        
        # 1. МАСКИ ДЕЙСТВИЙ
        act_up    = (actions == 0)
        act_down  = (actions == 1)
        act_left  = (actions == 2)
        act_right = (actions == 3)
        act_ul    = (actions == 4)
        act_ur    = (actions == 5)
        act_dl    = (actions == 6)
        act_dr    = (actions == 7)
        
        is_h_wall = (actions >= 8) & (actions < 72)
        is_v_wall = (actions >= 72)
        
        new_r = r.clone()
        new_c = c.clone()
        
        # 2. ОБРАБОТКА ХОДОВ И ПРЫЖКОВ
        new_r[act_up] -= 1
        new_r[act_down] += 1
        new_c[act_left] -= 1
        new_c[act_right] += 1
        
        op_match = (new_r == op_r) & (new_c == op_c)
        new_r[act_up & op_match] -= 1
        new_r[act_down & op_match] += 1
        new_c[act_left & op_match] -= 1
        new_c[act_right & op_match] += 1
        
        new_r[act_ul | act_ur] -= 1
        new_r[act_dl | act_dr] += 1
        new_c[act_ul | act_dl] -= 1
        new_c[act_ur | act_dr] += 1
        
        # [ПАТЧ 1] Предохранитель OOB
        new_r = torch.clamp(new_r, min=0, max=8)
        new_c = torch.clamp(new_c, min=0, max=8)
        
        is_move = act_up | act_down | act_left | act_right | act_ul | act_ur | act_dl | act_dr
        active_pos[:, 0] = torch.where(is_move, new_r, active_pos[:, 0])
        active_pos[:, 1] = torch.where(is_move, new_c, active_pos[:, 1])
        
        self.p1_pos = torch.where(is_p1.unsqueeze(1), active_pos, self.p1_pos)
        self.p2_pos = torch.where(~is_p1.unsqueeze(1), active_pos, self.p2_pos)
        
        # 3. ОБРАБОТКА УСТАНОВКИ СТЕН
        # [FIX-SPEED] Убраны ветки `if is_h_wall.any()` / `if is_v_wall.any()` —
        # каждая из них это синхронизация GPU->CPU на КАЖДОМ шаге. Теперь
        # векторный masked-OR без булева индексирования (индексы b_idx уникальны).
        h_idx = torch.clamp(actions - 8, min=0, max=63)
        hr, hc = h_idx // 8, h_idx % 8
        self.h_walls[self.b_idx, hr, hc] |= is_h_wall
        self.h_walls[self.b_idx, hr, hc + 1] |= is_h_wall

        v_idx = torch.clamp(actions - 72, min=0, max=63)
        vr, vc = v_idx // 8, v_idx % 8
        self.v_walls[self.b_idx, vr, vc] |= is_v_wall
        self.v_walls[self.b_idx, vr + 1, vc] |= is_v_wall

        is_wall_action = is_h_wall | is_v_wall
        center_r = torch.where(is_h_wall, hr, vr)
        center_c = torch.where(is_h_wall, hc, vc)
        self.centers[self.b_idx, center_r, center_c] |= is_wall_action

        # Списание стен (clamp от ухода в минус)
        self.p1_walls = torch.where(is_p1 & is_wall_action, torch.clamp(self.p1_walls - 1, min=0), self.p1_walls)
        self.p2_walls = torch.where(~is_p1 & is_wall_action, torch.clamp(self.p2_walls - 1, min=0), self.p2_walls)

        # 4. СМЕНА ХОДА И РАСЧЕТ НАГРАДЫ
        self.turn = 3 - self.turn

        # Один батч на обоих игроков вместо двух вызовов
        p1_dist, p2_dist = self.batched_flood_fill()

        delta_p1 = self.p1_last_dist - p1_dist
        delta_p2 = self.p2_last_dist - p2_dist

        rewards = torch.where(is_p1, (delta_p1 - delta_p2) * 0.01 - 0.01, (delta_p2 - delta_p1) * 0.01 - 0.01)

        self.p1_last_dist = p1_dist
        self.p2_last_dist = p2_dist

        p1_wins = (self.p1_pos[:, 0] == 0)
        p2_wins = (self.p2_pos[:, 0] == 8)

        # [ПАТЧ 2] Приоритет победы над тупиками
        invalid_path = ((p1_dist == 81) | (p2_dist == 81)) & ~(p1_wins | p2_wins)

        rewards = torch.where(invalid_path, torch.tensor(-1.0, device=self.device), rewards)
        rewards = torch.where(p1_wins & is_p1, torch.tensor(1.0, device=self.device), rewards)
        rewards = torch.where(p2_wins & ~is_p1, torch.tensor(1.0, device=self.device), rewards)

        terminated = p1_wins | p2_wins | invalid_path
        truncated = self.step_count > 400
        dones = terminated | truncated

        if bool(dones.any()):
            self._reset_state(self.b_idx[dones])

        return self._get_obs(), rewards, dones, truncated, self._get_info()

    def _get_obs(self):
        obs = torch.zeros((self.num_envs, 6, 9, 9), dtype=torch.float32, device=self.device)
        
        # Заполняем как Игрок 1 (Абсолютное представление)
        obs[self.b_idx, 0, self.p1_pos[:, 0], self.p1_pos[:, 1]] = 1.0
        obs[self.b_idx, 1, self.p2_pos[:, 0], self.p2_pos[:, 1]] = 1.0
        obs[:, 2] = self.h_walls.float()
        obs[:, 3] = self.v_walls.float()
        obs[:, 4] = (self.p1_walls.float() / 10.0).view(-1, 1, 1).expand(-1, 9, 9)
        obs[:, 5] = (self.p2_walls.float() / 10.0).view(-1, 1, 1).expand(-1, 9, 9)
        
        is_p2 = (self.turn == 2)
        if is_p2.any():
            flipped_obs = obs[is_p2].clone()

            # Каналы 0-1 (пешки): точка (r, c) -> (8-r, 8-c), обычный поворот на 180°
            flipped_obs[:, 0:2] = torch.flip(flipped_obs[:, 0:2], dims=[2, 3])

            # [ПАТЧ 5] Каналы стен: у сегментов маркировка НЕ центрально-симметрична.
            # h_walls (r, c) -> (7-r, 8-c), v_walls (r, c) -> (8-r, 7-c),
            # как в QuoridorEnv._get_obs. Голый torch.flip давал (8-r, 8-c) —
            # смещение на клетку и рассинхрон с CPU-средой/инференсом.
            h_flip = torch.flip(self.h_walls[is_p2].float(), dims=[1, 2])
            flipped_obs[:, 2] = torch.roll(h_flip, shifts=-1, dims=1)
            v_flip = torch.flip(self.v_walls[is_p2].float(), dims=[1, 2])
            flipped_obs[:, 3] = torch.roll(v_flip, shifts=-1, dims=2)

            # Меняем местами каналы:
            # 0 (Своя позиция) <-> 1 (Позиция оппонента)
            # 4 (Свои стены) <-> 5 (Стены оппонента)
            flipped_obs = flipped_obs[:, [1, 0, 2, 3, 5, 4], :, :]

            obs[is_p2] = flipped_obs
            
        return obs

    def _get_info(self):
        mask = torch.zeros((self.num_envs, self.num_actions), dtype=torch.bool, device=self.device)
        is_p1 = (self.turn == 1)
        
        active_pos = torch.where(is_p1.unsqueeze(1), self.p1_pos, self.p2_pos)
        op_pos = torch.where(is_p1.unsqueeze(1), self.p2_pos, self.p1_pos)
        walls_left = torch.where(is_p1, self.p1_walls, self.p2_walls)
        
        r, c = active_pos[:, 0], active_pos[:, 1]
        op_r, op_c = op_pos[:, 0], op_pos[:, 1]
        
        base_up = (r > 0) & ~self.h_walls[self.b_idx, torch.clamp(r-1, min=0), c]
        base_down = (r < 8) & ~self.h_walls[self.b_idx, r, c]
        base_left = (c > 0) & ~self.v_walls[self.b_idx, r, torch.clamp(c-1, min=0)]
        base_right = (c < 8) & ~self.v_walls[self.b_idx, r, c]
        
        op_up = (op_r == r-1) & (op_c == c)
        op_down = (op_r == r+1) & (op_c == c)
        op_left = (op_r == r) & (op_c == c-1)
        op_right = (op_r == r) & (op_c == c+1)
        
        jump_up_ok = (r > 1) & ~self.h_walls[self.b_idx, torch.clamp(r-2, min=0), c]
        jump_down_ok = (r < 7) & ~self.h_walls[self.b_idx, torch.clamp(r+1, max=7), c]
        jump_left_ok = (c > 1) & ~self.v_walls[self.b_idx, r, torch.clamp(c-2, min=0)]
        jump_right_ok = (c < 7) & ~self.v_walls[self.b_idx, r, torch.clamp(c+1, max=7)]
        
        mask[:, 0] = base_up & (~op_up | jump_up_ok)
        mask[:, 1] = base_down & (~op_down | jump_down_ok)
        mask[:, 2] = base_left & (~op_left | jump_left_ok)
        mask[:, 3] = base_right & (~op_right | jump_right_ok)
        
        mask[:, 4] = (base_up & op_up & ~jump_up_ok & (c > 0) & ~self.v_walls[self.b_idx, torch.clamp(r-1, min=0), torch.clamp(c-1, min=0)]) | \
                     (base_left & op_left & ~jump_left_ok & (r > 0) & ~self.h_walls[self.b_idx, torch.clamp(r-1, min=0), torch.clamp(c-1, min=0)])
                     
        mask[:, 5] = (base_up & op_up & ~jump_up_ok & (c < 8) & ~self.v_walls[self.b_idx, torch.clamp(r-1, min=0), c]) | \
                     (base_right & op_right & ~jump_right_ok & (r > 0) & ~self.h_walls[self.b_idx, torch.clamp(r-1, min=0), torch.clamp(c+1, max=8)])
                     
        mask[:, 6] = (base_down & op_down & ~jump_down_ok & (c > 0) & ~self.v_walls[self.b_idx, torch.clamp(r+1, max=8), torch.clamp(c-1, min=0)]) | \
                     (base_left & op_left & ~jump_left_ok & (r < 8) & ~self.h_walls[self.b_idx, r, torch.clamp(c-1, min=0)])
                     
        mask[:, 7] = (base_down & op_down & ~jump_down_ok & (c < 8) & ~self.v_walls[self.b_idx, torch.clamp(r+1, max=8), c]) | \
                     (base_right & op_right & ~jump_right_ok & (r < 8) & ~self.h_walls[self.b_idx, r, torch.clamp(c+1, max=8)])

        has_walls = (walls_left > 0)
        
        h_overlap = self.h_walls[:, :8, :8] | self.h_walls[:, :8, 1:9] | self.centers
        valid_h = ~h_overlap & has_walls.view(-1, 1, 1)
        mask[:, 8:72] = valid_h.reshape(self.num_envs, 64)
        
        v_overlap = self.v_walls[:, :8, :8] | self.v_walls[:, 1:9, :8] | self.centers
        valid_v = ~v_overlap & has_walls.view(-1, 1, 1)
        mask[:, 72:136] = valid_v.reshape(self.num_envs, 64)
        
        # [ПАТЧ 4] Переводим абсолютную маску в каноническую для хода игрока 2
        # (перестановка — инволюция, действует в обе стороны), как в game_env.
        mask_canonical = mask[:, self.action_perm]
        mask = torch.where(is_p1.unsqueeze(1), mask, mask_canonical)

        no_moves = ~mask.any(dim=1)
        mask[no_moves, 0] = True
        
        return {'valid_actions_mask': mask}

    def batched_flood_fill(self, indices=None):
        """BFS-расстояния до целевой горизонтали для ОБОИХ игроков одним батчем.

        [FIX-SPEED] Раньше было 2 отдельных вызова (p1 и p2) с синхронизацией
        GPU->CPU (`found.all()`) на КАЖДОЙ из до 81 итераций — это десятки
        pipeline-stall'ов на каждый env.step. Теперь: один батч 2*K и проверка
        раннего выхода раз в 8 итераций.
        Возвращает (p1_dists, p2_dists) для сред indices (None = все).
        """
        if indices is None:
            indices = self.b_idx
        K = indices.shape[0]
        b2 = torch.arange(2 * K, device=self.device)

        pos_r = torch.cat([self.p1_pos[indices, 0], self.p2_pos[indices, 0]])
        pos_c = torch.cat([self.p1_pos[indices, 1], self.p2_pos[indices, 1]])
        target_r = torch.cat([
            torch.zeros(K, dtype=torch.long, device=self.device),
            torch.full((K,), 8, dtype=torch.long, device=self.device),
        ])
        h_walls = self.h_walls[indices].repeat(2, 1, 1)
        v_walls = self.v_walls[indices].repeat(2, 1, 1)

        R = torch.zeros((2 * K, 9, 9), dtype=torch.bool, device=self.device)
        R[b2, pos_r, pos_c] = True

        dists = torch.full((2 * K,), 81, dtype=torch.float32, device=self.device)
        found = torch.zeros(2 * K, dtype=torch.bool, device=self.device)

        for step in range(81):
            reached = R[b2, target_r, :].any(dim=-1)
            dists[reached & ~found] = float(step)
            found |= reached

            # [ПАТЧ 3] Идеальный сдвиг без Shape Mismatch
            R_up = torch.roll(R, shifts=-1, dims=1)
            R_up[:, -1, :] = False
            R_up = R_up & ~h_walls

            R_down = torch.roll(R, shifts=1, dims=1)
            R_down[:, 0, :] = False
            blocked_down = torch.roll(h_walls, shifts=1, dims=1)
            blocked_down[:, 0, :] = False
            R_down = R_down & ~blocked_down

            R_left = torch.roll(R, shifts=-1, dims=2)
            R_left[:, :, -1] = False
            R_left = R_left & ~v_walls

            R_right = torch.roll(R, shifts=1, dims=2)
            R_right[:, :, 0] = False
            blocked_right = torch.roll(v_walls, shifts=1, dims=2)
            blocked_right[:, :, 0] = False
            R_right = R_right & ~blocked_right

            R = R | R_up | R_down | R_left | R_right

        return dists[:K], dists[K:]