import numpy as np
import gymnasium as gym
from gymnasium import spaces
from numba import njit

@njit(cache=True)
def bfs_get_path(start_r, start_c, target_row, v_walls, h_walls):
    visited = np.zeros((9, 9), dtype=np.bool_)
    parent_r = np.full((9, 9), -1, dtype=np.int32)
    parent_c = np.full((9, 9), -1, dtype=np.int32)
    queue_r = np.zeros(81, dtype=np.int32)
    queue_c = np.zeros(81, dtype=np.int32)

    head = 0; tail = 0
    queue_r[tail] = start_r; queue_c[tail] = start_c; tail += 1
    visited[start_r, start_c] = True
    target_c = -1

    while head < tail:
        r = queue_r[head]; c = queue_c[head]; head += 1
        if r == target_row:
            target_c = c
            break
        if r > 0 and not h_walls[r - 1, c] and not visited[r - 1, c]:
            visited[r - 1, c] = True
            parent_r[r - 1, c] = r; parent_c[r - 1, c] = c
            queue_r[tail] = r - 1; queue_c[tail] = c; tail += 1
        if r < 8 and not h_walls[r, c] and not visited[r + 1, c]:
            visited[r + 1, c] = True
            parent_r[r + 1, c] = r; parent_c[r + 1, c] = c
            queue_r[tail] = r + 1; queue_c[tail] = c; tail += 1
        if c > 0 and not v_walls[r, c - 1] and not visited[r, c - 1]:
            visited[r, c - 1] = True
            parent_r[r, c - 1] = r; parent_c[r, c - 1] = c
            queue_r[tail] = r; queue_c[tail] = c - 1; tail += 1
        if c < 8 and not v_walls[r, c] and not visited[r, c + 1]:
            visited[r, c + 1] = True
            parent_r[r, c + 1] = r; parent_c[r, c + 1] = c
            queue_r[tail] = r; queue_c[tail] = c + 1; tail += 1

    if target_c == -1:
        return -1, np.zeros((8, 8), dtype=np.bool_), np.zeros((8, 8), dtype=np.bool_)

    path_h = np.zeros((8, 8), dtype=np.bool_)
    path_v = np.zeros((8, 8), dtype=np.bool_)
    curr_r = target_row; curr_c = target_c
    dist = 0
    while curr_r != start_r or curr_c != start_c:
        pr = parent_r[curr_r, curr_c]; pc = parent_c[curr_r, curr_c]
        if pr == curr_r - 1:
            if pc < 8: path_h[pr, pc] = True
            if pc > 0: path_h[pr, pc - 1] = True
        elif pr == curr_r + 1:
            if pc < 8: path_h[curr_r, pc] = True
            if pc > 0: path_h[curr_r, pc - 1] = True
        elif pc == curr_c - 1:
            if pr < 8: path_v[pr, pc] = True
            if pr > 0: path_v[pr, pc - 1] = True
        elif pc == curr_c + 1:
            if pr < 8: path_v[pr, curr_c] = True
            if pr > 0: path_v[pr, curr_c] = True
        curr_r = pr; curr_c = pc
        dist += 1
    return dist, path_h, path_v

@njit(cache=True)
def get_absolute_actions_mask(my_r, my_c, op_r, op_c, my_target, op_target, walls_left, v_walls, h_walls, centers, 
                              p1_path_h, p1_path_v, p2_path_h, p2_path_v): # Добавили пути
    mask = np.zeros(136, dtype=np.bool_)

    # ... (код для действий 0-7 остается без изменений) ...
    # 0-3: Прямые перемещения
    if my_r > 0 and not h_walls[my_r - 1, my_c]:
        if op_r == my_r - 1 and op_c == my_c:
            if my_r - 2 >= 0 and not h_walls[my_r - 2, my_c]: mask[0] = True
        else: mask[0] = True
    if my_r < 8 and not h_walls[my_r, my_c]:
        if op_r == my_r + 1 and op_c == my_c:
            if my_r + 2 <= 8 and not h_walls[my_r + 1, my_c]: mask[1] = True
        else: mask[1] = True
    if my_c > 0 and not v_walls[my_r, my_c - 1]:
        if op_r == my_r and op_c == my_c - 1:
            if my_c - 2 >= 0 and not v_walls[my_r, my_c - 2]: mask[2] = True
        else: mask[2] = True
    if my_c < 8 and not v_walls[my_r, my_c]:
        if op_r == my_r and op_c == my_c + 1:
            if my_c + 2 <= 8 and not v_walls[my_r, my_c + 1]: mask[3] = True
        else: mask[3] = True

    # 4-7: Диагональные прыжки
    if my_r > 0 and op_r == my_r - 1 and op_c == my_c and not h_walls[my_r - 1, my_c]:
        straight_blocked = (my_r - 2 < 0) or h_walls[my_r - 2, my_c]
        if straight_blocked and my_c > 0 and not v_walls[my_r - 1, my_c - 1]: mask[4] = True
        if straight_blocked and my_c < 8 and not v_walls[my_r - 1, my_c]: mask[5] = True
        
    if my_c > 0 and op_r == my_r and op_c == my_c - 1 and not v_walls[my_r, my_c - 1]:
        straight_blocked = (my_c - 2 < 0) or v_walls[my_r, my_c - 2]
        if straight_blocked and my_r > 0 and not h_walls[my_r - 1, my_c - 1]: mask[4] = True
        if straight_blocked and my_r < 8 and not h_walls[my_r, my_c - 1]: mask[6] = True
        
    if my_r < 8 and op_r == my_r + 1 and op_c == my_c and not h_walls[my_r, my_c]:
        straight_blocked = (my_r + 2 > 8) or h_walls[my_r + 1, my_c]
        if straight_blocked and my_c > 0 and not v_walls[my_r + 1, my_c - 1]: mask[6] = True
        if straight_blocked and my_c < 8 and not v_walls[my_r + 1, my_c]: mask[7] = True
        
    if my_c < 8 and op_r == my_r and op_c == my_c + 1 and not v_walls[my_r, my_c]:
        straight_blocked = (my_c + 2 > 8) or v_walls[my_r, my_c + 1]
        if straight_blocked and my_r > 0 and not h_walls[my_r - 1, my_c + 1]: mask[5] = True
        if straight_blocked and my_r < 8 and not h_walls[my_r, my_c + 1]: mask[7] = True


    # 8-135: Установка стен с Fast Pass
    # 8-135: Установка стен (Убран BFS)
    if walls_left > 0:
        for r in range(8):
            for c in range(8):
                if centers[r, c]: continue

                # Горизонтальные стены
                if not h_walls[r, c] and not h_walls[r, c+1]:
                    mask[8 + r*8 + c] = True

                # Вертикальные стены
                if not v_walls[r, c] and not v_walls[r+1, c]:
                    mask[72 + r*8 + c] = True

    return mask

class QuoridorEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(6, 9, 9), dtype=np.float32)
        self.action_space = spaces.Discrete(136)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.p1_pos = np.array([8, 4], dtype=np.int32)
        self.p2_pos = np.array([0, 4], dtype=np.int32)
        self.p1_walls = 10
        self.p2_walls = 10
        self.v_walls = np.zeros((9, 9), dtype=np.bool_)
        self.h_walls = np.zeros((9, 9), dtype=np.bool_)
        self.centers = np.zeros((8, 8), dtype=np.bool_)
        self.turn = 1
        self.step_count = 0
        self.p1_last_dist, self.p1_path_h, self.p1_path_v = bfs_get_path(self.p1_pos[0], self.p1_pos[1], 0, self.v_walls, self.h_walls)
        self.p2_last_dist, self.p2_path_h, self.p2_path_v = bfs_get_path(self.p2_pos[0], self.p2_pos[1], 8, self.v_walls, self.h_walls)
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.step_count += 1
        absolute_action = self._map_action_to_absolute(action)
        pos = self.p1_pos if self.turn == 1 else self.p2_pos
        op_pos = self.p2_pos if self.turn == 1 else self.p1_pos

        my_target = 0 if self.turn == 1 else 8
        op_target = 8 if self.turn == 1 else 0
        walls_left = self.p1_walls if self.turn == 1 else self.p2_walls

        # === ИСПРАВЛЕННЫЙ БЛОК ===
        # Прокидываем p1_path_h, p1_path_v, p2_path_h, p2_path_v
        abs_mask = get_absolute_actions_mask(
            pos[0], pos[1], op_pos[0], op_pos[1], my_target, op_target, walls_left,
            self.v_walls, self.h_walls, self.centers,
            self.p1_path_h, self.p1_path_v, self.p2_path_h, self.p2_path_v
        )
        # =========================

        if not abs_mask[absolute_action]:
            return self._get_obs(), -1.0, True, False, self._get_info()

        if absolute_action < 4:
            step_size = self._get_step_size(pos, op_pos, absolute_action)
            if absolute_action == 0: pos[0] -= step_size
            elif absolute_action == 1: pos[0] += step_size
            elif absolute_action == 2: pos[1] -= step_size
            elif absolute_action == 3: pos[1] += step_size
        elif absolute_action < 8:
            if absolute_action == 4:   pos[0] -= 1; pos[1] -= 1
            elif absolute_action == 5: pos[0] -= 1; pos[1] += 1
            elif absolute_action == 6: pos[0] += 1; pos[1] -= 1
            elif absolute_action == 7: pos[0] += 1; pos[1] += 1
        elif absolute_action < 72:
            idx = absolute_action - 8
            r, c = idx // 8, idx % 8
            self.h_walls[r, c] = True; self.h_walls[r, c+1] = True
            self.centers[r, c] = True
            if self.turn == 1: self.p1_walls -= 1
            else: self.p2_walls -= 1
        else:
            idx = absolute_action - 72
            r, c = idx // 8, idx % 8
            self.v_walls[r, c] = True; self.v_walls[r+1, c] = True
            self.centers[r, c] = True
            if self.turn == 1: self.p1_walls -= 1
            else: self.p2_walls -= 1

        reward, terminated = self._calculate_reward()
        self.turn = 2 if self.turn == 1 else 1
        truncated = self.step_count > 400
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_step_size(self, pos, op_pos, action):
        if action == 0 and pos[0] - 1 == op_pos[0] and pos[1] == op_pos[1]: return 2
        if action == 1 and pos[0] + 1 == op_pos[0] and pos[1] == op_pos[1]: return 2
        if action == 2 and pos[0] == op_pos[0] and pos[1] - 1 == op_pos[1]: return 2
        if action == 3 and pos[0] == op_pos[0] and pos[1] + 1 == op_pos[1]: return 2
        return 1

    def _calculate_reward(self):
        if self.p1_pos[0] == 0: return 10.0, True 
        if self.p2_pos[0] == 8: return 10.0, True 

        # Записываем новые пути в стейт среды
        p1_dist, self.p1_path_h, self.p1_path_v = bfs_get_path(self.p1_pos[0], self.p1_pos[1], 0, self.v_walls, self.h_walls)
        p2_dist, self.p2_path_h, self.p2_path_v = bfs_get_path(self.p2_pos[0], self.p2_pos[1], 8, self.v_walls, self.h_walls)

        delta_p1 = self.p1_last_dist - p1_dist
        delta_p2 = self.p2_last_dist - p2_dist
        
        if self.turn == 1:
            reward = (delta_p1 - delta_p2) * 0.01 - 0.01
        else:
            reward = (delta_p2 - delta_p1) * 0.01 - 0.01

        self.p1_last_dist, self.p2_last_dist = p1_dist, p2_dist
        return reward, False

    def _get_obs(self):
        obs = np.zeros((6, 9, 9), dtype=np.float32)
        if self.turn == 1:
            obs[0, self.p1_pos[0], self.p1_pos[1]] = 1.0
            obs[1, self.p2_pos[0], self.p2_pos[1]] = 1.0
            obs[2] = self.h_walls.astype(np.float32)
            obs[3] = self.v_walls.astype(np.float32)
            obs[4] = np.full((9, 9), self.p1_walls / 10.0, dtype=np.float32)
            obs[5] = np.full((9, 9), self.p2_walls / 10.0, dtype=np.float32)
        else:
            obs[0, 8 - self.p2_pos[0], 8 - self.p2_pos[1]] = 1.0
            obs[1, 8 - self.p1_pos[0], 8 - self.p1_pos[1]] = 1.0
            for r in range(8):
                for c in range(9):
                    if self.h_walls[r, c]:
                        obs[2, 7 - r, 8 - c] = 1.0
            for r in range(9):
                for c in range(8):
                    if self.v_walls[r, c]:
                        obs[3, 8 - r, 7 - c] = 1.0
            obs[4] = np.full((9, 9), self.p2_walls / 10.0, dtype=np.float32)
            obs[5] = np.full((9, 9), self.p1_walls / 10.0, dtype=np.float32)
        return obs

    def _get_info(self):
        my_r, my_c = (self.p1_pos if self.turn == 1 else self.p2_pos)
        op_r, op_c = (self.p2_pos if self.turn == 1 else self.p1_pos)
        my_target = 0 if self.turn == 1 else 8
        op_target = 8 if self.turn == 1 else 0
        walls_left = self.p1_walls if self.turn == 1 else self.p2_walls

        abs_mask = get_absolute_actions_mask(
            my_r, my_c, op_r, op_c, my_target, op_target, walls_left,
            self.v_walls, self.h_walls, self.centers,
            self.p1_path_h, self.p1_path_v, self.p2_path_h, self.p2_path_v # Передаем кэш
        )
        can_mask = self._map_mask_to_canonical(abs_mask) if self.turn == 2 else abs_mask
        return {'valid_actions_mask': can_mask}

    def _map_action_to_absolute(self, action):
        if self.turn == 1: return action
        if action == 0: return 1
        if action == 1: return 0
        if action == 2: return 3
        if action == 3: return 2
        if action == 4: return 7
        if action == 5: return 6
        if action == 6: return 5
        if action == 7: return 4
        if action < 72:
            r, c = (action - 8) // 8, (action - 8) % 8
            return 8 + (7-r)*8 + (7-c)
        r, c = (action - 72) // 8, (action - 72) % 8
        return 72 + (7-r)*8 + (7-c)

    def _map_mask_to_canonical(self, abs_mask):
        can_mask = np.zeros(136, dtype=np.bool_)
        can_mask[0], can_mask[1] = abs_mask[1], abs_mask[0]
        can_mask[2], can_mask[3] = abs_mask[3], abs_mask[2]
        can_mask[4], can_mask[5] = abs_mask[7], abs_mask[6]
        can_mask[6], can_mask[7] = abs_mask[5], abs_mask[4]
        for r in range(8):
            for c in range(8):
                can_mask[8 + r*8 + c] = abs_mask[8 + (7-r)*8 + (7-c)]
                can_mask[72 + r*8 + c] = abs_mask[72 + (7-r)*8 + (7-c)]
        return can_mask