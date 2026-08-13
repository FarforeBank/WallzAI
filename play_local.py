import sys
import pygame
import numpy as np

from game_env import QuoridorEnv
from npu_inference import QuoridorNPUInference

# ==========================================
# НАСТРОЙКИ PYGAME И ГРАФИКИ
# ==========================================
CELL_SIZE = 60
GAP_SIZE = 15
MARGIN = 50
BOARD_SIZE = 9

WIDTH = MARGIN * 2 + BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * GAP_SIZE
HEIGHT = MARGIN * 2 + BOARD_SIZE * CELL_SIZE + (BOARD_SIZE - 1) * GAP_SIZE

# Цвета
BG_COLOR = (240, 240, 240)
CELL_COLOR = (200, 200, 200)
CELL_HOVER_COLOR = (220, 235, 255)
P1_COLOR = (50, 150, 255)
P2_COLOR = (255, 100, 100)
WALL_COLOR = (50, 50, 50)
WALL_HOVER_COLOR = (100, 200, 100)
WALL_INVALID_HOVER = (255, 100, 100)

def get_cell_rect(r, c):
    x = MARGIN + c * (CELL_SIZE + GAP_SIZE)
    y = MARGIN + r * (CELL_SIZE + GAP_SIZE)
    return pygame.Rect(x, y, CELL_SIZE, CELL_SIZE)

def get_h_wall_rect(r, c):
    x = MARGIN + c * (CELL_SIZE + GAP_SIZE)
    y = MARGIN + r * (CELL_SIZE + GAP_SIZE) + CELL_SIZE
    return pygame.Rect(x, y, CELL_SIZE * 2 + GAP_SIZE, GAP_SIZE)

def get_v_wall_rect(r, c):
    x = MARGIN + c * (CELL_SIZE + GAP_SIZE) + CELL_SIZE
    y = MARGIN + r * (CELL_SIZE + GAP_SIZE)
    return pygame.Rect(x, y, GAP_SIZE, CELL_SIZE * 2 + GAP_SIZE)

def get_action_from_click(pos, p1_pos):
    x, y = pos

    # 1. Проверка клика по ячейкам (Шаг пешкой, прыжок или диагональный прыжок)
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if get_cell_rect(r, c).collidepoint(x, y):
                pr, pc = p1_pos
                dr, dc = r - pr, c - pc

                if dr < 0 and dc == 0: return 0   # Up
                if dr > 0 and dc == 0: return 1   # Down
                if dr == 0 and dc < 0: return 2   # Left
                if dr == 0 and dc > 0: return 3   # Right
                if dr < 0 and dc < 0: return 4    # Up-Left (diagonal)
                if dr < 0 and dc > 0: return 5    # Up-Right (diagonal)
                if dr > 0 and dc < 0: return 6    # Down-Left (diagonal)
                if dr > 0 and dc > 0: return 7    # Down-Right (diagonal)
                return None

    # 2. Проверка клика по горизонтальным зазорам (H-Wall: 8-71)
    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            if get_h_wall_rect(r, c).collidepoint(x, y):
                return 8 + r * 8 + c

    # 3. Проверка клика по вертикальным зазорам (V-Wall: 72-135)
    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            if get_v_wall_rect(r, c).collidepoint(x, y):
                return 72 + r * 8 + c

    return None

def draw_board(screen, env, hover_action=None, valid_mask=None):
    screen.fill(BG_COLOR)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            cell_rect = get_cell_rect(r, c)
            if hover_action is not None and valid_mask is not None:
                pr, pc = env.p1_pos
                dr, dc = r - pr, c - pc
                action = None
                if dr < 0 and dc == 0: action = 0
                elif dr > 0 and dc == 0: action = 1
                elif dr == 0 and dc < 0: action = 2
                elif dr == 0 and dc > 0: action = 3
                elif dr < 0 and dc < 0: action = 4
                elif dr < 0 and dc > 0: action = 5
                elif dr > 0 and dc < 0: action = 6
                elif dr > 0 and dc > 0: action = 7
                if action == hover_action and valid_mask[action]:
                    pygame.draw.rect(screen, CELL_HOVER_COLOR, cell_rect, border_radius=8)
                else:
                    pygame.draw.rect(screen, CELL_COLOR, cell_rect, border_radius=8)
            else:
                pygame.draw.rect(screen, CELL_COLOR, cell_rect, border_radius=8)

    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            h_rect = get_h_wall_rect(r, c)
            action = 8 + r * 8 + c
            if hover_action == action and valid_mask is not None:
                color = WALL_HOVER_COLOR if valid_mask[action] else WALL_INVALID_HOVER
                pygame.draw.rect(screen, color, h_rect, border_radius=4)
            elif env.centers[r, c] and env.h_walls[r, c] and env.h_walls[r, c+1]:
                pygame.draw.rect(screen, WALL_COLOR, h_rect, border_radius=4)

    for r in range(BOARD_SIZE - 1):
        for c in range(BOARD_SIZE - 1):
            v_rect = get_v_wall_rect(r, c)
            action = 72 + r * 8 + c
            if hover_action == action and valid_mask is not None:
                color = WALL_HOVER_COLOR if valid_mask[action] else WALL_INVALID_HOVER
                pygame.draw.rect(screen, color, v_rect, border_radius=4)
            elif env.centers[r, c] and env.v_walls[r, c] and env.v_walls[r+1, c]:
                pygame.draw.rect(screen, WALL_COLOR, v_rect, border_radius=4)

    def draw_pawn(pos, color):
        rect = get_cell_rect(pos[0], pos[1])
        center = (rect.x + CELL_SIZE // 2, rect.y + CELL_SIZE // 2)
        pygame.draw.circle(screen, color, center, CELL_SIZE // 2 - 8)

    draw_pawn(env.p1_pos, P1_COLOR)
    draw_pawn(env.p2_pos, P2_COLOR)

    font = pygame.font.SysFont(None, 24)
    p1_text = font.render(f"Игрок 1 Стены: {env.p1_walls}", True, P1_COLOR)
    p2_text = font.render(f"Бот Стены: {env.p2_walls}", True, P2_COLOR)
    screen.blit(p1_text, (MARGIN, HEIGHT - MARGIN // 1.5))
    screen.blit(p2_text, (MARGIN, 10))

    pygame.display.flip()

def main():
    try:
        engine = QuoridorNPUInference("checkpoints/QuoridorNPU.mlpackage")
    except Exception as e:
        print("\033[31mФайл CoreML модели не найден! Сначала запустите python export_to_coreml.py\033[0m")
        sys.exit(1)

    env = QuoridorEnv()
    obs, info = env.reset()

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Quoridor AI - Local GUI Test")

    running = True
    game_over = False
    hover_action = None
    valid_mask = info.get('valid_actions_mask', None)

    draw_board(screen, env, hover_action, valid_mask)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEMOTION and env.turn == 1 and not game_over:
                hover_action = get_action_from_click(event.pos, env.p1_pos)
                if hover_action is not None:
                    draw_board(screen, env, hover_action, valid_mask)
                elif hover_action is None and valid_mask is not None:
                    draw_board(screen, env, None, valid_mask)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and env.turn == 1 and not game_over:
                action = get_action_from_click(event.pos, env.p1_pos)

                if action is not None:
                    mask = env._get_info()['valid_actions_mask']

                    if mask[action]:
                        print(f"[Человек] Выполнил действие: {action}")
                        obs, reward, terminated, truncated, info = env.step(action)
                        valid_mask = info.get('valid_actions_mask', None)
                        hover_action = None
                        draw_board(screen, env, None, valid_mask)

                        if terminated or truncated:
                            game_over = True
                            print(f"🏁 Игра окончена! Награда: {reward}")
                    else:
                        print(f"⚠️ Предупреждение: Невозможный ход (Action: {action}). Попробуйте еще раз.")

        if env.turn == 2 and not game_over:
            pygame.time.delay(300)

            mask = env._get_info()['valid_actions_mask']
            action = engine.predict_action(obs, mask)

            action_type = "Шаг" if action < 4 else ("Диаг." if action < 8 else ("H-Стена" if action < 72 else "V-Стена"))
            print(f"[Бот AI] Выбрал действие: {action} ({action_type})")

            obs, reward, terminated, truncated, info = env.step(action)
            valid_mask = info.get('valid_actions_mask', None)
            hover_action = None
            draw_board(screen, env, None, valid_mask)

            if terminated or truncated:
                game_over = True
                print(f"🏁 Игра окончена! Награда: {reward}")

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()