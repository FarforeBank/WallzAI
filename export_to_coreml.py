import os
import torch
import coremltools as ct
from model import QuoridorActorCritic

def export():
    checkpoint_path = "checkpoints/best_model.pt"
    fallback_checkpoint = "quoridor_ppo_checkpoint.pt"
    output_path = "checkpoints/QuoridorNPU.mlpackage"

    os.makedirs("checkpoints", exist_ok=True)

    print("Инициализация базовой модели PyTorch...")
    base_agent = QuoridorActorCritic(in_channels=6, hidden_dim=256, num_actions=136)

    target_path = checkpoint_path if os.path.exists(checkpoint_path) else fallback_checkpoint

    if os.path.exists(target_path):
        print(f"Загрузка весов из {target_path}...")
        checkpoint = torch.load(target_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        base_agent.load_state_dict(state_dict)
        print("Весы успешно загружены.")
    else:
        print(f"ВНИМАНИЕ: Файлы чекпоинтов не найдены! Экспорт случайных весов для теста.")

    base_agent.eval()

    dummy_obs = torch.randn(1, 6, 9, 9, dtype=torch.float32)
    dummy_mask = torch.ones(1, 136, dtype=torch.float32)
    dummy_mask[0, ::2] = 0.0

    traced_model = torch.jit.trace(base_agent, (dummy_obs, dummy_mask))

    assert len(list(traced_model.graph.inputs())) == 3, "Traced model должен принимать 2 входа"
    print(f"[OK] Трассировка успешна. Входы: {len(list(traced_model.graph.inputs())) - 1}, Выходы: {len(list(traced_model.graph.outputs()))}")

    print("Конвертация в Core ML (.mlpackage)...")

    inputs = [
        ct.TensorType(name="obs", shape=ct.Shape(shape=(1, 6, 9, 9)), dtype=float),
        ct.TensorType(name="action_mask", shape=ct.Shape(shape=(1, 136)), dtype=float)
    ]

    outputs = [
        ct.TensorType(name="logits", dtype=float),
        ct.TensorType(name="value", dtype=float)
    ]

    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS13
    )

    mlmodel.save(output_path)
    print(f"Успешно! Модель сохранена в: {output_path}")

if __name__ == "__main__":
    export()