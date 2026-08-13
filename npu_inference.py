import time
import numpy as np
import coremltools as ct

class QuoridorNPUInference:
    def __init__(self, model_path="checkpoints/QuoridorNPU.mlpackage"):
        print(f"Загрузка Core ML модели из {model_path}...")

        try:
            self.model = ct.models.MLModel(
                model_path, 
                compute_units=ct.ComputeUnit.ALL
            )
        except Exception as e:
            print(f"\n[ERROR] Не удалось загрузить Core ML модель: {e}")
            print("[HINT] Убедитесь, что:")
            print("  1. Файл checkpoints/QuoridorNPU.mlpackage существует")
            print("  2. Вы запустили: python export_to_coreml.py")
            print("  3. Установлена совместимая версия coremltools")
            raise RuntimeError(f"Core ML model load failed: {model_path}") from e

        print("Прогрев Apple Neural Engine (Warmup)...")
        dummy_obs = np.zeros((1, 6, 9, 9), dtype=np.float32)
        dummy_mask = np.ones((1, 136), dtype=np.float32)
        for _ in range(15):
            _ = self.model.predict({"obs": dummy_obs, "action_mask": dummy_mask})
        print("Warmup завершен.")

    def predict_action(self, obs_np: np.ndarray, mask_np: np.ndarray) -> int:
        if obs_np.ndim == 3:
            obs_np = np.expand_dims(obs_np, axis=0)

        if mask_np.ndim == 1:
            mask_np = np.expand_dims(mask_np, axis=0)

        if mask_np.shape[-1] != 136:
            raise ValueError(f"Размер маски должен быть 136, получено {mask_np.shape[-1]}")

        mask_np = mask_np.astype(np.float32)
        mask_np = np.clip(mask_np, 0.0, 1.0)

        if mask_np.sum() == 0:
            mask_np[0, 0] = 1.0
            print("[WARN] Все действия замаскированы! Принудительно разрешён action 0.")

        predictions = self.model.predict({"obs": obs_np, "action_mask": mask_np})
        logits = predictions["logits"].flatten()

        return int(np.argmax(logits))

if __name__ == "__main__":
    model_path = "checkpoints/QuoridorNPU.mlpackage"

    try:
        engine = QuoridorNPUInference(model_path=model_path)
    except Exception as e:
        print(f"Не удалось загрузить модель {model_path}.")
        print("Запустите сначала: python export_to_coreml.py")
        exit(1)

    test_obs = np.random.randn(6, 9, 9).astype(np.float32)
    test_mask = np.random.choice([True, False], size=(136,), p=[0.2, 0.8])
    test_mask[0] = True

    num_runs = 1000
    latencies = []

    print(f"\nЗапуск замера задержки ANE ({num_runs} итераций)...")
    for _ in range(num_runs):
        t0 = time.perf_counter()
        action = engine.predict_action(test_obs, test_mask)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    p99_lat = np.percentile(latencies, 99)

    print("\n" + "="*45)
    print("      APPLE NEURAL ENGINE BENCHMARK      ")
    print("="*45)
    print(f"Средняя задержка (Avg) : {avg_lat:.3f} мс")
    print(f"95-й перцентиль (P95)  : {p95_lat:.3f} мс")
    print(f"99-й перцентиль (P99)  : {p99_lat:.3f} мс")
    print(f"Пропускная способность : {1000.0 / avg_lat:.0f} FPS")
    print("="*45)