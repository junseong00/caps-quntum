import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from stable_baselines3 import PPO

# -----------------------------------------------------------------
# 경로 및 모듈 설정
# -----------------------------------------------------------------
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
FINAL_FINAL_ROOT = PROJECT_ROOT / "gnn_quantum_battery_final_final"
RESULTS_DIR = FINAL_FINAL_ROOT / "results"
SRC_DIR = FINAL_FINAL_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from battery_rl.environments import (
    GraphObservationQuantumBatteryEnv,
    FullStateQuantumBatteryEnv,
)

# Flask 앱 초기화
app = Flask(__name__)

# -----------------------------------------------------------------
# CORS 설정
# -----------------------------------------------------------------
ALLOWED_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5501",
    "http://localhost:5501",
    "http://127.0.0.1:5502",
    "http://localhost:5502",
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "null",
]

CORS(app, resources={r"/simulate": {"origins": ALLOWED_ORIGINS}}, supports_credentials=False)

# -----------------------------------------------------------------
# 실험 데이터 적재 도우미
# -----------------------------------------------------------------
MODEL_NAME_MAP = {
    "MLP": "mlp",
    "MLP (Set)": "set",
    "GCN": "gcn",
    "GAT": "gat",
}

MODEL_POLICY_TEMPLATE = {
    "mlp": "mlp_n{n}.zip",
    "set": "set_n{n}.zip",
    "gcn": "gcn_n{n}.zip",
    "gat": "gat_n{n}.zip",
}


def _validate_paths() -> None:
    if not FINAL_FINAL_ROOT.exists():
        raise FileNotFoundError("gnn_quantum_battery_final_final 디렉터리를 찾을 수 없습니다.")
    if not (RESULTS_DIR / "scalability_metrics.csv").exists():
        raise FileNotFoundError("scalability_metrics.csv 파일이 존재하지 않습니다.")


_validate_paths()

SCALABILITY_DF = pd.read_csv(RESULTS_DIR / "scalability_metrics.csv")
PARAMETER_DF = pd.read_csv(RESULTS_DIR / "model_parameters.csv")


def _metrics_lookup(model_key: str, n_qubits: int) -> dict:
    subset = SCALABILITY_DF[(SCALABILITY_DF["model_type"] == model_key) & (SCALABILITY_DF["n_qubits"] == n_qubits)]
    if subset.empty:
        raise ValueError(f"{model_key} 모델에 대한 n={n_qubits} 실험 데이터가 없습니다.")
    row = subset.iloc[0]
    return {
        "final_energy": float(row["mean_final_energy"]),
        "training_time": float(row["training_time_sec"]),
    }


def _parameter_lookup(model_key: str, n_qubits: int) -> int:
    subset = PARAMETER_DF[(PARAMETER_DF["model_type"] == model_key) & (PARAMETER_DF["n_qubits"] == n_qubits)]
    if subset.empty:
        raise ValueError(f"{model_key} 모델에 대한 n={n_qubits} 파라미터 데이터가 없습니다.")
    return int(subset.iloc[0]["param_count"])


@lru_cache(maxsize=16)
def _load_policy(model_key: str, n_qubits: int) -> PPO:
    policy_file = MODEL_POLICY_TEMPLATE[model_key].format(n=n_qubits)
    policy_path = RESULTS_DIR / policy_file
    if not policy_path.exists():
        raise FileNotFoundError(f"정책 파일을 찾을 수 없습니다: {policy_path}")
    return PPO.load(policy_path, device="cpu")


def run_simulation(qubits: int, strength: float, frontend_model_name: str) -> dict:
    model_key = MODEL_NAME_MAP.get(frontend_model_name)
    if model_key is None:
        raise ValueError(f"지원하지 않는 모델: {frontend_model_name}")

    metrics = _metrics_lookup(model_key, qubits)
    baseline_metrics = _metrics_lookup("mlp", qubits)
    baseline_energy = float(baseline_metrics["final_energy"])
    param_count = _parameter_lookup(model_key, qubits)

    policy = _load_policy(model_key, qubits)
    if model_key == "mlp":
        env = FullStateQuantumBatteryEnv(n_qubits=qubits)
    else:
        env = GraphObservationQuantumBatteryEnv(n_qubits=qubits)

    obs, info = env.reset()
    pulses: list[float] = []
    phase_angles: list[float] = []
    time_axis: list[float] = []
    stored_energy: list[float] = []
    step_energy_gain: list[float] = []
    charge_percent: list[float] = []
    final_energy_run = float(info.get("energy", 0.0))
    previous_energy = final_energy_run
    cumulative_phase = 0.0
    dt = float(getattr(env, "dt", 0.1))
    max_energy = float(qubits)

    scale = max(strength, 0.0)

    for step_index in range(env.max_steps):
        action, _ = policy.predict(obs, deterministic=True)
        scaled_action = np.clip(action * scale, -1.0, 1.0).astype(np.float32)
        obs, _, terminated, truncated, step_info = env.step(scaled_action)
        current_omega = float(step_info.get("omega", 0.0))
        pulses.append(current_omega)
        cumulative_phase += current_omega * dt
        phase_angles.append(cumulative_phase)
        time_axis.append((step_index + 1) * dt)

        energy_after = float(step_info.get("energy", final_energy_run))
        energy_delta = energy_after - previous_energy
        stored_energy.append(round(energy_after, 6))
        step_energy_gain.append(round(energy_delta, 6))
        if max_energy > 0:
            charge_percent.append(round((energy_after / max_energy) * 100.0, 4))
        else:
            charge_percent.append(0.0)

        previous_energy = energy_after
        final_energy_run = energy_after
        if terminated or truncated:
            break

    time.sleep(0.2)

    return {
        "optimalPulse": pulses,
        "finalEnergy": round(final_energy_run, 3),
        "learningTime": round(metrics["training_time"], 2),
        "modelParams": param_count,
        "baselineEnergy": round(baseline_energy, 3),
        "superabsorptionGain": round(final_energy_run / baseline_energy, 3) if baseline_energy > 0 else None,
        "phaseTrajectory": phase_angles,
        "phaseTimeAxis": time_axis,
        "storedEnergyTrajectory": stored_energy,
        "stepEnergyGainTrajectory": step_energy_gain,
        "chargePercentTrajectory": charge_percent,
        "timeAxis": time_axis,
        "timeStep": dt,
        "maxEnergy": max_energy,
    }


# -----------------------------------------------------------------
# 라우트 정의
# -----------------------------------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(str(CURRENT_DIR), "index.html")


@app.route("/simulate", methods=["POST"])
def handle_simulation():
    try:
        data = request.json or {}
        qubits = int(data.get("qubits", 8))
        strength = float(data.get("strength", 1.0))
        model = data.get("model", "GCN")

        results = run_simulation(qubits, strength, model)
        return jsonify(results)
    except Exception as exc:  # pragma: no cover - 간단한 에러 핸들링
        print(f"Error during simulation: {exc}")
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
