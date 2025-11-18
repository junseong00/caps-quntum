import time
import random
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Flask 앱 초기화
app = Flask(__name__)

# -----------------------------------------------------------------
# CORS 설정
# -----------------------------------------------------------------
allowed_origins = [
    "http://127.0.0.1:5500",
    "http://localhost:5500",
    "http://127.0.0.1:5501",
    "http://localhost:5501"
]

CORS(app, resources={r"/simulate": {"origins": allowed_origins}})

# -----------------------------------------------------------------
# index.html 서빙
# -----------------------------------------------------------------
@app.route('/')
def serve_index():
    return send_from_directory('.', 'index.html')

# -----------------------------------------------------------------
# 시뮬레이션 함수
# -----------------------------------------------------------------
def run_simulation(qubits, strength, model):
    """
    실제 모델로 대체될 시뮬레이션 함수 (현재는 가상 로직)
    """
    pulse_steps = 50
    optimal_pulse_data = []

    # 기본 펄스
    base_pulse = np.sin(np.linspace(0, np.pi * 2, pulse_steps)) * strength
    noise = np.random.randn(pulse_steps) * 0.1 * (qubits / 50)
    optimal_pulse = base_pulse + noise
    optimal_pulse_data = list(optimal_pulse)

    # 모델별 결과 변경
    final_energy = 100.0
    learning_time = 5.0
    model_params = 10000

    if model == 'MLP':
        final_energy += 30.0
        learning_time += 5.0
        model_params = 50000

        base_pulse = np.sin(np.linspace(0, np.pi * 2, pulse_steps)) * strength * 0.8
        noise = np.random.randn(pulse_steps) * 0.15
        optimal_pulse_data = list(base_pulse + noise)

    elif model == 'MLP (Set)':
        final_energy += 15.0
        learning_time += 10.0
        model_params = 80000

        base_pulse = np.sin(np.linspace(0, np.pi * 3, pulse_steps)) * strength
        noise = np.random.randn(pulse_steps) * 0.1
        optimal_pulse_data = list(base_pulse + noise)

    elif model == 'GCN':
        final_energy -= 10.0
        learning_time += 15.0
        model_params = 150000

        base_pulse = np.sin(np.linspace(0, np.pi * 2, pulse_steps)) * strength
        noise = np.random.randn(pulse_steps) * 0.1 * (qubits / 8.0)
        optimal_pulse_data = list(base_pulse + noise)

    elif model == 'GAT':
        final_energy -= 15.0
        learning_time += 20.0
        model_params = 250000

        base_pulse = np.cos(np.linspace(0, np.pi * 2, pulse_steps)) * strength
        noise = np.random.randn(pulse_steps) * 0.05
        optimal_pulse_data = list(base_pulse + noise)

    # 공통 조정값
    final_energy += (qubits * 0.5)
    learning_time += (qubits * 0.8)
    model_params += (model_params * (qubits / 10))
    final_energy -= (strength * 5)
    learning_time += (strength * 1.2)

    time.sleep(1.2)

    return {
        "optimalPulse": optimal_pulse_data,
        "finalEnergy": round(final_energy, 2),
        "learningTime": round(learning_time, 1),
        "modelParams": int(model_params)
    }

# -----------------------------------------------------------------
# /simulate API
# -----------------------------------------------------------------
@app.route('/simulate', methods=['POST'])
def handle_simulation():
    try:
        data = request.json

        # 기본값 수정됨 (10 → 8)
        qubits = int(data.get('qubits', 8))
        strength = float(data.get('strength', 1.0))
        model = data.get('model', 'GCN')

        results = run_simulation(qubits, strength, model)

        return jsonify(results)

    except Exception as e:
        print(f"Error during simulation: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------------------------------------
# 서버 실행
# -----------------------------------------------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
