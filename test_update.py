from modules.replay import KITTIReplay
from modules.ekf_state import EKFState
from modules.ekf_predict import EKFPredict
from modules.adaptive_ekf import AdaptiveEKF
from modules.ekf_update import EKFUpdate

replay = KITTIReplay()
replay.load()

packet = replay.next_frame()

# -------------------------
# Initialize EKF
# -------------------------

state = EKFState()
state.initialize(packet)

# -------------------------
# Prediction
# -------------------------

predictor = EKFPredict(dt=0.1)
predictor.predict(state)

# -------------------------
# Adaptive covariance
# -------------------------

weights = {

    "camera": 0.3386,

    "gps": 0.3346,

    "imu": 0.3268

}

adaptive = AdaptiveEKF()

cov = adaptive.compute(weights)

# -------------------------
# Update
# -------------------------

updater = EKFUpdate()

result = updater.update(

    state,

    packet,

    cov["R"]

)

print("\nUpdated State\n")
print(result["state"])

print("\nKalman Gain\n")
print(result["kalman_gain"])

print("\nInnovation\n")
print(result["innovation"])

print("\nUpdated Covariance\n")
print(result["covariance"])