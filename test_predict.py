from modules.replay import KITTIReplay
from modules.ekf_state import EKFState
from modules.ekf_predict import EKFPredict

replay = KITTIReplay()
replay.load()

packet = replay.next_frame()

state = EKFState()
state.initialize(packet)

predictor = EKFPredict(dt=0.1)

result = predictor.predict(state)

print("Predicted State\n")
print(result["state"])

print("\nPredicted Covariance\n")
print(result["covariance"])