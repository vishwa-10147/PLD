from modules.replay import KITTIReplay
from modules.ekf_state import EKFState

replay = KITTIReplay()

replay.load()

packet = replay.next_frame()

state = EKFState()

state.initialize(packet)

print("State Vector")

print(state.get_state())

print()

print("Covariance")

print(state.get_covariance())