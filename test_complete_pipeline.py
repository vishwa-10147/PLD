from modules.replay import KITTIReplay
from modules.fusion_engine import FusionEngine

replay = KITTIReplay()
replay.load()

fusion = FusionEngine()

packet = replay.next_frame()

result = fusion.process(packet)

print("\n========== SENSOR QUALITY ==========")
print(result["quality"])

print("\n========== ADAPTIVE WEIGHTS ==========")
print(result["weights"])

print("\n========== FUSED LOCALIZATION ==========")
print(result["fusion"])

print("\n========== LOCALIZATION FEATURES ==========")
print(result["features"])

print("\n========== LOCALIZATION RELIABILITY ==========")
print(result["localization"])