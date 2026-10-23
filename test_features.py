from modules.replay import KITTIReplay
from modules.fusion_engine import FusionEngine
from modules.localization_features import LocalizationFeatures

replay = KITTIReplay()
replay.load()

fusion = FusionEngine()
builder = LocalizationFeatures()

packet = replay.next_frame()

result = fusion.process(packet)

features = builder.build(result)

print("\nFeature Dictionary\n")

for k, v in features.items():
    print(f"{k:20} : {v}")

print("\nFeature Vector\n")

print(builder.to_numpy(features))