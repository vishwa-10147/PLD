from modules.replay import KITTIReplay
from modules.fusion_engine import FusionEngine
from modules.localization_features import LocalizationFeatures
from modules.localization_model import LocalizationModel


print("=" * 60)
print("Loading KITTI Dataset")
print("=" * 60)

replay = KITTIReplay()
replay.load()

fusion = FusionEngine()

builder = LocalizationFeatures()

model = LocalizationModel()

packet = replay.next_frame()

fusion_result = fusion.process(packet)

features = builder.build(fusion_result)

feature_df = builder.to_dataframe(features)

prediction = model.predict(feature_df)

print("\nFeature DataFrame\n")

print(feature_df)

print("\nLocalization Reliability\n")

print(prediction)