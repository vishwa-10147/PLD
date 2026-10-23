from modules.adaptive_weights import AdaptiveWeights

quality = {

    "camera_score":0.86,

    "gps_score":0.85,

    "imu_score":0.83

}

model = AdaptiveWeights()

print(model.compute(quality))