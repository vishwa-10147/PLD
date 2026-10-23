from modules.replay import KITTIReplay
from modules.sensor_metrics import SensorMetrics
from modules.sensor_quality import SensorConfidence

replay = KITTIReplay()
replay.load()

metrics = SensorMetrics()
quality = SensorConfidence()

packet = replay.next_frame()

raw_metrics = metrics.compute(packet)

print("\nGPS Metrics")
print(raw_metrics["gps"])

scores = quality.compute(raw_metrics)

print("\nScores")
print(scores)