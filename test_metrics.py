from modules.replay import KITTIReplay
from modules.sensor_metrics import SensorMetrics

replay = KITTIReplay()

replay.load()

packet = replay.next_frame()

metrics = SensorMetrics()

result = metrics.compute(packet)

print(result)