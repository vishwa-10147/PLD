from modules.replay import KITTIReplay
from modules.fusion_engine import FusionEngine

replay = KITTIReplay()

replay.load()

fusion = FusionEngine()

for i in range(5):

    packet = replay.next_frame()

    result = fusion.process(packet)

    print("\n==============================")

    print("Frame", i)

    print("==============================")

    print("SCI")

    print(result["quality"]["sci"])

    print()

    print("Weights")

    print(result["weights"])

    print()

    print("Fusion")

    print(result["fusion"])
