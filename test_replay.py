from modules.replay import KITTIReplay

replay = KITTIReplay()

replay.load()

print()

print("Frames :", len(replay))

print()

packet = replay.next_frame()

print(packet.keys())

print()

print(packet["frame_id"])

print(packet["timestamp"])