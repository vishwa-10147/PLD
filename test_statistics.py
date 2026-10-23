from modules.statistics import DatasetStatistics

stats = DatasetStatistics(
    "2011_09_26_drive_0022_sync"
)

result = stats.save(
    "outputs/statistics/kitti_stats.json"
)

print(result)