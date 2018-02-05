import numpy as np

class ReplayMemory:
    def __init__(
            self,
            config
    ):
        # 建造資料庫
        self.MemoryObservation = np.empty(
            config.MemorySize,
            config.imgRows,
            config.imgCols,
            dtype=np.float16
        )
        self.MemoryAction = np.empty(
            config.MemorySize,
            dtype=np.uint8
        )
        self.MemoryReward = np.empty(
            config.MemorySize
        )
        self.Memory

        # Sample 參數
        self.batchSize = config.batchSize

        # 索引
        self.MemoryCount = 0                                # 紀錄目前的大小
        self.MemoryCurrentIndex = 0                         # 紀錄目前存到哪裡 （因為 numpy 是要先給固定大小，所以會導致無法像 list 一樣）

    def Add(self):
