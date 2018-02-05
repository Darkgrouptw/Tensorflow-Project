class ParamsConfig:
    # 亂數相關
    RandomNoMoveMaximum = 30                                    # 最多可以幾步不動
    RandomSeed = 123                                            # 亂數種子

    # 跟 Replay Memory 相關的
    MemorySize = 10000
    batchSize = 32

    # 螢幕大小
    imgRows = 105
    imgCols = 80