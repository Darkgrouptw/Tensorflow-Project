class ParamsConfig:
    # 亂數相關
    RandomNoMoveMaximum = 30                                    # 最多可以幾步不動
    RandomSeed = 123                                            # 亂數種子

    # 跟 Replay Memory 相關的
    MemorySize = 10000
    batchSize = 32

    # Epsilon
    EpsilonMax = 0.9
    EpsilonIncreasement = 0.0008

    # 學習相關參數
    LearningRate = 0.00025                                       # 2.5 e-4
    LearningDecay = 0.96
    LearningMomentum = 0.95
    LearningEpsilon = 0.01

    # 螢幕大小
    imgRows = 52
    imgCols = 40