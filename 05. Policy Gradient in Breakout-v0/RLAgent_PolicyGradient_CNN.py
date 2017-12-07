import tensorflow as tf
import numpy as np
import os


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            imgRows,
            imgCols,
            imgNumber,
            imgChannel,
            LearningRate=0.01,
            RewardDecay=0.95,  # 衰減比例
            IsOutputGraph=False
    ):
        self.n_actions = n_actions

        np.random.seed(1)
        tf.set_random_seed(1)

        # 圖片相關
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.imgNumber = imgNumber
        self.imgChannel = imgChannel

        # 預測參數
        self.LearningRate = LearningRate
        self.RewardDecay = RewardDecay

        # 輸出檔案相關
        self.logName = "/Convolution Neural Network"

        # Ouput 好壞用的變數
        # self.epoch = 0

        self.BuildNet()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # 將 Memory Set 清空
        self.MemoryObservation = []
        self.MemoryAction = []
        self.MemoryReward = []

        # 將 Tensorboard Graph 輸出出去
        if IsOutputGraph:
            tf.summary.FileWriter("./logs" + self.logName, self.session.graph)

        # 設定要輸出分數的檔案
        # self.logFile = tf.summary.FileWriter("./logs" + self.logName)

    # Observation 跟 State 是代表同樣的事情，觀察到的東西就是 state，但是 Policy Gradient 裡面，沒有 State 的概念，是以機率來說
    # 建造 Neural Network
    # Observation 跟 State 是代表同樣的事情，觀察到的東西就是 state，但是 Policy Gradient 裡面，沒有 State 的概念，是以機率來說
    # 建造 Neural Network
    def BuildNet(self):
        # 參數設定
        # 用的 filter 數目及大小
        filter1_Size = 5
        filter1_Count = 4
        filter2_Size = 5
        filter2_Count = 8

        fc1_units = 64

        # Input
        with tf.name_scope("Input"):
            # 前 k 張圖片，用 channel 判斷
            # [batch, in_depth, in_height(rows), in_width(cols), in_channel]
            self.observationsArray = tf.placeholder(
                tf.float32,
                [None, self.imgNumber, self.imgRows, self.imgCols, self.imgChannel],
                name="observationsArray")

            # 每個 Frame 使用的動作
            self.actionsNum = tf.placeholder(tf.int32, [None], name="Action")  # 注意 [None] 跟 [None,] 是一樣的，一維的陣列，裡面裝著不知道大小的 scalar，跟 [None, 1] 不一樣，這個是 二維陣列

            # 該更新的幅度
            self.DeltaValue = tf.placeholder(tf.float32, [None], name="ActionValue")

            # 最後要輸出這一輪的到幾分用
            self.RewardValue = tf.placeholder(tf.float32, [None], name="RewardValue")

        # 亂數
        weightInit = tf.random_normal_initializer(mean=0, stddev=0.3)
        biasInit = tf.random_normal_initializer(mean=0, stddev=0.1)

        # Conv2d 會做一件事，把 Input 攤平 [filter_height * filter_width * in_channels, output_channels].
        # 所以如果要處理多個 Channel 的資訊，例如三個顏色，可能需要用到 conv3d
        # flatInput = tf.reshape(self.observationsArray,(-1, self.imgNumber * self.imgRows * self.imgCols * self.imgChannel))

        with tf.name_scope("Layer1"):
            conv1 = tf.layers.conv3d(
                inputs=self.observationsArray,
                filters=filter1_Count,
                kernel_size=filter1_Size,
                strides=1,
                padding="SAME",
                activation=tf.nn.tanh,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Conv1"
            )
            maxpooling1 = tf.layers.max_pooling3d(
                inputs=conv1,
                pool_size=3,
                strides=3,
                padding="SAME",
                name="MaxPooling1"
            )
        with tf.name_scope("Layer2"):
            conv2 = tf.layers.conv3d(
                inputs=maxpooling1,
                filters=filter2_Count,
                kernel_size=filter2_Size,
                strides=1,
                padding="SAME",
                activation=tf.nn.tanh,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Conv2"
            )
            maxpooling2 = tf.layers.max_pooling3d(
                inputs=conv2,
                pool_size=3,
                strides=3,
                padding="SAME",
                name="MaxPooling2"
            )
            
        flatData = tf.reshape(maxpooling2, [-1, 12 * 9 * 8])
        fc1 = tf.layers.dense(
            inputs=flatData,
            units=fc1_units,
            activation=tf.nn.tanh,
            kernel_initializer=weightInit,
            bias_initializer=biasInit,
            name="FC1"
        )

        output = tf.layers.dense(
            inputs=fc1,
            units=self.n_actions,
            kernel_initializer=weightInit,
            bias_initializer=biasInit,
            name="Output"
        )

        # 每個動作的機率
        self.ActionProb = tf.nn.softmax(output, name="ActionProb")

        # Loss Function
        with tf.name_scope("Loss"):
            # 將動作轉乘 One Hot 的形式，因為預測出來是以 One Hot 的方式計算 cross entropy
            actionNum_OneHot = tf.one_hot(self.actionsNum, self.n_actions)
            # cost 跟 cos1 算出來的結果不一樣，一個是已經 reduce_mean 完的結果
            # cost1 = tf.losses.softmax_cross_entropy(logits=layer2, onehot_labels=actionNum_OneHot)
            # cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=self.actionsNum)
            cost = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=actionNum_OneHot)

            # 這裡的意思是，只有剛開始不會倒的情況下， DeltaValue 就會越大，代表他這部分要越重視!!
            # 下面那個註解掉的
            # 是當時犯的錯
            # 因為 loss.softmax_cross_entropy 已經把誤差平均了
            # 但是 DeltaValue 是一個陣列 (代表好壞的一個陣列)
            # loss = cost * self.DeltaValue
            loss = tf.reduce_mean(cost * self.DeltaValue)

        # Train Function
        with tf.name_scope("Train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(loss)

        # self.scoreTensor = tf.reduce_sum(self.RewardValue, name = "RewardSum")
        # tf.summary.scalar("Score", self.scoreTensor)

    # 根據觀察的結果，選擇動作
    def chooseAction(self, observation):
        InputData = {self.observationsArray: [observation]}
        actionProb = self.session.run(self.ActionProb, feed_dict=InputData)

        # 因為確保動作是 Int(0 => 左，1 => 右)
        # 所以不給 size (所以只會回傳出一個 Int)
        # 如果給其他的 size 會回傳出一個陣列，連 1 也是個陣列 ex: [2]
        # 回傳回來的是一個 2維 Ex:[[0.3, 0.2, 0.5]]
        actionValue = np.random.choice(self.n_actions, p=actionProb[0])

        # Print 資料
        # print(format(actionProb) + " " + format(actionValue))
        return actionValue

    # 把 Action 跟 Reward 跟 State 儲存到記憶庫中
    def storeTransition(self, observation, action, reward):
        self.MemoryObservation.append(observation)
        self.MemoryAction.append(action)
        self.MemoryReward.append(reward)

    # 開始學習
    def learn(self):
        normalizedReward = self.MakeReward()

        self.session.run(
            self.optimizer,
            feed_dict={
                self.observationsArray: self.MemoryObservation,
                self.actionsNum: self.MemoryAction,
                self.DeltaValue: normalizedReward
            })

        # 每次學習要把前面的東西清空
        self.MemoryObservation = []
        self.MemoryAction = []
        self.MemoryReward = []

    """
    一般來說
    Policy Gradient 是沒有 Reward 的
    但 Tensorflow 的架構是需要有一個 Cost 的 Function
    然後要去 Minimize Cost
    所以必須產生一個假的 Reward
    是根據越接近開始的時候
    所做的動作就價值越高 (假設是立竿子的遊戲的話)
    """

    def MakeReward(self):
        # 先產生一連串的 0 陣列
        normalizeReward = np.zeros(len(self.MemoryReward))

        # reward，最一開始是0
        reward = 0

        # 在 Breakout 的遊戲中
        # 希望讓他打磚塊的前面幾個 reward 比較高
        # 其他的 reward 比較小
        # 目的是要讓他越打到越好
        # 要從 len(normalizeReward) - 1 到 0，但是range 不會包含到最後面，所以才要 -1
        for i in range(len(normalizeReward) - 1, -1, -1):
            reward = reward * self.RewardDecay + self.MemoryReward[i]
            normalizeReward[i] = reward

        # 這邊在做的事情是
        # 為了要讓 reward 平均
        # 不會太大或太小 (因為 array 可能會很大，所以要讓後面的東西變小)
        # 及 拉近差距 (/ dev)
        normalizeReward -= np.mean(normalizeReward)

        # 因為立竿子其他時間都是垃圾
        # normalizeReward -= 0.01

        # 讓資料間的距離縮短
        std = np.std(normalizeReward)
        if (std != 0):
            normalizeReward /= std
        return normalizeReward

    """
    Call 這條代表要儲存權重
    等下次近來在重新讀取
    """

    def SaveModel(self, globalStep=None):
        # 建立存擋器
        saver = tf.train.Saver()

        # 如果路徑不存在，就創建路徑
        if (not os.path.exists("./models/")):
            os.mkdir("./models/")

        saver.save(self.session, "./models" + self.logName, global_step=globalStep)

    """
    Call 這條代表要回復權重
    拿最新的權重
    """

    def RestoreModel(self):
        ckpt = tf.train.get_checkpoint_state("./models")
        if (ckpt != None):
            saver = tf.train.Saver()
            saver.restore(self.session, ckpt.model_checkpoint_path)