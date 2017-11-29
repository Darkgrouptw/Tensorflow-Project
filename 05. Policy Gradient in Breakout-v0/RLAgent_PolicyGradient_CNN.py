import numpy as np
import tensorflow as tf

np.random.seed(3)
tf.set_random_seed(3)

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            imgRows,
            imgCols,
            imgNumber,
            imgChannel,
            LearningRate = 0.01,
            RewardDecay = 0.95,                         # 衰減比例
            IsOutputGraph = False
    ):
        self.n_actions = n_actions

        # 圖片相關
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.imgNumber = imgNumber
        self.imgChannel = imgChannel

        self.LearningRate = LearningRate
        self.RewardDecay = RewardDecay

        self.BuildNet()
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # 將 Memory Set 清空
        self.MemoryObservation = []
        self.MemoryAction = []
        self.MemoryReward = []

        # 將 Tensorboard Graph 輸出出去
        if IsOutputGraph:
            tf.summary.FileWriter("./logs", self.session.graph)

    # Observation 跟 State 是代表同樣的事情，觀察到的東西就是 state，但是 Policy Gradient 裡面，沒有 State 的概念，是以機率來說
    # 建造 Neural Network
    def BuildNet(self):
        # 參數設定
        # 用的 filter 數目及大小
        filter1_Count = 16
        filter1_Size = 5
        filter2_Count = 20
        filter2_Size = 5
        fc1_Count = 128
        fc2_Count = 64

        # Input
        with tf.name_scope("Input"):
            # 前 k 張圖片，用 channel 判斷
            self.observationsArray = tf.placeholder(tf.float32, [None, self.imgRows, self.imgCols, self.imgNumber * self.imgChannel], name = "observationsArray")
            
            self.actionsNum = tf.placeholder(tf.int32, [None], name = "Action")                  # 注意 [None] 跟 [None,] 是一樣的，一維的陣列，裡面裝著不知道大小的 scalar，跟 [None, 1] 不一樣，這個是 二維陣列
            self.DeltaValue = tf.placeholder(tf.float32, [None], name="ActionValue")

        # 亂數
        weightInit = tf.random_normal_initializer(mean=0, stddev = 0.3)
        biasInit = tf.random_normal_initializer(mean=0, stddev = 0.1)
        
        # Conv2d 會做一件事，把 Input 攤平 [filter_height * filter_width * in_channels, output_channels].
        # 所以如果要處理多個 Channel 的資訊，例如三個顏色，可能需要用到 conv3d
        # Conv1
        conv1 = tf.layers.conv2d(
            inputs = self.observationsArray,
            filters = filter1_Count,
            kernel_size = filter1_Size,
            strides = 1,
            padding = "same",
            activation = tf.nn.relu,
            kernel_initializer = weightInit,
            bias_initializer = biasInit,
            name = "Conv1"
        )

        # Conv2
        conv2 = tf.layers.conv2d(
            inputs = conv1,
            filters = filter2_Count,
            kernel_size = filter2_Size,
            strides = 1,
            padding = "same",
            activation = tf.nn.relu,
            kernel_initializer = weightInit,
            bias_initializer = biasInit,
            name = "Conv2"
        )

        # 把資料壓平
        flatData = tf.reshape(conv2, [-1, 105 * 80 * 20])

        # Fully Connected 1
        fc1 = tf.layers.dense(
            inputs = flatData,
            units = fc1_Count,
            activation = tf.nn.relu,
            kernel_initializer = weightInit,
            bias_initializer = biasInit,
            name = "FC1"
        )

        fc2 = tf.layers.dense(
            inputs = fc1,
            units = fc2_Count,
            activation = tf.nn.relu,
            kernel_initializer = weightInit,
            bias_initializer = biasInit,
            name = "FC2"
        )

        output = tf.layers.dense(
            inputs = fc2,
            units = self.n_actions,
            kernel_initializer=weightInit,
            bias_initializer=biasInit,
            name = "output"
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

    # 根據觀察的結果，選擇動作
    def chooseAction(self, observation):
        InputData = {self.observationsArray: [observation]}
        actionProb = self.session.run(self.ActionProb, feed_dict= InputData)
        #print(actionProb)
        
        # 因為確保動作是 Int(0 => 左，1 => 右)
        # 所以不給 size (所以只會回傳出一個 Int)
        # 如果給其他的 size 會回傳出一個陣列，連 1 也是個陣列 ex: [2]
        # 回傳回來的是一個 2維 Ex:[[0.3, 0.2, 0.5]]
        actionValue = np.random.choice(self.n_actions, p=actionProb.ravel())
        return actionValue

    # 把 Action 跟 Reward 跟 State 儲存到記憶庫中
    def storeTransition(self, observation, action, reward):
        self.MemoryObservation.append(observation)
        self.MemoryAction.append(action)
        self.MemoryReward.append(reward)

    # 開始學習
    def learn(self):
        normalizedReward = self.MakeReward()
        print(normalizedReward)

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
        
        # 讓資料間的距離縮短
        std = np.std(normalizeReward)
        if(std != 0):
            normalizeReward /= std
        return normalizeReward

