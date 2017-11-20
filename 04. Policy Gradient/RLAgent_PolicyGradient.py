import numpy as np
import tensorflow as tf

class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            LearningRate = 0.01,
            RewardDecay = 0.95,                         # 衰減比例
            IsOutputGraph = False
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.LearningRate = LearningRate
        self.RewardDecay = RewardDecay

        BuildNet()
        self.session = tf.Session()

        # 將 Memory Set 清空
        self.MemoryObservation = []
        self.MemoryAction = []
        self.MemoryReward = []

        # 將 Tensorboard Graph 輸出出去
        if IsOutputGraph:
            tf.summary.FileWriter(self.session.graph)

    # Observation 跟 State 是代表同樣的事情，觀察到的東西就是 state，但是 Policy Gradient 裡面，沒有 State 的概念，是以機率來說
    # 建造 Neural Network
    def BuildNet(self):
        hiddenUnits = 10

        # Input
        with tf.name_scope("Input"):
            self.observations = tf.placeholder(tf.float32, [None, self.n_features], name = "observations")
            self.actionsNum = tf.placeholder(tf.int32, [None], name = "Action")                  # 注意 [None] 跟 [None,] 是一樣的，一維的陣列，裡面裝著不知道大小的 scalar，跟 [None, 1] 不一樣，這個是 二維陣列
            self.DeltaValue = tf.placeholder(tf.float32, [None], name="ActionValue")

        # 亂數
        weightInit = tf.random_normal_initializer(mean=0, stddev = 0.3)
        biasInit = tf.random_normal_initializer(mean=0, stddev = 0.1)

        # Fully Connected 1
        layer1 = tf.layers.dense(
            inputs = self.observations,
            units = hiddenUnits,
            activation = tf.nn.tanh,
            kernel_initializer=weightInit,
            bias_initializer=biasInit,
            name="layer1"
        )
        # Fully Connected 2
        layer2 = tf.layers.dense(
            inputs = layer1,
            units = self.n_actions,
            kernel_initializer=weightInit,
            bias_initializer=biasInit,
            name="Layer2"
        )

        # 每個動作的機率
        ActionProb = tf.nn.softmax(layer2, name="ActionProb")

        # Loss Function
        with tf.name_scope("Loss"):
            actionNum_OneHot = tf.one_hot(self.actionsNum, self.n_actions)
            cost = tf.losses.softmax_cross_entropy(logits=ActionProb, labels=actionNum_OneHot)

            # 這裡的意思是，只有剛開始不會倒的情況下， DeltaValue 就會越大，代表他這部分要越重視!!
            loss = cost * self.DeltaValue

        # Train Function
        with tf.name_scope("Train"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.LearningRate).minimize(loss)

    # 根據觀察的結果，選擇動作
    # def chooseAction(self, observation):

    # 把 Action 跟 Reward 跟 State 儲存到記憶庫中
    def storeTransition(self, observation, action, reward):
        self.MemoryObservation.append(observation)
        self.MemoryAction.append(action)
        self.MemoryReward.append(reward)

    # 開始學習
    # def learn(self):

    """
        一般來說
        Policy Gradient 是沒有 Reward 的
        但 Tensorflow 的架構是需要有一個 Cost 的 Function
        然後要去 Minimize Cost
        所以必須產生一個假的 Reward
        是根據越接近開始的時候
        所做的動作就價值越高 (假設是立竿子的遊戲的話)
        """
    # def MakeReward(self):


