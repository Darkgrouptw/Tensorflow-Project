import numpy as np
import tensorflow as tf
import os

class DeepQNetwork:
    def __init__(
            self,                                   # 每個 Function 好像都要
            imgRows,                                # 圖片高
            imgCols,                                # 圖片寬
            imgNumber,                              # 圖片數目
            imgChannel,                             # Channel Number
            n_actions,                              # Action 數目
            LearningRate = 0.01,                   # Learning Rate
            RewardDecay = 0.9,                      # 獎勵的遞減比例
            e_greedy = 0.9,                         # 90% 的時間，都是使用 Q 最大的動作
            replace_target_iter = 300,              # 多少步之後，要把 Evaluate Net 的參數，取代 Target Net 的參數
            memory_size = 500,                      # Memory Size 有多少就可以記下多少的步驟
            batch_size = 32,                        # Batch Size 的大小
            e_greedy_increment = None,              # 這個不太清楚
            IsOutputGraph = False                   # 是否要輸出 Graph 到 Tensorboard
    ):
        # 輸入相關
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.imgNumber = imgNumber
        self.imgChannel = imgChannel

        # 動作相關
        self.n_actions = n_actions
        self.LearningRate = LearningRate
        self.gamma = RewardDecay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Learning Counter
        self.learn_step_counter = 0

        # Memory 庫
        self.MemoryOb = []
        self.MemoryNextOb = []
        self.MemoryAction = []
        self.MemoryReward = []

        # Build Net
        self.BuildNet()

        # 要覆蓋過去的 operation
        EvalParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="EvalNet")
        TargetParams = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="TargetNet")
        self.replaceOP = [tf.assign(t, e) for t, e in zip(TargetParams, EvalParams)]

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # 輸出檔案相關
        self.logName = "/Deep Q Network"

        if IsOutputGraph:
            log = tf.summary.FileWriter("./logs" + self.logName)
            log.add_graph(self.session.graph)

        # Memory 目前存到第幾個 Index
        self.memoryIndex = 0

    # 建造 Net
    def BuildNet(self):
        # 因為是 conv3d，所以是 none, depth, rows, cols, channel
        self.observation = tf.placeholder(
            tf.float32,
            [None, self.imgNumber, self.imgRows, self.imgCols, self.imgChannel],
            name="State"
        )
        self.q_target = tf.placeholder(
            tf.float32,
            [None, self.n_actions],
            name="Q_Target"
        )
        self.nextObservation = tf.placeholder(
            tf.float32,
            [None, self.imgNumber, self.imgRows, self.imgCols, self.imgChannel],
            name="TargetState")

        # 參數設定
        # 用的 filter 數目及大小
        filter1_Size = 5
        filter1_Count = 16
        filter2_Size = 5
        filter2_Count = 20

        fc1_units = 1024
        fc2_units = 256
        fc3_units = 64

        weightInit = tf.random_normal_initializer(mean=0, stddev=0.3)
        biasInit = tf.random_normal_initializer(mean=0, stddev=0.1)

        #############################################################
        # Evaluate Net
        # 用來預估的網路
        # 會即時更新 Q 值
        #############################################################
        with tf.variable_scope("EvalNet"):
            with tf.name_scope("Layer1"):
                evalConv1 = tf.layers.conv3d(
                    inputs=self.observation,
                    filters=filter1_Count,
                    kernel_size=filter1_Size,
                    strides=1,
                    padding="SAME",
                    activation=tf.nn.relu,
                    kernel_initializer=weightInit,
                    bias_initializer=biasInit,
                    name="Eval_Conv1"
                )
                evalMaxpooling1 = tf.layers.max_pooling3d(
                    inputs=evalConv1,
                    pool_size=3,
                    strides=3,
                    padding="SAME",
                    name="Eval_MaxPooling1"
                )
            with tf.name_scope("Layer2"):
                evalConv2 = tf.layers.conv3d(
                    inputs=evalMaxpooling1,
                    filters=filter2_Count,
                    kernel_size=filter2_Size,
                    strides=1,
                    padding="SAME",
                    activation=tf.nn.relu,
                    kernel_initializer=weightInit,
                    bias_initializer=biasInit,
                    name="Eval_Conv2"
                )
                evalMaxpooling2 = tf.layers.max_pooling3d(
                    inputs=evalConv2,
                    pool_size=3,
                    strides=3,
                    padding="SAME",
                    name="Eval_MaxPooling2"
                )
            evalflatData = tf.layers.flatten(
                evalMaxpooling2,
                name="Eval_Flat"
            )

            evalfc1 = tf.layers.dense(
                inputs=evalflatData,
                units=fc1_units,
                activation=tf.nn.relu,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Eval_FC1"
            )

            evalfc2 = tf.layers.dense(
                inputs=evalfc1,
                units=fc2_units,
                activation=tf.nn.relu,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Eval_FC2"
            )

            evalfc3 = tf.layers.dense(
                inputs=evalfc2,
                units=fc3_units,
                activation=tf.nn.relu,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Eval_FC3"
            )

            self.q_eval = tf.layers.dense(
                inputs=evalfc3,
                units=self.n_actions,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Q_Eval"
            )

        with tf.name_scope("Loss"):
            # 要依照現實預測的 QTarget，及 預測出來的差距做 loss function
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.name_scope("Train"):
            self.optimizer = tf.train.RMSPropOptimizer(self.LearningRate).minimize(self.loss)

        #############################################################
        # Target Net
        # 用來玩遊戲的網路
        # 此網路根據前幾次的 Eval Net 的執來預測
        # 並出差距
        #############################################################
        with tf.variable_scope("TargetNet"):
            with tf.name_scope("Layer1"):
                targetConv1 = tf.layers.conv3d(
                    inputs=self.nextObservation,
                    filters=filter1_Count,
                    kernel_size=filter1_Size,
                    strides=1,
                    padding="SAME",
                    activation=tf.nn.relu,
                    kernel_initializer=weightInit,
                    bias_initializer=biasInit,
                    name="Target_Conv1"
                )
                targetMaxpooling1 = tf.layers.max_pooling3d(
                    inputs=targetConv1,
                    pool_size=3,
                    strides=3,
                    padding="SAME",
                    name="Target_MaxPooling1"
                )
            with tf.name_scope("Layer2"):
                targetConv2 = tf.layers.conv3d(
                    inputs=targetMaxpooling1,
                    filters=filter2_Count,
                    kernel_size=filter2_Size,
                    strides=1,
                    padding="SAME",
                    activation=tf.nn.relu,
                    kernel_initializer=weightInit,
                    bias_initializer=biasInit,
                    name="Target_Conv2"
                )
                targetMaxpooling2 = tf.layers.max_pooling3d(
                    inputs=targetConv2,
                    pool_size=3,
                    strides=3,
                    padding="SAME",
                    name="Target_MaxPooling2"
                )
            targetflatData = tf.layers.flatten(
                targetMaxpooling2,
                name="Target_Flat")

            targetfc1 = tf.layers.dense(
                inputs=targetflatData,
                units=fc1_units,
                activation=tf.nn.relu,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Target_FC1"
            )
            targetfc2 = tf.layers.dense(
                inputs=targetfc1,
                units=fc2_units,
                activation=tf.nn.relu,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Target_FC2"
            )
            targetfc3 = tf.layers.dense(
                inputs=targetfc2,
                units=fc3_units,
                activation=tf.nn.relu,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Target_FC3"
            )

            self.q_next = tf.layers.dense(
                inputs=targetfc3,
                units=self.n_actions,
                kernel_initializer=weightInit,
                bias_initializer=biasInit,
                name="Q_Next"
            )

    # 要把觀察，拿進來做處理
    def storeTransition(self, observation, action, reward, nextObservation):
        self.MemoryOb.append(observation)
        self.MemoryAction.append(action)
        self.MemoryReward.append(reward)
        self.MemoryNextOb.append(nextObservation)

        # 超過大小了
        if(len(self.MemoryReward) > self.memory_size):
            self.MemoryOb.pop(0)
            self.MemoryAction.pop(0)
            self.MemoryReward.pop(0)
            self.MemoryNextOb.pop(0)

    # 選擇 Action
    def chooseAction(self, observation, IsTraining = True):
        if(IsTraining):
            # Uniform 是一個 0 ~ 1 的數字
            if np.random.uniform() < self.epsilon:
                action_value = self.session.run(self.q_eval, feed_dict={self.observation: [observation]})
                # print(action_value)

                # 取最大的出來
                action = np.argmax(action_value)
            else:
                action = np.random.randint(0, self.n_actions)
        else:
            action_value = self.session.run(self.q_eval, feed_dict={self.observation: [observation]})
            # print(action_value)

            # 取最大的出來
            action = np.argmax(action_value)
        return action

    # learn 的 Function
    def learn(self):
        # 如果到了要 Replace 的次數，就 Replace 掉
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.session.run(self.replaceOP)
            # print("\nTarget Params Change\n")

        # 從過去的記憶中，取一個 Batch Size(如果資料小於 Batch Size 的話，可以重複取吧?)
        sample_index = np.random.choice(len(self.MemoryReward), size=self.batch_size)

        batchOb = []
        batchNextOb = []
        batchAction = []
        batchReward = []

        for i in range(0, self.batch_size):
            batchOb.append(self.MemoryOb[sample_index[i]])
            batchNextOb.append(self.MemoryNextOb[sample_index[i]])
            batchAction.append(self.MemoryAction[sample_index[i]])
            batchReward.append(self.MemoryReward[sample_index[i]])

        q_next, q_eval = self.session.run([self.q_next, self.q_eval], feed_dict={
            self.observation: batchOb,
            self.nextObservation: batchNextOb
        })

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batchAction
        reward = batchReward

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 跑優化
        self.session.run(self.optimizer, feed_dict={
            self.observation: batchNextOb,
            self.q_target: q_target
        })


        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1


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