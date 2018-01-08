import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

class DeepQNetwork:
    def __init__(
            self,                                   # 每個 Function 好像都要
            n_actions,                              # Action 數目
            n_features,                             # Features 數目
            learning_rate = 0.01,                   # Learning Rate
            reward_decay = 0.9,                     # 獎勵的遞減比例
            e_greedy = 0.9,                         # 90% 的時間，都是使用 Q 最大的動作
            replace_target_iter = 300,              # 多少步之後，要把 Evaluate Net 的參數，取代 Target Net 的參數
            memory_size = 500,                      # Memory Size 有多少就可以記下多少的步驟
            batch_size = 32,                        # Batch Size 的大小
            e_greedy_increment = None,              # 這個不太清楚
            IsOutputGraph = False                   # 是否要輸出 Graph 到 Tensorboard
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.learning_rate = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # Learning Counter
        self.learn_step_counter = 0

        # 創建 [State (size => n_features),
        #       Action (size => 1 (0 1 2 3)),
        #       Reward (size => 1),
        #       Next State (Size => n_features)]
        self.memory = np.zeros([memory_size, n_features * 2 + 2])

        # Build Net
        self.BuildNet()

        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        if IsOutputGraph:
            log = tf.summary.FileWriter("./logs")
            log.add_graph(self.session.graph)

        # 在最後面 plot 圖的時候，會用到
        self.cost_histogram = []

        # Memory 目前存到第幾個 Index
        self.memoryIndex = 0

    # 建造 Net
    def BuildNet(self):
        self.state = tf.placeholder(tf.float32, [None, self.n_features], name="State")
        self.q_target = tf.placeholder(tf.float32,[None, self.n_actions], name="Q_Target")
        
        # Hidden Layer
        num_hidden_unit = 20

        #############################################################
        # Evaluate Net
        #############################################################
        with tf.variable_scope("EvalNet"):
            collectionName = ["EvalNet_params", tf.GraphKeys.GLOBAL_VARIABLES]

            with tf.variable_scope("Layer1"):
                layer1_weight = tf.Variable(
                    tf.random_normal([self.n_features, num_hidden_unit]),
                    collections=collectionName,
                    name="layer1_weight"
                )
                layer1_bias = tf.Variable(
                    tf.random_normal([num_hidden_unit]),
                    collections=collectionName,
                    name="layer1_bias"
                )

                # 經過 Relu，將大的越大，沒用的消除為0
                layer1 = tf.nn.relu(tf.matmul(self.state, layer1_weight) + layer1_bias)
                
            with tf.variable_scope("Layer2"):
                layer2_weight = tf.Variable(
                    tf.random_normal([num_hidden_unit, self.n_actions]),
                    collections=collectionName,
                    name="layer2_weight"
                )
                layer2_bias = tf.Variable(
                    tf.random_normal([self.n_actions]),
                    collections=collectionName,
                    name="layer2_bias"
                )
                
                self.q_eval = tf.matmul(layer1, layer2_weight) + layer2_bias
                
        with tf.name_scope("Loss"):
            # 要依照現實預測的 QTarget，及 預測出來的差距做 loss function
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.name_scope("Train"):
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

        #############################################################
        # Target Net
        #############################################################
        self.nextState = tf.placeholder(tf.float32, [None, self.n_features], name="TargetState")

        with tf.variable_scope("TargetNet"):
            #############################################################
            # Evaluate Net
            #############################################################
            collectionName = ["TargetNet_params", tf.GraphKeys.GLOBAL_VARIABLES]
            

            with tf.variable_scope("Layer1"):
                layer1_weight = tf.Variable(
                    tf.random_normal([self.n_features, num_hidden_unit]),
                    collections=collectionName,
                    name="layer1_weight"
                )
                layer1_bias = tf.Variable(
                    tf.random_normal([num_hidden_unit]),
                    collections=collectionName,
                    name="layer1_bias"
                )

                # 經過 Relu，將大的越大，沒用的消除為0
                layer1 = tf.nn.relu(tf.matmul(self.state, layer1_weight) + layer1_bias)

            with tf.variable_scope("Layer2"):
                layer2_weight = tf.Variable(
                    tf.random_normal([num_hidden_unit, self.n_actions]),
                    collections=collectionName,
                    name="layer2_weight"
                )
                layer2_bias = tf.Variable(
                    tf.random_normal([self.n_actions]),
                    collections=collectionName,
                    name="layer2_bias"
                )

                self.q_next = tf.matmul(layer1, layer2_weight) + layer2_bias

    # 要把觀察，拿進來做處理
    def store_transition(self, state, action, reward, nextState):
        transition = np.hstack((state, [action, reward], nextState))

        # 現在的 Index
        index = self.memoryIndex % self.memory_size
        self.memory[index, :] = transition

        self.memoryIndex += 1

    # 選擇 Action
    def choose_action(self, state):
        # 要把一維的結果，變成二維的結果輸出
        state = state[np.newaxis, :]

        # Uniform 是一個 0 ~ 1 的數字
        if np.random.uniform() < self.epsilon:
            action_vale = self.session.run(self.q_eval, feed_dict={self.state: state})

            # 取最大的出來
            action = np.argmax(action_vale)
        else:
            action = np.random.randint(0,self.n_actions)
        return action

    # 每過一段時間，要把 Evaluate Network 值，複製到 Target Network 上面
    def _replaceTargetParams(self):
        TargetParams = tf.get_collection("TargetNet_params")
        EvalutateParams = tf.get_collection("EvalNet_params")

        self.session.run([tf.assign(t, e) for t, e in zip(TargetParams, EvalutateParams)])

    # learn 的 Function
    def learn(self):
        # 如果到了要 Replace 的次數，就 Replace 掉
        if self.learn_step_counter % self.replace_target_iter == 0:
            self._replaceTargetParams()
            print("\nTarget Params Change\n")

        # 從過去的記憶中，取一個 Batch Size(如果資料小於 Batch Size 的話，可以重複取吧?)
        if self.memoryIndex > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memoryIndex, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        q_next, q_eval = self.session.run([self.q_next, self.q_eval], feed_dict={
            self.nextState: batch_memory[:, -self.n_features:],         # 負代表後四個
            self.state: batch_memory[:, :self.n_features]               # 前四個
        })

        """
        For example in this batch I have 2 samples and 3 actions:
        q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        q_target = q_eval =
        [[1, 2, 3],
         [4, 5, 6]]
        Then change q_target with the real q_target value w.r.t the q_eval's action.
        For example in:
            sample 0, I took action 0, and the max q_target value is -1;
            sample 1, I took action 2, and the max q_target value is -2:
        q_target =
        [[-1, 2, 3],
         [4, 5, -2]]
        So the (q_target - q_eval) becomes:
        [[(-1)-(1), 0, 0],
         [0, 0, (-2)-(6)]]
        We then backpropagate this error w.r.t the corresponding action to network,
        leave other action as error=0 cause we didn't choose it.
        """
        q_target = q_eval.copy()
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        q_target[batch_index, eval_act_index] = reward + self.gamma * np.max(q_next, axis=1)

        # 跑優化
        self.session.run(self.optimizer, feed_dict={
            self.state: batch_memory[:, :self.n_features],
            self.q_target: q_target
        })
        cost = self.session.run(self.loss, feed_dict={
            self.state: batch_memory[:, :self.n_features],
            self.q_target: q_target
        })
        self.cost_histogram.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_histogram)), self.cost_histogram)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()