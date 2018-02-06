import tensorflow as tf
import numpy as np
import os
from .Params import ParamsConfig

class DeepQNetworkAgent:
    def __init__(
            self,
            imgRows,                                    # 圖片高
            imgCols,                                    # 圖片寬
            numAction,                                  # 總共有幾個動作
            imgNumber,                                  # 圖片個數
            imgChannel = 1,                             # 圖片的 Channel 數目

            config = ParamsConfig,                      # 設定的參數
            IsOutputGraph = False
    ):
        # 輸出圖片相關
        self.imgRows = imgRows
        self.imgCols = imgCols
        self.imgNumber = imgNumber
        self.imgChannel = imgChannel

        # 動作空間
        self.numAction = numAction

        # 參數
        self.config = config
        self.epsilon = 0

        # 暫存位置
        self.logName = "/Deep Q Network"
        self.logDir = "./logs" + self.logName
        self.modelCheckPointDir = "./models" + self.logName

        # 建造網路架構
        BuildNet()

        # 建立 Log 相關 Function
        BuildLog()

        # 建造 Session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

        # 計算 Step 的
        with tf.name_scope("StepCounter"):
            self.StepCounter = tf.Variable(initial_value = 0, trainable = False, name = "Step")
            self.AddStep_op = tf.assign(self.StepCounter, self.StepCounter + 1, name = "AddStep")
            self.loggingStep = tf.summary.scalar("Step", self.StepCounter)

        if IsOutputGraph:
            self.LogFile.add_graph(self.session.graph)

    ########################################################################
    # 建造網路架構
    ########################################################################
    def BuildNet(self):
        with tf.name_scope("Input"):
            # 這個狀態的
            self.Observations = tf.placeholder(dtype=tf.float32, [None, self.imgNumber * self.imgChannel, self.imgRows, self.imgCols], name="Observations"

            # 下個狀態
            self.NextObservations = tf.placeholder(dtype=tf.float32, [None, self.imgNumber * self.imgChannel, self.imgRows, self.imgCols], name="NextObservations")

            # 經過 r + gamma * argmax(Q_Next, axis = 1)
            self.q_target = tf.placeholder(dtype = tf.float32, [None, self.numAction], name = "Q_Target")

        weightInit = tf.random_normal_initializer(mean=0, stddev=0.02)

        #############################################################
        # Evaluate Net
        # 用來預估的網路
        # 會即時更新 Q 值
        #############################################################
        with tf.variable_scope("EvalNet"):
            EvalLayer1 = tf.layers.dense(
                inputs = self.Observations,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = weightInit,
                name = "Layer1"
            )

            self.Q_Eval = tf.layers.dense(
                inputs = EvalLayer1,
                units = self.numAction,
                name = "Q_Eval"
            )

        #############################################################
        # Target Net
        # 用來預估下一個狀態 Q 值的網路
        # 這個通常會過一段時間再更新 (為了避免每次更新影響到它的結果)
        # 原本的 Q 值 = 這個 state 的 reward + gramma * argmax(下一個 state 的 Q 值)
        #############################################################
        with tf.variable_scope("TargetNet"):
            TargetLayer1 = tf.layers.dense(
                inputs = self.NextObservations,
                units = 20,
                activation = tf.nn.relu,
                kernel_initializer = weightInit,
                name = "Layer1"
            )
            self.Q_Next = tf.layers.dense(
                inputs = TargetLayer1,
                units = self.numAction,
                name = "Q_Next"
            )

            with tf.name_scope("Loss"):
                squareDifferent = tf.squared_difference(self.Q_Eval, self.q_target)
                self.loss = tf.reduce_sum(tf.reduce_mean(squareDifferent, axis=1))
                self.loggingLoss = tf.summary.scalar("Loss", self.loss)

            with tf.name_scope("Train"):
                learning_rate_op = tf.maximum(
                    config.LearningMinimum,
                    tf.train.exponential_decay(
                        config.LearningRate,
                    )
                )
                self.optimizer = tf.train.RMSPropOptimizer(
                    config.LearningRate,
                    momentum = config.LearningMomentum,
                    epsilon = config.LearningEpsilon
                ).minimize(self.loss)


    ########################################################################
    # 建造 Log Function
    ########################################################################
    def BuildLog(self):
        # Log 部分
        self.LogFile = tf.summary.FileWriter(self.logDir)

        with tf.name_scope("Log"):
            # Log 這一個 Epoch 的 Score
            self.thisEpochScoreListTensor = tf.placeholder(tf.int32, [None], name="ThisEpochScore")
            thisEpochScore = tf.reduce_sum(self.thisEpochScoreListTensor)
            loggingScore = tf.summary.scalar("Score", thisEpochScore)

            # Log 這一個 Epoch 的 Max Q Value
            self.QValueListTensor = tf.placeholder(tf.float32, [None, self.numAction], name="MaxQValueTensor")
            MaxQValueTensor = tf.reduce_mean(tf.reduce_max(self.QValueListTensor, axis=1))
            MinQValueTensor = tf.reduce_mean(tf.reduce_min(self.QValueListTensor, axis=1))
            self.loggingAverageMaxQValue = tf.summary.scalar("Average_MaxQValue", MaxQValueTensor)
            self.loggingAverageMinQValue = tf.summary.scalar("Average_MinQValue", MinQValueTensor)

        self.Log = tf.summary.merge([loggingScore, self.loggingLoss, self.loggingAverageMaxQValue, self.loggingAverageMinQValue], name="Log")


    ########################################################################
    # 選擇動作
    ########################################################################
    def chooseAction(self, observations, IsTrainning = True):
        if IsTrainning:
            if np.random.uniform() > self.epsilon:
                # 亂數選一個值
                return np.random.randint(0, self.numAction)
            else:
                QValue = self.session.run(self.q_eval, feed_dict={self.Observations: [observations]})
                QValueMaxIndex = np.argmax(QValue[0])
                return QValueMaxIndex
        else:
            QValue = self.session.run(self.q_eval, feed_dict={self.Observations: [observations]})
            QValueMaxIndex = np.argmax(QValue[0])
            return QValueMaxIndex


    ########################################################################
    # 將模型存出去
    ########################################################################
    def SaveModel(
            self,
            step = None
    ):
        # 如果檔案路徑不存在就創建一個
        if not os.path.exists(self.modelCheckPointDir):
            os.mkdir(self.modelCheckPointDir)

        # 存 Model 出去
        saver = tf.train.Saver()
        saver.save(self.session, self.modelCheckPointDir, global_step=step)


    ########################################################################
    # 將模型回復
    ########################################################################
    def RestoreModel(
            self
    ):
        ckpt = tf.train.get_checkpoint_state(self.modelCheckPointDir)
        if (ckpt != None):
            saver = tf.train.Saver();
            saver.restore(self.session, ckpt.model_checkpotint_path)