import tensorflow as tf
from .Params import ParamsConfig
import os

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

        # 暫存位置
        self.logDir = "./logs"
        self.modelCheckPointDir = "./models/DeepQNetwork"

        # 建造網路架構
        BuildNet()

        # 建造 Session
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())

    ########################################################################
    # 建造網路架構
    ########################################################################
    def BuildNet(self):

    ########################################################################
    # 將模型存出去
    ########################################################################
    # def SaveModel(
    #         self,
    #         step = None
    # ):
    #     # 如果檔案路徑不存在就創建一個
    #     if not os.path.exists(self.modelCheckPointDir):
    #         os.mkdir(self.modelCheckPointDir)
    #
    #     # 存 Model 出去
    #     saver = tf.train.Saver()
    #     saver.save(self.session, self.modelCheckPointDir, global_step=step)
    #
    # ########################################################################
    # # 將模型回復
    # ########################################################################
    # def RestoreModel(
    #         self
    # ):
    #     ckpt = tf.train.get_checkpoint_state(self.modelCheckPointDir)
    #     if (ckpt != None):
    #         saver = tf.train.Saver();
    #         saver.restore(self.session, ckpt.model_checkpotint_path)