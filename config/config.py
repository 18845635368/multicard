import ml_collections


def config_test():
    config = ml_collections.ConfigDict()
    # !随机参数配置
    config.seed = 1

    # !硬件配置
    # 使用的GPU数目
    config.gpus = 2
    config.world_size = 2
    # 使用的进程平台
    config.backend = 'nccl'
    #
    config.init_method = 'tcp://10.249.178.201:34567'
    # !transform配置
    config.face = 'scale'
    config.size = 224
    # !训练参数配置
    config.batch = 64
    config.epochs = 30
    config.syncbn = True
    # 空域分支选用
    config.net = 'EfficientNetB4'
    # 训练集与验证集切分标准
    config.traindb = ["ff-c23-720-140-140"]
    config.valdb = ["ff-c23-720-140-140"]
    # !数据集配置
    # 切割脸部照片的存放目录
    config.ffpp_faces_df_path = '/mnt/8T/FFPP/df/output/FFPP_df.pkl'
    # 切割脸部的Dataframe存放地点
    config.ffpp_faces_dir = '/mnt/8T/FFPP/faces/output'
    # 多久验证一次模型，单位（batch）
    config.valint = 100
    config.valsamples = 6000
    # 多久记录一次log，单位（batch）
    config.logint = 100
    # 多久保存一次模型，
    config.modelperiod = 500
    # !优化器配置
    config.lr = 1e-3
    config.patience = 10
    # !模型加载配置
    config.scratch = False
    config.models_dir = '/mnt/8T/multicard/weights/binclass/'
    # 1会加载最优模型，2会加载最新的模型，3会加载制定的模型
    config.mode = 1
    config.index = 0
    config.workers = 4

    # !logpath
    config.log_dir = '/mnt/8T/multicard/runs/binclass/'

    # !暂时无用配置
    config.debug = False
    config.dfdc_faces_df_path = ''
    config.dfdc_faces_dir = ''

    return config
