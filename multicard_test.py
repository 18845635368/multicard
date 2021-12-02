'''
deepfake detection multicard 

@author hzm
@date 2021.12.1
'''

import os

from datetime import datetime
import argparse

from isplutils.data import FrameFaceIterableDataset, load_face, FrameFaceDataset
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.distributed as dist
# from apex.parallel import DistributedDataParallel as DDP
# from apex import amp

from architectures import fornet
from isplutils import utils, split
from tqdm import tqdm
import shutil
from tensorboardX import SummaryWriter
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


###################
# 分布式训练
###################

# ! 流程标识
# * 方法具体执行解释
# ? 测试代码注释

'''
:step 1     进入分布式训练，不同的进程先展示自己所属的id=对应所用的GPU
:step 2     进行初始参数配置  
:step 3     设置优化器和学习率调节器
:step 4     模型加载
:step 5     加载图片载入要用的transform
:step 6     数据集处理
:step 7     初始化tensorboard
:step 8     开始训练
'''
# Todo
'''
1.训练的模型再次载入报错                                           fix： 2021.12.2  15:05
2.再次载入后，在优化器的step阶段，变量不在一个tensor上，有些在cpu上   fix：2021.12.2  15:05
3.config类的编写

'''


def dist_train(gpu, args):
    # !step 1
    rank = gpu  # 当前进程号
    print('Rank id: ', rank)

    # !step 2
    # *将args获取的参数转化为变量
    train_datasets = args.traindb
    val_datasets = args.valdb
    dfdc_df_path = args.dfdc_faces_df_path
    ffpp_df_path = args.ffpp_faces_df_path
    dfdc_faces_dir = args.dfdc_faces_dir
    ffpp_faces_dir = args.ffpp_faces_dir
    face_policy = args.face
    face_size = args.size
    batch_size = args.batch
    initial_lr = args.lr
    validation_interval = args.valint
    patience = args.patience
    initial_model = args.init
    train_from_scratch = args.scratch
    # ?max_train_samples = args.trainsamples
    max_val_samples = args.valsamples
    log_interval = args.logint
    num_workers = args.workers
    seed = args.seed
    debug = args.debug
    suffix = args.suffix
    # ?enable_attention = args.attention
    weights_folder = args.models_dir
    logs_folder = args.log_dir
    world_size = args.world_size
    backend = args.backend
    init_method = args.init_method
    epoch_run = args.epochs
    model_period = args.modelperiod

    # *初始化进程组，决定进程的通信方式，自己的进程标志
    dist.init_process_group(
        backend=backend, init_method=init_method, world_size=world_size, rank=rank)
    torch.manual_seed(0)
    # ?model = ConvNet()

    # *获取nodel_class,生成model
    model_class = getattr(fornet, args.net)
    model = model_class()
    transformer = utils.get_transformer(face_policy=face_policy, patch_size=face_size,
                                        net_normalizer=model.get_normalizer(), train=True)

    # *生成model对应的tag
    tag = utils.make_train_tag(net_class=model_class,
                               traindb=train_datasets,
                               face_policy=face_policy,
                               patch_size=face_size,
                               seed=seed,
                               debug=debug,
                               note=''
                               )

    # *生成saved model的路径还有,生成文件夹
    bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, tag, 'last.pth')
    periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')
    os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

    # !step3
    # *选择BCEWithLogits作为损失函数，并将其部署到GPU上
    criterion = nn.BCEWithLogitsLoss().cuda(gpu)
    optimizer = torch.optim.Adam(
        model.get_trainable_parameters(), lr=initial_lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode='min',
        factor=0.1,
        patience=patience,
        cooldown=2 * patience,
        min_lr=initial_lr*1e-5,
    )

    # !step4
    # *模型超参配置
    val_loss = min_val_loss = 10
    epoch = iteration = 0
    model_state = None
    opt_state = None

    # *对模型进行加载

    # *选定某个model作为特定的其实训练model
    if initial_model is not None:

        print('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        model_state = state['model']
    # *接着上次的训练继续训练
    elif not args.scratch and os.path.exists(last_path):
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        model_state = state['model']
        opt_state = state['opt']
        iteration = state['iteration'] + 1
        epoch = state['epoch']
    # *接着最棒的model进行训练
    if not args.scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']

    # *开始将参数载入model
    if model_state is not None:
        incomp_keys = model.load_state_dict(model_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = args.lr
        optimizer.load_state_dict(opt_state)
    print(epoch)

    model = model.cuda(gpu)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[gpu])

    # 将所有optimizer的数据放回到cuda上
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda(gpu)

    # !step5
    # *生成数据增强用到的transform

    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # *同步batch归一化
    if args.syncbn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if gpu == 0:
            print('Use SyncBN in training')
    torch.cuda.set_device(gpu)

    # !step6

    # *加载数据
    print("Loading data")

    # *从总dfs中，提取所需的dfs
    splits = split.make_splits(dfdc_df=dfdc_df_path, ffpp_df=ffpp_df_path, dfdc_dir=dfdc_faces_dir, ffpp_dir=ffpp_faces_dir,
                               dbs={'train': train_datasets, 'val': val_datasets})
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]

    # *验证数据集是否提供
    for dataset in args.traindb:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError(
                'Specify DataFrame and directory for DFDC faces for training!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError(
                'Specify DataFrame and directory for FF++ faces for training!')
    for dataset in val_datasets:
        if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
            raise RuntimeError(
                'Specify DataFrame and directory for DFDC faces for validation!')
        elif dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError(
                'Specify DataFrame and directory for FF++ faces for validation!')
    # *制作iterable数据集，DDP不能用iterable的数据集，数据集应实现方法__getitem__
    train_dataset = FrameFaceDataset(root=train_roots[0],
                                     df=train_dfs[0],
                                     scale=face_policy,
                                     transformer=transformer,
                                     size=face_size,
                                     )
    val_dataset = FrameFaceIterableDataset(roots=val_roots,
                                           dfs=val_dfs,
                                           scale=face_policy,
                                           num_samples=max_val_samples,
                                           transformer=transformer,
                                           size=face_size,
                                           )

    # 验证生成的数据集长度
    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    if len(val_dataset) == 0:
        print('No validation samples. Halt.')
        return

    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation samples: {}'.format(len(val_dataset)))

    # *将数据集提供给sampler
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset,
                                                                  num_replicas=args.world_size,
                                                                  rank=rank)

    # *生成Dataloader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               pin_memory=True,
                                               sampler=train_sampler)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, num_workers=num_workers, batch_size=batch_size, )

    # !step 7
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # *若是一开始训练，log已经存在，将其删掉
        shutil.rmtree(logdir, ignore_errors=True)
    tb = SummaryWriter(logdir=logdir)

    # !step 8
    while epoch != epoch_run:
        # ?optimizer.zero_grad()

        train_loss = train_num = 0
        train_pred_list = []
        train_labels_list = []
        for train_batch in tqdm(train_loader, desc='Epoch {:03d}'.format(epoch), leave=False,
                                total=len(train_loader)):
            model.train()
            batch_data, batch_labels = train_batch
            train_batch_num = len(batch_labels)
            # *param train_num 用于统计训练总数
            train_num += train_batch_num
            train_labels_list.append(batch_labels.numpy().flatten())

            train_batch_loss, train_batch_pred = batch_forward(
                model, criterion, batch_data, batch_labels)
            train_pred_list.append(train_batch_pred.flatten())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # *运用优化器
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # *记录训练阶段数据，保存模型
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)

                # *500个batch，会保存一次model
                if (iteration % model_period == 0):
                    save_model_v2(model, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, periodic_path.format(iteration))
                    save_model_v2(model, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, last_path)

                train_loss = train_num = 0

            # *对模型进行验证
            if iteration > 0 and (iteration % validation_interval == 0):
                # Model checkpoint
                # save_model(model, optimizer, train_loss, val_loss, iteration, batch_size, epoch,
                #            periodic_path.format(iteration))

                # Train cumulative stats
                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)
                train_labels_list = []
                train_pred_list = []

                train_roc_auc = roc_auc_score(train_labels, train_pred)
                # train_f1 = f1_score(train_labels, train_pred)
                # tb.add_scalar('train/f1', train_f1, iteration)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.add_pr_curve('train/pr', train_labels,
                                train_pred, iteration)

                # Validation
                device = 1
                val_loss = validation_routine(
                    model, val_loader, criterion, tb, iteration, 'val')
                tb.flush()

                # LR Scheduler
                lr_scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model_v2(model, optimizer, train_loss, val_loss,
                                  iteration, batch_size, epoch, bestval_path)
            # *每迭代一个batch +1
            iteration = iteration + 1
        epoch = epoch + 1


'''
method definition:
------------------
:param net          训练所用模型
:param device       训练所用的cuda编号
:param criterion    训练所用的损失函数
:param data         训练用的batch大小图片，ndarray
:param labels       训练用的标签
-----------------
:return loss[float]          训练所产生的损失    
:return pred[ndarray]        训练的预测值
'''


def batch_forward(model: nn.Module, criterion, data: torch.Tensor, labels: torch.Tensor):
    data = data.cuda(non_blocking=True)
    labels = labels.cuda(non_blocking=True)
    outputs = model(data)
    # 将网络的输出转化为[0,1]，同时转为nadarray
    pred = torch.sigmoid(outputs).detach().cpu().numpy()
    # 计算Loss
    loss = criterion(outputs, labels)
    return loss, pred


'''
method definition :用于将训练过程中得到weight进行保存 
------------------
:param net[nn.Module]   过程中需要进行保存的model 
:param optimizer    训练过程中所使用的的优化器 
:param train_loss   当前的训练损失    
:param val_loss     当前的验证损失  
:param iteration    当前的迭代次数(迭代了多少个batch) 
:param batch_size   当前所采用的的batch大小
:param epoch        当前处于第几个eopch 
:param path         以上所有数据的存储路径
------------------
'''


def save_model_v2(model: nn.Module, optimizer: torch.optim.Optimizer,
                  train_loss: float, val_loss: float,
                  iteration: int, batch_size: int, epoch: int,
                  path: str):
    path = str(path)
    model_state_dict = model.state_dict()
    # optimizer_state_dict =optimizer.state_dict()
    for key in model_state_dict.keys():
        model_state_dict[key] = model_state_dict[key].cpu()

    # for key in optimizer.:
    #     optimizer_state_dict[key] = optimizer_state_dict[key].cpu()
    state = dict(model=model_state_dict,
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


def save_model(model: nn.Module, optimizer: torch.optim.Optimizer,
               train_loss: float, val_loss: float,
               iteration: int, batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(model=model.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)


'''
method definition:
-----------------
:param  net              需要验证的模型  
:param  device           运行设备编号
:param  val_loader       验证集的dataloader
:param  criterion        损失函数
:param  tb               tensorboard的实例化对象
:param  iteration        当前迭代次数
:param  tag              训练数据集名称，划分相应的一些信息
:param  loader_len_norm  每次验证集读取的个数
----------------
:return val_loss         模型在验证集上的loss 

'''


def validation_routine(net, val_loader, criterion, tb, iteration, tag: str, loader_len_norm: int = None):
    # switch to eval mode
    net.eval()

    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader)):

        batch_data, batch_labels = val_data
        # 给定batch大小
        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(net, criterion, batch_data,
                                                           batch_labels)
        pred_list.append(val_batch_pred.flatten())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        val_labels = np.concatenate(labels_list)
        val_pred = np.concatenate(pred_list)
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        #val_f1 = f1_score(val_labels, val_pred)
        tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
        #tb.add_scalar('{}/f1'.format(tag), val_f1, iteration)
        tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')  # gpu总数量
    parser.add_argument('--epochs', default=30, type=int,
                        metavar='N', help='number of total epochs to run')
    parser.add_argument('--backend', default='nccl', type=str,
                        help='backend used for distributed train')
    parser.add_argument('--syncbn', default=False, action="store_true",
                        help='whether to use syncbn while training')
    parser.add_argument('--net', type=str,
                        help='Net model class', default="EfficientNetB4")

    parser.add_argument('--traindb', type=str, help='Training datasets', nargs='+', choices=split.available_datasets,
                        required=True)
    parser.add_argument('--valdb', type=str, help='Validation datasets', nargs='+', choices=split.available_datasets,
                        required=True)

    parser.add_argument('--ffpp_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.', default="/mnt/8T/FFPP/df/output/FFPP_df.pkl")
    parser.add_argument('--ffpp_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.', default="/mnt/8T/FFPP/faces/output")

    parser.add_argument('--dfdc_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')
    parser.add_argument('--dfdc_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the DFDC dataset. '
                             'Required for training/validating on the DFDC dataset.')

    parser.add_argument('--face', type=str, help='Face crop or scale',
                        choices=['scale', 'tight'], default='scale')
    parser.add_argument('--size', type=int,
                        help='Train patch size', default=224)
    parser.add_argument('--batch', type=int,
                        help='Batch size to fit in GPU memory', default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--valint', type=int,
                        help='Validation interval (iterations)', default=100)
    parser.add_argument('--logint', type=int,
                        help='Training log interval (iterations)', default=100)
    parser.add_argument('--modelperiod', type=int,
                        help='model save period (iterations)', default=50)
    parser.add_argument('--patience', type=int, help='Patience before dropping the LR [validation intervals]',
                        default=10)
    # parser.add_argument('--maxiter', type=int,
    #                     help='Maximum number of iterations', default=20000)
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true',
                        help='Train from scratch')

    # parser.add_argument('--trainsamples', type=int,
    #                     help='Limit the number of train samples per epoch', default=-1)
    parser.add_argument('--valsamples', type=int, help='Limit the number of validation samples per epoch',
                        default=6000)

    parser.add_argument('--workers', type=int,
                        help='Num workers for data loaders', default=4)
    # parser.add_argument('--device', type=int, help='GPU device id', default=0)
    parser.add_argument('--seed', type=int, help='Random seed', default=3)

    parser.add_argument('--debug', action='store_true', help='Activate debug')
    parser.add_argument('--suffix', type=str, help='Suffix to default tag')

    parser.add_argument('--attention', action='store_true',
                        help='Enable Tensorboard log of attention masks')
    parser.add_argument('--log_dir', type=str, help='Directory for saving the training logs',
                        default='/mnt/8T/multicard/runs/binclass/')
    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default='/mnt/8T/multicard/weights/binclass/')

    args = parser.parse_args()

    args.world_size = args.gpus  # 进程总数
    args.init_method = 'tcp://10.249.178.201:34567'
    mp.spawn(dist_train, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
