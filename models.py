
from cv2 import Feature2D
from architectures import fornet
from torch import nn
import torch
from pytorch_wavelets import DWTForward
from torch.nn.modules import linear
from torch.nn.modules.batchnorm import BatchNorm2d
from torchvision import transforms


# this part is for wavelet data generation


class WaveGen(nn.Module):

    def __init__(self):
        super().__init__()
        self.DWT1 = DWTForward(J=1, mode='symmetric', wave='db2')
        self.DWT2 = DWTForward(J=1, mode='symmetric', wave='coif1')
        self.DWT3 = DWTForward(J=1, mode='symmetric', wave='sym2')

    '''
    :method description: gain the DWT descompose of X, only keep the high frequency coefficients
    ----------------------
    :param: x
    ----------------------
    :return 1 layer wavelet decompose of x 
    '''

    def DWT(self, x: torch.Tensor):
        size = int(x.shape[-1]/2)
        batch = x.shape[0]

        x_1 = self.DWT1(x)[1][0][..., :size, :size]
        x_2 = self.DWT2(x)[1][0][..., :size, :size]
        x_3 = self.DWT3(x)[1][0][..., :size, :size]
        x_1 = torch.reshape(x_1, (batch, -1, size, size))
        x_2 = torch.reshape(x_2, (batch, -1, size, size))
        x_3 = torch.reshape(x_3, (batch, -1, size, size))
        y = torch.cat((x_1, x_2, x_3), dim=1)
        return y

    def features(self, x: torch.Tensor):
        # get the Tensor first, Tensor size=[B,C,D(HVD),H,W]
        x = self.DWT(x)
        # x = self.DWT(x)

        return x

    def forward(self, x):
        x = self.features(x)
        return x


# this branch only work with wavelet
class ProcessNet_wave(nn.Module):
    def __init__(self, waves_n):
        super().__init__()

        # the argument I need
        in_channel = waves_n * 9
        self.in_channel = in_channel
        # the module I need

        # Head
        self.DWT = WaveGen()
        self.entry_l1 = nn.Conv2d(
            in_channel, in_channel*3, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channel*3)
        self.relu = nn.ReLU(inplace=True)

        # middle flow
        # Block缩小尺寸的方法是池化
        self.block1 = Block(
            in_channel*3, in_channel*9, 4, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(
            in_channel*9, in_channel*6, 2, 1, start_with_relu=False, grow_first=True)

        # 此时需要concat l2的DWT
        self.block3 = Block(
            in_channel*15, in_channel*30, 4, 2, start_with_relu=False, grow_first=True)
        self.block4 = Block(
            in_channel*30, in_channel*18, 2, 1, start_with_relu=False, grow_first=True)

        # 压缩L3的DWT
        self.block5 = Block(
            in_channel*81, in_channel*9, 4, 1, start_with_relu=False, grow_first=True)
        # 此时需要concat l3的DWT
        self.block6 = Block(
            in_channel*27, in_channel*16, 2, 1, start_with_relu=False, grow_first=True)

        self.block7 = Block(
            in_channel*16, in_channel*8, 2, 1, start_with_relu=False, grow_first=True)
        # TODO
        self.block8 = Block(
            in_channel*8, in_channel, 2, 1, start_with_relu=False, grow_first=True)

        self.block9 = Block(
            in_channel, 8, 2, 1, start_with_relu=False, grow_first=True)

        self.fea_compress = nn.Linear(6272, 1000)
        self.bn2 = nn.BatchNorm2d(1000)

    def features(self, input):
        x = input
        # 第一步 将input[B,C,H,W]转化为w_l1[B,C*9,H/2,w/2]
        w_l1 = self.DWT(x)

        # 第二步 对一阶段的高频系数进行卷积操作，得x[B,c*54,H/4,W/4]
        x = self.entry_l1(w_l1)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)

        # 第三步 求x的二阶段的高频系数w_l2[B,c*81,H/4,W/4]
        w_l2 = self.DWT(w_l1)
        w_l1 = 0

        # 第四步 将w_l2与step2 concat，得x[B,c*135,H/4,w/4]
        x = torch.cat((w_l2, x), dim=1)

        # 第五步 将第四部所得继续卷积操作,得x[B,c*162,H/8,W/8]
        x = self.block3(x)
        x = self.block4(x)

        # 第六步 求x的第三阶高频系数w_l3[b,c*729,H/8,W/8]，并将其压缩到[b,c*81]

        w_l3 = self.DWT(w_l2)
        w_l3 = self.block5(w_l3)
        w_l2 = 0

        # 第七步 将w_l3与step5 concat，得x[B,c*972,H/8,w/8]
        x = torch.cat((w_l3, x), dim=1)

        # 第八步 开始压缩，得x[B,c*27,H/32,W/32]
        x = self.block6(x)
        x = self.block7(x)

        x = self.block8(x)
        x = self.block9(x)
        # 第九步 x展开成长条
        x = x.view(x.size(0), -1)

        # 第十步，最终该分支就会输出对应的特征
        x = self.fea_compress(x)
        x = self.bn2(x)
        return x

    def forward(self, input):
        # step 1 turn input to level1 DWT high frequency coefficients
        x = input
        fea = self.features(x)

        return fea

# use the logits I gain to generate the pred


class JudgeNet(nn.Module):
    def __init__(self, fea_n, judgeWay):
        super().__init__()
        self.fea_n = fea_n
        self.judgeWay = judgeWay
        self.j_linear = self.GenJudgeList(fea_n, judgeWay)

        self.sig = nn.Sigmoid()

    def GenJudgeList(self, fea_n, judgeWay):

        return nn.Linear(fea_n, judgeWay)

    def Confidentiate(self, judges):
        x = judges

        judges = torch.mul(x, x)
        sig = torch.sign(x)
        judges = torch.mul(sig, judges)
        return judges

    def forward(self, fea):
        x = fea
        judges = self.j_linear(x)
        judges = self.Confidentiate(judges)
        # judges = torch.Tensor(judges)
        # judges = judges/10

        # 1 先平均值再sigmoid
        # judges = self.sig(judges)
        # judge = judges.mean(1, keepdim=True)

        # 2 先sigmoid再平均值
        judge = judges.mean(1, keepdim=True)

        w = self.j_linear.weight
        return judge, w

# 此处研究不同模型得到fea的关系


class feaProcess(nn.Module):
    def __init__(self, fea_n):
        super().__init__()
        self.feaCompress1 = nn.Linear(fea_n, int(fea_n/2))
        self.dp1 = nn.Dropout(p=0.2)
        self.acti1 = nn.LeakyReLU(0.01)

        self.feaCompress2 = nn.Linear(int(fea_n/2), int(fea_n/2))
        self.dp2 = nn.Dropout(p=0.2)
        self.acti2 = nn.LeakyReLU(0.01)

        # self.feaCompress3 = nn.Linear(int(fea_n/4), int(fea_n/4))
        # self.dp3 = nn.Dropout(p=0.2)
        # self.acti3 = nn.LeakyReLU(0.01)

    def features(self, input):
        x = input
        x = self.feaCompress1(x)
        x = self.dp1(x)
        x = self.acti1(x)
        x = self.feaCompress2(x)
        x = self.dp2(x)
        x = self.acti2(x)
        # x = self.feaCompress3(x)
        # x = self.dp3(x)
        # x = self.acti3(x)
        return x

    def forward(self, input):
        x = self.features(input)
        w_1 = self.feaCompress1.weight
        w_2 = self.feaCompress2.weight
        # w_3 = self.feaCompress3.weight
        w = torch.mm(w_2, w_1)
        # w = torch.mm(w, w_3)
        # w可视作fea处理所用矩阵
        return x, w


# the whole net
class WholeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = self.EfficientNetB4Gen()
        self.branch2 = ProcessNet_wave(3)
        self.judge = JudgeNet(1396, 10)
        self.feaProcess = feaProcess(2792)

    def get_trainable_parameters(self):
        a = filter(lambda p: p.requires_grad, self.parameters())
        return a

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def EfficientNetB4Gen(self):
        path = '/mnt/8T/hou/multicard/weights/binclass/net-EfficientNetB4_traindb-ff-c23-720-140-140_face-scale_size-224_seed-3_note-/bestval.pth'
        # net = fornet.EfficientNetB4()
        net = fornet.EfficientNetB4()
        # for k, v in net.named_parameters():
        #     v.requires_grad = False
        # for m in net.modules():
        #     if isinstance(m, nn.BatchNorm2d):
        #         m.eval()
        #         m.weight.requires_grad = False
        #         m.bias.requires_grad = False

        state_dict = torch.load(path)
        incomp_keys = net.load_state_dict(
            {k.replace('module.', ''): v for k, v in state_dict['model'].items()})
        print(incomp_keys)
        for name, parameter in net.named_parameters():
            parameter.requries_grad = False
        return net

    def featureCon(self, fea_l):
        return torch.cat((fea_l), dim=0)

    def forward(self, input):
        x = input
        # step 1 dispatch data to branches,and then they shall return the features we need for judging
        fea1 = self.branch1.features(x)
        fea2 = self.branch2(x)
        # 用于运算KD_loss
        # t_out = self.branch1.classifier(fea1)
        # t_out = torch.sigmoid(t_out)
        fea = torch.cat((fea1, fea2), dim=1)
        # fea = fea2
        fea, w_feaP = self.feaProcess(fea)
        judge, w_j = self.judge(fea)
        w = torch.mm(w_j, w_feaP)

        w = torch.abs(w)
        # pow_fea1 = w[0:1396].mean()
        # pow_fea2 = w[1396:].mean()
        return judge, w


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size,
                               stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        # torch.cuda.empty_cache()
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()

        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters,
                                  1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None

        self.relu = nn.ReLU(inplace=True)
        rep = []

        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters

        for i in range(reps-1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))

        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters,
                       3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))

        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)

        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)

        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp

        x += skip
        return x
