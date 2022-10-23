import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


#生成器定义
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc    #输入通道数 --> 3
        self.output_nc = output_nc  #输出通道数 --> 3
        self.ngf = ngf              #第一层卷积后的通道数 --> 64
        self.n_blocks = n_blocks    #残差块数 --> 6
        self.img_size = img_size    #图像size --> 256
 
        DownBlock = []
        # 先通过一个卷积核尺寸为7的卷积层，图片大小不变，通道数变为64
        DownBlock += [nn.ReflectionPad2d(3),
                      nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=False),
                      nn.InstanceNorm2d(ngf),
                      nn.ReLU(True)]
 
        # Down-Sampling --> 下采样模块
        n_downsampling = 2
        # 两层下采样，img_size缩小4倍（64），通道数扩大4倍（256）
        for i in range(n_downsampling): 
            mult = 2**i
            DownBlock += [nn.ReflectionPad2d(1),
                          nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=0, bias=False),
                          nn.InstanceNorm2d(ngf * mult * 2),
                          nn.ReLU(True)]
 
        # Down-Sampling Bottleneck  --> 编码器中的残差模块
        mult = 2**n_downsampling
        # 6个残差块，尺寸和通道数都不变
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]
 
        # Class Activation Map --> 产生类别激活图
        #接着global average pooling后的全连接层
        self.gap_fc = nn.Linear(ngf * mult, 1, bias=False)
        #接着global max pooling后的全连接层
        self.gmp_fc = nn.Linear(ngf * mult, 1, bias=False)
        #下面1x1卷积和激活函数，是为了得到两个pooling合并后的特征图
        self.conv1x1 = nn.Conv2d(ngf * mult * 2, ngf * mult, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
 
        # Gamma, Beta block --> 生成自适应 L-B Normalization(AdaILN)中的Gamma, Beta
        #1024x1024 --> 256的全连接层和一个256 --> 256的全连接层
        FC = [nn.Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias=False), # (1024x1014, 64x4) crazy
                nn.ReLU(True),
                nn.Linear(ngf * mult, ngf * mult, bias=False),
                nn.ReLU(True)]
        #AdaILN中的Gamma, Beta
        self.gamma = nn.Linear(ngf * mult, ngf * mult, bias=False)
        self.beta = nn.Linear(ngf * mult, ngf * mult, bias=False)
        
        # Up-Sampling Bottleneck --> 解码器中的自适应残差模块
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetSoftAdaLINBlock(ngf * mult, use_bias=False))
 
        # Up-Sampling --> 解码器中的上采样模块
        UpBlock2 = []
        #上采样与编码器的下采样对应
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.ReflectionPad2d(1),
                         nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=0, bias=False),
                         adaLIN(int(ngf * mult / 2)), 
                         nn.ReLU(True)]
        #最后一层卷积层，与最开始的卷积层对应
        UpBlock2 += [nn.ReflectionPad2d(3),
                     nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0, bias=False),
                     nn.Tanh()]
        
        self.DownBlock = nn.Sequential(*DownBlock) #编码器整个模块
        self.FC = nn.Sequential(*FC)               #生成gamma,beta的全连接层模块
        self.UpBlock2 = nn.Sequential(*UpBlock2)   #只包含上采样后的模块，不包含残差块
 
    def forward(self, input):
        skip = x = self.DownBlock(input)  #得到编码器的输出,对应途中encoder feature map
 
        gap = torch.nn.functional.adaptive_avg_pool2d(x, 1) #全局平均池化
        gap_logit = self.gap_fc(gap.view(x.shape[0], -1)) #gap的预测
        gap_weight = list(self.gap_fc.parameters())[0] #self.gap_fc的权重参数
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3) #得到全局平均池化加持权重的特征图
 
        gmp = torch.nn.functional.adaptive_max_pool2d(x, 1) #全局最大池化
        gmp_logit = self.gmp_fc(gmp.view(x.shape[0], -1)) #gmp的预测
        gmp_weight = list(self.gmp_fc.parameters())[0] #self.gmp_fc的权重参数
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3) #得到全局最大池化加持权重的特征图
 
        cam_logit = torch.cat([gap_logit, gmp_logit], 1) #结合gap和gmp的cam_logit预测
        x = torch.cat([gap, gmp], 1)  #结合两种池化后的特征图，通道数512
        x = self.relu(self.conv1x1(x)) #接入一个卷积层，通道数512转换为256
 


        x_ = self.FC(x.view(x.shape[0], -1))
        gamma, beta = self.gamma(x_), self.beta(x_) #得到自适应gamma和beta
 
 
        for i in range(self.n_blocks):
            #将自适应gamma和beta送入到AdaILN
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        x = x+skip
        out = self.UpBlock2(x) #通过上采样后的模块，得到生成结果
 
        return out, cam_logit #模型输出为生成结果，cam预测以及热力图


#用到了adaLIN
# 先将输入图像的编码特征统计信息和卡通特征统计信息相融合
#softadaLIN只是在adlin基础上 修改了gama和beta的来源，通过内容和特征进行计算融合
class SoftAdaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(SoftAdaLIN, self).__init__()
        self.norm = adaLIN(num_features, eps)

        self.w_gamma = Parameter(torch.zeros(1, num_features))
        self.w_beta = Parameter(torch.zeros(1, num_features))

        #计算content_gamma的网络
        self.c_gamma = nn.Sequential(nn.Linear(num_features, num_features),
                                     nn.ReLU(True),
                                     nn.Linear(num_features, num_features))
        #计算content_beta
        self.c_beta = nn.Sequential(nn.Linear(num_features, num_features),
                                    nn.ReLU(True),
                                    nn.Linear(num_features, num_features))
        #计算style_gamma
        self.s_gamma = nn.Linear(num_features, num_features)
        #计算style_beta
        self.s_beta = nn.Linear(num_features, num_features)

    def forward(self, x, content_features, style_features):
        content_gamma, content_beta = self.c_gamma(content_features), self.c_beta(content_features)
        style_gamma, style_beta = self.s_gamma(style_features), self.s_beta(style_features)

        w_gamma, w_beta = self.w_gamma.expand(x.shape[0], -1), self.w_beta.expand(x.shape[0], -1)
        soft_gamma = (1. - w_gamma) * style_gamma + w_gamma * content_gamma
        soft_beta = (1. - w_beta) * style_beta + w_beta * content_beta

        out = self.norm(x, soft_gamma, soft_beta)
        return out
#自适应层归一化 通道归一化
#先求两种规范化的值
class adaLIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(adaLIN, self).__init__()
        self.eps = eps
        #adaILN的参数ρ，通过这个参数来动态调整LN和IN的占比
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))
        self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        #IN只对长和宽进行求均值和标准差
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        #LN对通道，长和宽三个维度一起求均值和标准差
        #0是N 1是C，2和3是W和H
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)

        #out由部分out_in与部分out_ln组成
        #合并两种规范化(IN, LN)
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        #扩张得到结果
        out = out * gamma.unsqueeze(2).unsqueeze(3) + beta.unsqueeze(2).unsqueeze(3)

        return out
#ResnetGenerator用了 层归一化

#没有加入自适应的Layer-Instance Normalization，用于上采样
class LIN(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(LIN, self).__init__()
        self.eps = eps#防止分母为0的极小值ε
        self.rho = Parameter(torch.Tensor(1, num_features, 1, 1))#权重ρ
        self.gamma = Parameter(torch.Tensor(1, num_features, 1, 1))#γ
        self.beta = Parameter(torch.Tensor(1, num_features, 1, 1))#beta
        self.rho.data.fill_(0.0)
        self.gamma.data.fill_(1.0)
        self.beta.data.fill_(0.0)

    def forward(self, input):
        #使用IN，对W和H维度做均值和方差计算，得到IN下的均值in_mean和方差in_var
        in_mean, in_var = torch.mean(input, dim=[2, 3], keepdim=True), torch.var(input, dim=[2, 3], keepdim=True)
        #通过得到的in_mean和in_var进行归一化,得到IN结果out_in
        out_in = (input - in_mean) / torch.sqrt(in_var + self.eps)
        #使用LN，对C，W和H维度做均值和方差计算
        ln_mean, ln_var = torch.mean(input, dim=[1, 2, 3], keepdim=True), torch.var(input, dim=[1, 2, 3], keepdim=True)
        #同理得到LN归一化结果out_ln
        out_ln = (input - ln_mean) / torch.sqrt(ln_var + self.eps)
        #通过ρ控制IN和LN的加权融合
        out = self.rho.expand(input.shape[0], -1, -1, -1) * out_in + (1-self.rho.expand(input.shape[0], -1, -1, -1)) * out_ln
        #最外边再加上γ和β做调整变换
        out = out * self.gamma.expand(input.shape[0], -1, -1, -1) + self.beta.expand(input.shape[0], -1, -1, -1)
        #out就是我们LIN的最终结果啦
        return out


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConvBlock, self).__init__()
        self.dim_out = dim_out

        self.ConvBlock1 = nn.Sequential(nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_in, dim_out//2, kernel_size=3, stride=1, bias=False))

        self.ConvBlock2 = nn.Sequential(nn.InstanceNorm2d(dim_out//2),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_out//2, dim_out//4, kernel_size=3, stride=1, bias=False))

        self.ConvBlock3 = nn.Sequential(nn.InstanceNorm2d(dim_out//4),
                                        nn.ReLU(True),
                                        nn.ReflectionPad2d(1),
                                        nn.Conv2d(dim_out//4, dim_out//4, kernel_size=3, stride=1, bias=False))

        self.ConvBlock4 = nn.Sequential(nn.InstanceNorm2d(dim_in),
                                        nn.ReLU(True),
                                        nn.Conv2d(dim_in, dim_out, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        residual = x

        x1 = self.ConvBlock1(x)
        x2 = self.ConvBlock2(x1)
        x3 = self.ConvBlock3(x2)
        out = torch.cat((x1, x2, x3), 1)

        if residual.size(1) != self.dim_out:
            residual = self.ConvBlock4(residual)

        return residual + out
class ResnetBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]

        conv_block += [nn.ReflectionPad2d(1),
                       nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm2d(dim)]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


#判别器定义
class Discriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        #第一层
        model = [nn.ReflectionPad2d(1),#第一层下采样, 尺寸减半(128)，通道数为64
                 nn.utils.spectral_norm(
                 nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=0, bias=True)),
                 nn.LeakyReLU(0.2, True)]
        #第二层和第三层
        for i in range(1, n_layers - 2):#第二，三层下采样，尺寸再缩4倍(32)，通道数为256
            mult = 2 ** (i - 1)
            model += [nn.ReflectionPad2d(1),
                      nn.utils.spectral_norm(
                      nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=2, padding=0, bias=True)),
                      nn.LeakyReLU(0.2, True)]
        #第四层
        mult = 2 ** (n_layers - 2 - 1) 
        model += [nn.ReflectionPad2d(1),# 尺寸不变（32），通道数为512
                  nn.utils.spectral_norm(
                  nn.Conv2d(ndf * mult, ndf * mult * 2, kernel_size=4, stride=1, padding=0, bias=True)),
                  nn.LeakyReLU(0.2, True)]
        #最后一层
        model += [nn.Conv2d(ndf * mult * 2, 1, kernel_size=4, stride=1, padding=0,bias=True)]#输出通道数为1
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


#用到了SoftAdaLIN
class ResnetSoftAdaLINBlock(nn.Module):
    def __init__(self, dim, use_bias=False):
        super(ResnetSoftAdaLINBlock, self).__init__()
        self.pad1 = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm1 = SoftAdaLIN(dim)
        self.relu1 = nn.ReLU(True)

        self.pad2 = nn.ReflectionPad2d(1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=0, bias=use_bias)
        self.norm2 = SoftAdaLIN(dim)

    def forward(self, x, content_features, style_features):
        out = self.pad1(x)
        out = self.conv1(out)
        out = self.norm1(out, content_features, style_features)
        out = self.relu1(out)

        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, content_features, style_features)
        return out + x
