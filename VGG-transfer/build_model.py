import torch.nn as nn
import torchvision.models as models

import loss

#直接加载torch.hub中 vgg模型
vgg = models.vgg19(pretrained=True).features
vgg = vgg.cuda()


#定义风格特征与内容特征层的位置
#用第四层的输出来匹配内容
content_layers_default = ['conv_4'] #内容取相对靠后一点的层
#用1 2 3 4 5层的输出来匹配风格
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_style_model_and_loss(style_img,
                             content_img,
                             cnn=vgg,
                             style_weight=1000,
                             content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential()
    model = model.cuda()
    gram = loss.Gram()
    gram = gram.cuda()

    i = 1
    #遍历这个vgg网络 通过vgg网络辅助构建
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):#如果这个层是卷积层
            name = 'conv_' + str(i)
            model.add_module(name, layer)#是卷积层就放进model的sequential
            if name in content_layers_default:#如果它是内容特征层
                target = model(content_img)#把内容图片送入模型，得到当前累计层的结果
                content_loss = loss.Content_Loss(target, content_weight)#进行内容损失计算 这里没有传入input_image，input_image在run_code.py的model(input_param)时送入
                model.add_module('content_loss_' + str(i), content_loss)#添加这个内容损失
                content_loss_list.append(content_loss)#记录下内容损失
            if name in style_layers_default:#如果它是风格特征层
                target = model(style_img)#送入模型之后
                target = gram(target)#计算风格
                style_loss = loss.Style_Loss(target, style_weight) #计算风格损失
                model.add_module('style_loss_' + str(i), style_loss)#添加进模型层
                style_loss_list.append(style_loss)#记录风格损失

            i += 1
        if isinstance(layer, nn.MaxPool2d):#如果这个层是池化层
            name = 'pool_' + str(i) #把池化层加进去
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU): #如果事激活函数曾
            name = 'relu' + str(i)
            model.add_module(name, layer)#命名并加进模型

    return model, style_loss_list, content_loss_list
