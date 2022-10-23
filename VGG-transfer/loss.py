import torch.nn as nn
import torch
#内容损失
class Content_Loss(nn.Module):
    def __init__(self, target, weight):#weight处理输入，与target计算损失
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight
        # 必须要用detach来分离出target，这时候target不再是一个Variable，这是为了动态计算梯度，否则forward会出错，不能向前传播
        self.criterion = nn.MSELoss()

    def forward(self, input):
        #直接用mse损失来计算交叉熵损失函数
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out#原样输出

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss


class Gram(nn.Module):#计算风格 对于输入
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)#计算初步特征
        gram = torch.mm(feature, feature.t())#mm矩阵相乘，得到最终特征
        gram /= (a * b * c * d)#进行标准化
        return gram

#风格损失
class Style_Loss(nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach() * self.weight#target是不需要更新的，固定不变的
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    
    def forward(self, input):
        G = self.gram(input) * self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out#原样输出即可

    def backward(self, retain_variabels=True):
        #总的来说进行一次backward之后，各个节点的值会清除，这样进行第二次backward会报错，如果加上retain_graph==True后,可以再来一次backward。
        self.loss.backward(retain_graph=retain_variabels)#pytorch是动态图计算机制，在每一次反向传播计算梯度的循环内，pytorch先建立正向计算图，然后使用反向传播计算梯度，同时被销毁计算图
        return self.loss
