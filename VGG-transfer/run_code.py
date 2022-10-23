import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from load_img import load_img, show_img
from torch.autograd import Variable
from build_model import get_style_model_and_loss

def get_input_param_optimier(input_img):#得到要训练的图片参数和优化器
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer #input_param就是要训练的图片  optimizer优化器


def run_style_transfer(content_img, style_img, input_img, num_epoches=300):#传入内容图像，风格图像，
    print('构建风格转换模型')
    # 提前用固定参数的网络计算好内容图片和风格图片的内容和风格
    model, style_loss_list, content_loss_list = get_style_model_and_loss(style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)

    print('优化中')
    epoch = [0]
    while epoch[0] < num_epoches:

        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)#把目标图像送入模型，计算得到相关的内容loss和风格loss
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()#通过这里反向传播 定位到loss.py当中的Style_Loss类函数backward
            for cl in content_loss_list:
                content_score += cl.backward()#通过这里反向传播 定位到loss.py当中的Content_Loss类函数backward

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)

    return input_param.data

#获得内容图片
style_img = load_img('./picture/style1.png')
style_img = Variable(style_img).cuda()
#获得风格图片
content_img = load_img('./picture/content1.png')
content_img = Variable(content_img).cuda()
#用内容图片初始化结果图片 也就是input_img，我们后续就是训练这个input_img
input_img = content_img.clone()

#把三个img放入函数训练
out = run_style_transfer(content_img, style_img, input_img, num_epoches=250)

#展示结果
# show_img(out.cpu())
show_img(out.cuda())

#保存结果
save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('./picture/saved_picture1.png')