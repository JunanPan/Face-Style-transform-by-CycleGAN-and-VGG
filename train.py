from models import CycleGAN
import argparse
import shutil
from utils import *

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1" #设置指定GPU

def parse_args():
    #进行参数的解析和配置
    desc = "photo2cartoon"#真实图像到卡通图像
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='train', help='训练/测试')
    parser.add_argument('--dataset', type=str, default='photo2cartoon', help='数据集名称')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size大小')
    parser.add_argument('--iteration', type=int, default=1000000, help='总共训练轮数')
    parser.add_argument('--print_freq', type=int, default=1000, help='训练时打印的频率')
    parser.add_argument('--save_freq', type=int, default=10000, help='训练时模型保存的频率')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='训练时学习率是否衰减')
    parser.add_argument('--lr', type=float, default=0.0001, help='训练的学习率')
    parser.add_argument('--adv_weight', type=int, default=1, help='对抗损失的权重')
    parser.add_argument('--cycle_weight', type=int, default=50, help='循环一致性损失的权重')
    parser.add_argument('--identity_weight', type=int, default=10, help='identity损失（A放入GBA的结果与A自身')
    parser.add_argument('--cam_weight', type=int, default=1000, help='CAM权重')#CAM 指的是 经过w加权的特征图集重叠而成的一个特征图
    parser.add_argument('--faceid_weight', type=int, default=1, help='Face ID权重')#调用facenet获得的面部损失

    parser.add_argument('--ch', type=int, default=32, help='每层的基础通道数')
    parser.add_argument('--n_dis', type=int, default=6, help='判别器层数')

    parser.add_argument('--img_size', type=int, default=256, help='图像大小')
    parser.add_argument('--img_ch', type=int, default=3, help='图像通道数（默认3通道）')

    parser.add_argument('--device', type=str, default='cuda:0', help='设置使用cpu还是cuda')
    parser.add_argument('--resume', type=str2bool, default=False)#中断训练后，从之前的模型继续开始


    args = parser.parse_args()
    #用当前的参数命名结果目录
    args.result_dir = './experiment/{}-size{}-ch{}-{}-lr{}-adv{}-cyc{}-id{}-identity{}-cam{}'.format(
        os.path.basename(__file__)[:-3],
        args.img_size,
        args.ch,
        args.light,
        args.lr,
        args.adv_weight,
        args.cycle_weight,
        args.faceid_weight,
        args.identity_weight,
        args.cam_weight)

    return check_args(args)


def check_args(args):#检查目录都是否存在
    #目录存在则没问题，不存在则创建
    check_folder(os.path.join(args.result_dir, args.dataset, 'model'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'img'))
    check_folder(os.path.join(args.result_dir, args.dataset, 'test'))
    shutil.copy(__file__, args.result_dir)
    return args


def main():
    #主函数里，先获取参数
    args = parse_args()
    if args is None:
        exit()#无参数则退出

    # gan = UgatitSadalinHourglass(args)#创建模型
    gan = CycleGAN(args)#创建模型类
    gan.build_model()#建立模型

    if args.phase == 'train':
        gan.train()
        print(" [*] Training finished!")

    if args.phase == 'test':
        gan.test()
        print(" [*] Test finished!")


if __name__ == '__main__':
    main()
