import time
import itertools
from dataset import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from .networks import *
from utils import *
from glob import glob
from .face_features import FaceFeatures


class CycleGAN(object):
    def __init__(self, args):
        #各类参数的初始化 对应的就是train里面的所有输入
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.iteration = args.iteration
        self.decay_flag = args.decay_flag
        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq
        self.lr = args.lr
        self.ch = args.ch
        #权重初始化
        self.adv_weight = args.adv_weight#对抗损失权重
        self.cycle_weight = args.cycle_weight#循环一致性损失权重
        self.identity_weight = args.identity_weight#自身一致性权重 A进入BA还是A
        self.cam_weight = args.cam_weight
        self.faceid_weight = args.faceid_weight
        
        self.n_dis = args.n_dis
        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.resume = args.resume

    def build_model(self):
        #图像预处理操作
        #训练图像
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((self.img_size + 30, self.img_size + 30)),
            transforms.RandomCrop(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        #测试图像的处理，就不需要随机裁剪与翻转了
        test_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        #把文件下的图片集加载进loader
        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'), train_transform)
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'), train_transform)
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'), test_transform)
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'), test_transform)

        self.trainA_loader = DataLoader(self.trainA, batch_size=self.batch_size, shuffle=True)
        self.trainB_loader = DataLoader(self.trainB, batch_size=self.batch_size, shuffle=True)
        self.testA_loader = DataLoader(self.testA, batch_size=1, shuffle=False)
        self.testB_loader = DataLoader(self.testB, batch_size=1, shuffle=False)

        # 定义生成器
        self.genA2B = ResnetGenerator(ngf=self.ch, img_size=self.img_size).to(self.device)
        self.genB2A = ResnetGenerator(ngf=self.ch, img_size=self.img_size).to(self.device)
        # 定义判别器
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7).to(self.device)

        # mobilefacenet是运行在移动设备上的网络，单个网络模型只有4M并且有很高的准确率
        self.facenet = FaceFeatures('models/model_mobilefacenet.pth', self.device)

        # 定义损失函数
        self.L1_loss = nn.L1Loss().to(self.device)
        self.MSE_loss = nn.MSELoss().to(self.device)  # mean square error 均方误差
        self.BCE_loss = nn.BCEWithLogitsLoss().to(self.device)  # 不带sigmoid的交叉熵损失函数

        # 定义优化器
        # 生成器的优化器 参数里放生成器的参数
        self.G_optim = torch.optim.Adam(itertools.chain(self.genA2B.parameters(), self.genB2A.parameters()), lr=self.lr,
                                        betas=(0.5, 0.999), weight_decay=0.0001)
        # 判别器的优化器 参数里放判别器的参数
        self.D_optim = torch.optim.Adam(
            itertools.chain(self.disGA.parameters(), self.disGB.parameters()),
            lr=self.lr, betas=(0.5, 0.999), weight_decay=0.0001
        )


    def train(self):
        # 所有生成器与判别器都调用trian方法
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train()
        # 初始化迭代次数为1
        start_iter = 1
        if self.resume:
            # 找到存模型的路径
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
            if not len(model_list) == 0:  # 假如原来有记录
                model_list.sort()  # 根据模型名进行排序
                # 找到最新的迭代次数，作为开始
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                # 载入这个模型
                self.load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                # 如果当前迭代次数大于了总迭代次数的一半了，相当于训练到了后半程
                # 学习率相应地减少一些
                if self.decay_flag and start_iter > (self.iteration // 2):
                    self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (
                                start_iter - self.iteration // 2)
                    self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (
                                start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        # 直接从开始轮数到总论述写一个range
        for step in range(start_iter, self.iteration + 1):
            # 如果当前迭代次数大于了总迭代次数的一半了，相当于训练到了后半程
            # 学习率相应地减少一些
            if self.decay_flag and step > (self.iteration // 2):
                self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            # 如果trainA_iter存在，就直接得到real_A,否则就先初始化trainA_iter再得到
            try:
                real_A, _ = trainA_iter.next()
            except:
                trainA_iter = iter(self.trainA_loader)
                real_A, _ = trainA_iter.next()

            try:
                real_B, _ = trainB_iter.next()
            except:
                trainB_iter = iter(self.trainB_loader)
                real_B, _ = trainB_iter.next()

            real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # 训练判别器
            self.D_optim.zero_grad()

            fake_A2B, _, _ = self.genA2B(real_A)  # 由real_A得到的fake_A2B
            fake_B2A, _, _ = self.genB2A(real_B)  # 由real_B得到的fake_B2A

            # 获得各判别器对real数据的判别结果
            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            # 获得各判别器对fake数据的判别结果
            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)

            # 给判别器A用的 由GA得到的综合损失（真+假）
            D_ad_loss_GA = self.MSE_loss(real_GA_logit, torch.ones_like(real_GA_logit).to(self.device)) + \
                           self.MSE_loss(fake_GA_logit, torch.zeros_like(fake_GA_logit).to(self.device))

            # GA 的总和cam损失（真+假）
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, torch.ones_like(real_GA_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_GA_cam_logit, torch.zeros_like(fake_GA_cam_logit).to(self.device))

            # GB也是一样 对称的
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, torch.ones_like(real_GB_logit).to(self.device)) + \
                           self.MSE_loss(fake_GB_logit, torch.zeros_like(fake_GB_logit).to(self.device))

            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, torch.ones_like(real_GB_cam_logit).to(self.device)) + \
                               self.MSE_loss(fake_GB_cam_logit, torch.zeros_like(fake_GB_cam_logit).to(self.device))

            # 判别器A总损失
            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA )
            # 判别器B总损失
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB )
            # 判别器损失
            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.step()

            # 训练生成器
            self.G_optim.zero_grad()
            # 得到第一层正向结果
            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)
            # 得到第二层逆向结果（循环一致性）
            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)
            # 以及自身不变性
            # A送入GBA，应原封不动送出A
            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            # B送入GAB，应原封不动送出B
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            #对抗损失
            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, torch.ones_like(fake_GA_logit).to(self.device))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, torch.ones_like(fake_GA_cam_logit).to(self.device))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, torch.ones_like(fake_GB_logit).to(self.device))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, torch.ones_like(fake_GB_cam_logit).to(self.device))
            # 循环一致性损失
            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)
            # 自身一致性损失
            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)
            # 调用facenet的面部损失计算
            G_id_loss_A = self.facenet.cosine_distance(real_A, fake_A2B)
            G_id_loss_B = self.facenet.cosine_distance(real_B, fake_B2A)
            # 生成器Acam损失
            G_cam_loss_A = self.BCE_loss(fake_B2A_cam_logit, torch.ones_like(fake_B2A_cam_logit).to(self.device)) + \
                           self.BCE_loss(fake_A2A_cam_logit, torch.zeros_like(fake_A2A_cam_logit).to(self.device))
            # 生成器Bcam损失
            G_cam_loss_B = self.BCE_loss(fake_A2B_cam_logit, torch.ones_like(fake_A2B_cam_logit).to(self.device)) + \
                           self.BCE_loss(fake_B2B_cam_logit, torch.zeros_like(fake_B2B_cam_logit).to(self.device))
            # 生成器A总loss
            G_loss_A = self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA) + \
                       self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + \
                       self.cam_weight * G_cam_loss_A + self.faceid_weight * G_id_loss_A
            # 生成器B总loss
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB) + \
                       self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + \
                       self.cam_weight * G_cam_loss_B + self.faceid_weight * G_id_loss_B
            # 总生成器loss
            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()  # 反向传播
            self.G_optim.step()  # 更新



            if step % 10 == 0:
                print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (
                step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval()
                with torch.no_grad():
                    for _ in range(train_sample_num):
                        try:
                            real_A, _ = trainA_iter.next()
                        except:
                            trainA_iter = iter(self.trainA_loader)
                            real_A, _ = trainA_iter.next()

                        try:
                            real_B, _ = trainB_iter.next()
                        except:
                            trainB_iter = iter(self.trainB_loader)
                            real_B, _ = trainB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)),
                                             1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)),
                                             1)

                    for _ in range(test_sample_num):
                        try:
                            real_A, _ = testA_iter.next()
                        except:
                            testA_iter = iter(self.testA_loader)
                            real_A, _ = testA_iter.next()

                        try:
                            real_B, _ = testB_iter.next()
                        except:
                            testB_iter = iter(self.testB_loader)
                            real_B, _ = testB_iter.next()
                        real_A, real_B = real_A.to(self.device), real_B.to(self.device)

                        fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                        fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                        fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                        fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                        fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                        fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                        A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                                   cam(tensor2numpy(fake_A2A_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                                   cam(tensor2numpy(fake_A2B_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                                   cam(tensor2numpy(fake_A2B2A_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)),
                                             1)

                        B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                                   cam(tensor2numpy(fake_B2B_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                                   cam(tensor2numpy(fake_B2A_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                                   cam(tensor2numpy(fake_B2A2B_heatmap[0]),
                                                                       self.img_size),
                                                                   RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)),
                                             1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 1000 == 0:
                params = {}
                params['genA2B'] = self.genA2B.state_dict()
                params['genB2A'] = self.genB2A.state_dict()
                params['disGA'] = self.disGA.state_dict()
                params['disGB'] = self.disGB.state_dict()
                torch.save(params, os.path.join(self.result_dir, self.dataset + '_params_latest.pt'))

    def save(self, dir, step):
        params = {}
        params['genA2B'] = self.genA2B.state_dict()
        params['genB2A'] = self.genB2A.state_dict()
        params['disGA'] = self.disGA.state_dict()
        params['disGB'] = self.disGB.state_dict()
        torch.save(params, os.path.join(dir, self.dataset + '_params_%07d.pt' % step))

    def load(self, dir, step):
        params = torch.load(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        self.genA2B.load_state_dict(params['genA2B'])
        self.genB2A.load_state_dict(params['genB2A'])
        self.disGA.load_state_dict(params['disGA'])
        self.disGB.load_state_dict(params['disGB'])

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pt'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return

        self.genA2B.eval(), self.genB2A.eval()
        with torch.no_grad():
            for n, (real_A, _) in enumerate(self.testA_loader):
                real_A = real_A.to(self.device)

                fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

                fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

                fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

                A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                      cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                      cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                      cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

            for n, (real_B, _) in enumerate(self.testB_loader):
                real_B = real_B.to(self.device)

                fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                      cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                      cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                      cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                      RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)
