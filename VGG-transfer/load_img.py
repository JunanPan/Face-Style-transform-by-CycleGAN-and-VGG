import PIL.Image as Image
import torchvision.transforms as transforms

img_size = 512

#加载图片并进行处理
def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img

#展示图片
def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()

