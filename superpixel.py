#  USAGE
#  python superpixel.py --image cactus.jpg
import torch
import matplotlib.pyplot as plt
from skimage import io
from skimage.segmentation import quickshift, mark_boundaries  # 导入mark_boundaries 以绘制实际的超像素分割
# 导入必要的包
from skimage.segmentation import slic  # 导入包以使用SLIC superpixel segmentation
from skimage.util import img_as_float


def superpixel_segmentation(image, numSegments=250):
    # segments = quickshift(image, ratio=0.8)
    segments = slic(image, n_segments=numSegments)
    print(type(segments), segments.shape, segments)
    return segments


my_input = torch.rand([128, 128]).numpy()
print('my_input.shape', my_input.shape)

image_path = './sample_1_image.png'
image = img_as_float(io.imread(image_path))
plt.imshow(image)
# 遍历超像素段的数量 研究3种尺寸不断增加的段，100、200、300
for numSegments in (350, 400):
    # 执行SLTC 超像素分割，该功能仅获取原始图像并覆盖我们的超像素段。
    # 仅有一个必需参数：
    # image：待执行SLTC超像素分割的图像
    # n_segments: 定义我们要生成多少个超像素段的参数，默认100
    # sigma：在分割之前应用的平滑高斯核
    segments = superpixel_segmentation(image, numSegments)
    # 绘制SLTC 的分割结果
    fig = plt.figure("Superpixels -- %d segments" % (numSegments))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(mark_boundaries(image, segments))
    plt.axis("off")

# 展示图像
plt.show()
