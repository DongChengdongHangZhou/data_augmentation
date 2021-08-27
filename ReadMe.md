To install the albumentations package with imgaug package, you should run
the following instruction:
pip install -U albumentations[imgaug]


the augmentation methods I used:
GlassBlur(iterations=1,max_delta=2) 玻璃模糊
MotionBlur 运动模糊
GaussianBlur 高斯模糊
ChannelDropout
ColorJitter
HorizontalFlip 
Affine
PiecewiseAffine  
Superpixels 基于SLIC的超分表达
Perspective 透视效果
Emboss 浮雕滤镜
Equalize
FancyPCA
ImageCompression(quality_lower=10, quality_upper=20)
Sharpen
DownScale(scale_min=0.5,scale_max=0.5) 先降采样再升采样，降低图像质量 
Cutout(num_holes=25)
ShiftScaleRotate(shift_limit=[-0.08,0.08],scale_limit=[-0.5,0.2],rotate_limit=[-20,20],border_mode=1) 平移缩放旋转
HueSaturationValue 色调饱和度值
InvertImg 反转图像色彩
RandomBrightnessContrast 随机亮度对比度
RandomRain(blur_value=3)
RandomShadow
RandomSnow
RandomFog
RandomSunFlare(src_radius=100)
CLAHE 对比度受限自适应直方图均衡
ChannelShuffle 通道洗牌
GaussNoise 高斯噪声
ISONoise
Solarize
ToGray
ToSepia 此为sepia滤镜

