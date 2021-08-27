import random
from albumentations.augmentations.transforms import Equalize, HorizontalFlip
from albumentations.core.composition import OneOf
from albumentations.core.serialization import save
import cv2
import numpy as np
from matplotlib import pyplot as plt
import albumentations as A


BOX_COLOR = (255, 0, 0)  # Red
TEXT_COLOR = (255, 255, 255)  # White


def visualize_bbox(img, bbox, class_name, color=BOX_COLOR, thickness=5):
    """Visualizes a single bounding box on the image"""
    x_min, y_min, w, h = bbox
    x_min, x_max, y_min, y_max = int(x_min), int(
        x_min + w), int(y_min), int(y_min + h)

    cv2.rectangle(img, (x_min, y_min), (x_max, y_max),
                  color=color, thickness=thickness)

    ((text_width, text_height), _) = cv2.getTextSize(
        class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img, (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min), BOX_COLOR, -1)
    cv2.putText(
        img,
        text=class_name,
        org=(x_min, y_min - int(0.3 * text_height)),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.35,
        color=TEXT_COLOR,
        lineType=cv2.LINE_AA,
    )
    return img

def visualize(image, bboxes, category_ids, category_id_to_name, save_path):
    img = image.copy()
    for bbox, category_id in zip(bboxes, category_ids):
        class_name = category_id_to_name[category_id]
        img = visualize_bbox(img, bbox, class_name)
    cv2.imwrite(save_path, img)


transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Cutout(num_holes=25,p=0.1),     # FutureWarning: This class has been deprecated. Please use CoarseDropout # BUT CoarseDropout cannot be applied on bbox!!!
    A.ShiftScaleRotate(shift_limit=[-0.08,0.08],scale_limit=[-0.5,0.2],rotate_limit=[-20,20],border_mode=1,p=0.7),
    A.OneOf([    # add noise to image
            A.GaussNoise(),
            A.ISONoise(),
            ], p=0.2),
    A.OneOf([    # make image blurry and decrease the quality, and apply weather effect
            A.MotionBlur(),
            A.GlassBlur(iterations=1,max_delta=1),
            A.GaussianBlur(),
            A.ImageCompression(quality_lower=10, quality_upper=20),
            A.Downscale(scale_min=0.5,scale_max=0.5),
            A.RandomSnow(),
            A.RandomRain(blur_value=3),
            A.RandomSunFlare(src_radius=100),
            ], p=0.2),
    A.OneOf([    # affine and perspective
            A.PiecewiseAffine(p=0.3),
            A.Affine(mode=1),
            A.Perspective(),
            ], p=0.5),
    A.OneOf([    # change the color hist of image
            A.CLAHE(clip_limit=2),
            A.FancyPCA(),
            A.Equalize(),
            A.HueSaturationValue(),
            A.ColorJitter(),
            A.RandomBrightnessContrast(),
            ], p=0.3),
    A.OneOf([    # apply filters on image
            A.Sharpen(),
            A.Emboss(),
            ], p=0.4),
    A.OneOf([    # change the color channel
            A.ToSepia(),
            A.ToGray(),
            A.ChannelShuffle(),
            # A.Solarize(),
            # A.InvertImg(),
            # A.ChannelDropout(),
            ], p=0.2),
], p=0.5,
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

transform_test = A.Compose([
    A.RandomShadow(p=1),
],
    bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']),
)

def main(obj):
    image = cv2.imread('test.jpg')
    bboxes = [[15.66, 170.0, 177.09, 274.88], [109.7, 90.84, 190.8, 281.84]]
    category_ids = [18, 2]

    # We will use the mapping from category_id to the class name
    # to visualize the class label for the bounding box on the image
    category_id_to_name = {18: 'dog', 2: 'person'}
    # random.seed(7)
    transformed = transform_test(
        image=image, bboxes=bboxes, category_ids=category_ids)
    if obj == 'origin':
        save_path = 'origin.jpg'
        visualize(image, bboxes, category_ids, {
                  18: 'dog', 2: 'person'}, save_path)
    if obj == 'aug':
        save_path = 'aug.jpg'
        visualize(
            transformed['image'],
            transformed['bboxes'],
            transformed['category_ids'],
            category_id_to_name,
            save_path
        )

def show_stack_img():
    image = cv2.imread('test.jpg')
    bboxes = [[15.66, 170.0, 177.09, 274.88], [209.7, 90.84, 90.8, 281.84]]
    category_ids = [18, 2]
    category_id_to_name = {18: 'dog', 2: 'person'}
    # random.seed(7)
    def visualize(image, bboxes, category_ids, category_id_to_name, save_path,index,background):
        img = image.copy()
        height,width = img.shape[0],img.shape[1]
        for bbox, category_id in zip(bboxes, category_ids):
            class_name = category_id_to_name[category_id]
            img = visualize_bbox(img, bbox, class_name)
        if index==0:
            background[0:height,0:width,:] = img
        if index==1:
            background[0:height,width:width*2,:] = img
        if index==2:
            background[0:height,width*2:width*3,:] = img
        if index==3:
            background[0:height,width*3:width*4,:] = img
        if index==4:
            background[0:height,width*4:width*5,:] = img
        if index==5:
            background[height:height*2,0:width,:] = img
        if index==6:
            background[height:height*2,width:width*2,:] = img
        if index==7:
            background[height:height*2,width*2:width*3,:] = img
        if index==8:
            background[height:height*2,width*3:width*4,:] = img
        if index==9:
            background[height:height*2,width*4:width*5,:] = img
            cv2.imwrite(save_path,background)

    for i in range(20):
        print(i)
        background = np.zeros((image.shape[0]*2,image.shape[1]*5,3))
        for j in range(10):
            transformed = transform(image=image, bboxes=bboxes, category_ids=category_ids)
            save_path = str(i)+'.jpg'

            visualize(
                    transformed['image'],
                    transformed['bboxes'],
                    transformed['category_ids'],
                    category_id_to_name,
                    save_path,
                    j,
                    background,
                )


if __name__ == '__main__':
    obj = 'origin'  # obj = 'origin'
    show_stack_img()
    # main('origin')
