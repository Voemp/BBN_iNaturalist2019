from ..utils import Registry
import torchvision.transforms as transforms


TRANSFORMS = Registry()


@TRANSFORMS.register("random_resized_crop")
def random_resized_crop(cfg, **kwargs):
    """
    随机裁剪并缩放图像到指定大小
    :param cfg: 配置对象，包含图像变换的参数
    :param kwargs: 其他可选参数（如输入图像的大小）
    :return: 一个随机裁剪并缩放到指定大小的图像变换操作
    """
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.RandomResizedCrop(
        size=size,
        scale=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.SCALE,
        ratio=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_RESIZED_CROP.RATIO,
    )



# @TRANSFORMS.register("random_crop")
# def random_crop(cfg, **kwargs):
#     size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
#     return transforms.RandomCrop(
#         size, padding=cfg.TRANSFORMS.PROCESS_DETAIL.RANDOM_CROP.PADDING
#     )


@TRANSFORMS.register("random_horizontal_flip")
def random_horizontal_flip(cfg, **kwargs):
    """
    随机水平翻转图像
    :return: 一个随机水平翻转的图像变换操作
    """
    return transforms.RandomHorizontalFlip(p=0.5)


@TRANSFORMS.register("shorter_resize_for_crop")
def shorter_resize_for_crop(cfg, **kwargs):
    """
    将图像按短边调整大小，用于之后的裁剪操作
    :return: 一个调整图像短边大小的图像变换操作
    """
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    assert size[0] == size[1], "this img-process only process square-image"
    return transforms.Resize(int(size[0] / 0.875))


# @TRANSFORMS.register("normal_resize")
# def normal_resize(cfg, **kwargs):
#     """
#     普通的图像尺寸调整
#     :return: 一个调整图像大小的变换操作
#     """
#     size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
#     return transforms.Resize(size)


@TRANSFORMS.register("center_crop")
def center_crop(cfg, **kwargs):
    """
    对图像进行中心裁剪
    :return: 一个中心裁剪的图像变换操作
    """
    size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
    return transforms.CenterCrop(size)

#
# @TRANSFORMS.register("ten_crop")
# def ten_crop(cfg, **kwargs):
#     """
#     对图像进行10次裁剪（包括四个角和中心点）
#     :return: 一个十次裁剪（TenCrop）的图像变换操作
#     """
#     size = kwargs["input_size"] if kwargs["input_size"] != None else cfg.INPUT_SIZE
#     return transforms.TenCrop(size)
#
#
# @TRANSFORMS.register("normalize")
# def normalize(cfg, **kwargs):
#     """
#     对图像进行归一化处理
#     :return: 一个归一化图像的变换操作
#     """
#     return transforms.Normalize(
#         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#     )