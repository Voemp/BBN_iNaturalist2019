import json
import random
from ctypes.wintypes import RGB

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


class BaseSet(Dataset):
    def __init__(self, json_file, mode="train", transform=None):
        """
        初始化数据集
        :param json_file: 数据集的 JSON 文件路径
        :param mode: 当前模式（"train" 或 "valid"）
        :param cfg: 配置文件，用于读取参数如颜色空间、双重采样和类别权重等
        :param transform: 数据增强和预处理操作
        """
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.annotations = data['annotations']
        self.num_classes = data['num_classes']
        self.mode = mode
        self.transform = transform
        self.input_size = (100, 100)  # 从配置中获取输入大小
        self.color_space = RGB  # 颜色空间设置（RGB/BGR等）
        self.dual_sample = True if mode == "train" else False  # 双重采样设置
        self.class_weight, self.sum_weight = self.get_weight(self.annotations, self.num_classes)  # 更新类别权重

        self.update_transform(input_size=self.input_size)  # 调用 update_transform 方法来设置数据预处理和增强操作

    def __getitem__(self, idx):
        anno = self.annotations[idx]  # 获取当前图像的注释信息
        img = self._get_image(anno)  # 加载图像
        meta = dict()  # 存储附加元数据
        image = self.transform(img)  # 对图像进行预处理（如数据增强）
        image_label = (
            anno["category_id"] if "test" not in self.mode else 0
        )  # 如果是测试模式，标签为 0；否则使用图像的类别 ID
        if self.mode not in ["train", "valid"]:
            meta["image_id"] = anno["image_id"]
            meta["fpath"] = anno["fpath"]
        return image, image_label, meta

    def update_transform(self, input_size=None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化操作
        transform_list = []  # 将图像转换为 PIL 格式

        # 选择训练或测试时的转换操作
        if self.mode == "train":
            transform_list.append(
                transforms.RandomResizedCrop(
                    size=input_size,  # 使用硬编码的输入尺寸
                    scale=(0.08, 1.0),  # 固定 scale 范围
                    ratio=(3. / 4., 4. / 3.)  # 固定 ratio 范围
                )
            )
            transform_list.append(
                transforms.RandomHorizontalFlip(p=0.5)
            )
        else:
            transform_list.append(
                transforms.Resize(int(input_size[0] / 0.875))
            )
            transform_list.append(
                transforms.CenterCrop(input_size)
            )

        transform_list.extend([transforms.ToTensor(), normalize])  # 转换为 Tensor 并进行标准化
        self.transform = transforms.Compose(transform_list)  # 构建完整的转换链

    def get_num_classes(self):
        return self.num_classes  # 返回类别数量

    def get_annotations(self):
        return self.data  # 返回图像的注释信息

    def __len__(self):
        return len(self.annotations)

    def _get_image(self, anno):
        img = Image.open(anno['fpath']).convert('RGB')
        return img

    def _get_class_dict(self):
        class_dict = dict()  # 存储每个类别的索引
        for i, anno in enumerate(self.annotations):
            cat_id = (
                anno["category_id"] if "category_id" in anno else anno["image_label"]
            )  # 获取类别 ID
            if not cat_id in class_dict:
                class_dict[cat_id] = []
            class_dict[cat_id].append(i)  # 将索引加入对应类别的列表中
        return class_dict

    def get_weight(self, annotations, num_classes):
        """
        根据样本的类别计算每个类别的权重（解决类别不平衡问题）
        :param annotations: 数据集的注释
        :param num_classes: 类别总数
        :return: 类别权重和权重总和
        """
        num_list = [0] * num_classes  # 初始化类别计数
        for anno in annotations:
            category_id = anno["category_id"]
            num_list[category_id] += 1  # 统计每个类别的样本数

        max_num = max(num_list)  # 获取样本数最多的类别
        class_weight = [max_num / num for num in num_list]  # 计算每个类别的权重
        sum_weight = sum(class_weight)  # 所有类别权重的总和

        return class_weight, sum_weight

    def sample_class_index_by_weight(self):
        """
        按照类别的权重进行采样，确保类别较少的样本有更高的概率被选中
        :return: 一个随机采样的类别索引
        """
        rand_number = random.random() * self.sum_weight
        now_sum = 0
        for i in range(self.num_classes):
            now_sum += self.class_weight[i]
            if rand_number <= now_sum:
                return i  # 返回选中的类别


class INatDataset(BaseSet):
    def __init__(self, json_file, mode='train', transform=None):
        super().__init__(json_file, mode, transform)
        random.seed(0)
        if self.dual_sample:
            self.class_weight, self.sum_weight = self.get_weight(self.annotations, self.num_classes)
            self.class_dict = self._get_class_dict()

    def __getitem__(self, index):
        anno = self.annotations[index]
        img = self._get_image(anno)
        image = self.transform(img)

        meta = dict()
        if self.dual_sample:
            sample_class = self.sample_class_index_by_weight()

            sample_indexes = self.class_dict[sample_class]
            sample_index = random.choice(sample_indexes)
            sample_info = self.annotations[sample_index]
            sample_img, sample_label = self._get_image(sample_info), sample_info['category_id']
            sample_img = self.transform(sample_img)
            meta['sample_image'] = sample_img
            meta['sample_label'] = sample_label

        if self.mode != 'test':
            image_label = anno['category_id']  # 0-index

        return image, image_label, meta
