import json, os  # 导入json库用于读取和保存JSON文件，os库用于处理文件路径
import tqdm  # 导入tqdm库用于显示进度条


def convert(type):
    """
    将 iNat 数据集（train/val）从原始格式转换为所需格式
    :param type: 要转换的数据类型，"train" 或 "val"
    """
    # 检查输入类型是否为 "train" 或 "val"，否则输出错误信息
    if (type != "train") and (type != "val"):
        print("Error: This part of the dataset does not exist.")
        return  # 如果输入类型错误，退出函数

    # 构建原始数据的路径
    load_path = "data/iNat/{}2019.json".format(type)  # 输入文件路径
    root_path = "data/iNat"  # 数据根路径
    save_path = "data/jsons"  # 保存文件路径

    # 打开并加载JSON数据
    all_annos = json.load(open(load_path, 'r'))  # 读取JSON文件，包含annotations和images
    annos = all_annos['annotations']  # 提取annotations（标注信息）
    images = all_annos['images']  # 提取images（图像信息）
    new_annos = []  # 用于存储转换后的标注信息

    print("Converting file {} ...".format(load_path))  # 打印转换开始的信息

    # 使用tqdm库的zip函数，并行遍历annotations和images列表
    for anno, image in tqdm.zip(annos, images):
        assert image["id"] == anno["id"]  # 确保每个annotation与image的id匹配

        # 将每个annotation转换为目标格式
        new_annos.append({
            "image_id": image["id"],  # 图像ID
            "im_height": image["height"],  # 图像高度
            "im_width": image["width"],  # 图像宽度
            "category_id": anno["category_id"],  # 类别ID
            "fpath": os.path.join(root_path, image["file_name"])  # 图像文件路径
        })

    # 获取类别数目
    num_classes = len(all_annos["categories"])  # 获取类别数量

    # 构造最终的转换结果字典
    converted_annos = {
        "annotations": new_annos,  # 转换后的标注信息
        "num_classes": num_classes  # 类别数目
    }

    # 构建保存路径并保存转换后的数据
    save_path = os.path.join(save_path, "converted_" + os.path.split(load_path)[-1])  # 生成新的保存路径
    print("Converted, Saving converted file to {}".format(save_path))  # 打印保存信息

    # 将转换后的数据保存为JSON文件
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)  # 将数据写入JSON文件
