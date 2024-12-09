import csv
import json
import os

from tqdm import tqdm


def convert(json_file, csv_file, image_root):
    """
    转换 JSON 文件，结合图像路径和来自 CSV 的预测标注数据。
    :param json_file: 输入的 JSON 文件路径 (test2019.json)
    :param csv_file: 输入的预测 CSV 文件路径 (包含 id 和预测标注的 CSV 文件)
    :param image_root: 图像的根目录路径
    :return: 转换后的标注数据
    """
    # 读取 test2019.json 文件
    all_annos = json.load(open(json_file, 'r'))
    images = all_annos['images']  # 提取图像数据

    # 读取 CSV 文件中的 id 和 predict
    id_to_category = {}
    with open(csv_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader)  # 跳过表头
        for row in reader:
            image_id = int(row[0])  # 假设 CSV 的第一列是 id
            predict = int(row[1])  # 假设 CSV 的第二列是预测类别
            id_to_category[image_id] = predict

    new_annos = []  # 用于存储新的标注数据

    print(f"Converting file {json_file} using predictions from {csv_file} ...")

    # 遍历图像数据并为每个图像分配预测的类别
    for image in tqdm(images):
        image_id = image["id"]
        # 获取对应的预测类别
        category_id = id_to_category.get(image_id, -1)  # 默认 -1 如果没有找到对应的 id

        new_annos.append({
            "image_id": image["id"],  # 图像 ID
            "im_height": image["height"],  # 图像高度
            "im_width": image["width"],  # 图像宽度
            "category_id": category_id,  # 类别 ID
            "fpath": os.path.join(image_root, image["file_name"])  # 图像文件路径
        })

    num_classes = len(set(id_to_category.values()))  # 获取类别数目（去重后的类别数量）

    return {"annotations": new_annos,  # 返回转换后的数据
            "num_classes": num_classes}


def main():
    # 输入的 test2019.json 文件路径
    json_file = "F:\\iNaturalist\\inaturalist-2019-fgvc6\\test2019.json"

    # 图像根目录路径 (请根据实际路径修改)
    image_root = "F:\\iNaturalist\\inaturalist-2019-fgvc6"

    # 输入的预测 CSV 文件路径 (假设 csv 文件位于当前路径)
    csv_file = "F:\\iNaturalist\\inaturalist-2019-fgvc6\\kaggle_sample_submission.csv"  # 请修改为实际的 CSV 文件路径

    # 保存转换后的文件路径
    save_path = "F:\\BBN_iNaturalist2019\\data\\jsons\\converted_test2019.json"

    # 转换 JSON 文件
    converted_annos = convert(json_file, csv_file, image_root)

    # 保存转换后的数据
    print(f"Converted, Saving converted file to {save_path}")
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)


if __name__ == "__main__":
    main()
