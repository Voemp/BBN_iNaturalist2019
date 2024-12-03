import json, os
import tqdm


def convert(type):
    if (type != "train") and (type != "val"):
        print("Error: This part of the dataset does not exist.")
        return

    load_path = "data/iNat/{}2019.json".format(type)
    root_path = "data/iNat"
    save_path = "data/jsons"

    all_annos = json.load(open(load_path, 'r'))
    annos = all_annos['annotations']
    images = all_annos['images']
    new_annos = []

    print("Converting file {} ...".format(load_path))
    for anno, image in tqdm(zip(annos, images)):
        assert image["id"] == anno["id"]

        new_annos.append({"image_id": image["id"],
                          "im_height": image["height"],
                          "im_width": image["width"],
                          "category_id": anno["category_id"],
                          "fpath": os.path.join(root_path, image["file_name"])})
    num_classes = len(all_annos["categories"])

    converted_annos = {"annotations": new_annos, "num_classes": num_classes}
    save_path = os.path.join(save_path, "converted_" + os.path.split(load_path)[-1])
    print("Converted, Saveing converted file to {}".format(save_path))
    with open(save_path, "w") as f:
        json.dump(converted_annos, f)
