import cv2
import json
from datetime import datetime
from tqdm import tqdm

def today_yyyymmdd():
    # Get current date and format it as YYYYMMDD
    return datetime.now().strftime("%Y%m%d")

def convert2coco(dataset, coco_json_path, debug=False):
    info = {
        "description": "Fully Documented Fisheries Dataset (converted to COCO Format)",
        "url": "https://data.4tu.nl/datasets/a6d5a40e-0358-47cf-9ec1-335df0e4a3c3",
        "version": "1.0",
        "year": 2026,
        "contributor": "Wageningen University and Research, Department of Plant Sciences",
        "date_created": today_yyyymmdd()
    }

    coco = {
        "info": info,
        "images": [],
        "annotations": [],
        "categories": []
    }

    if "multi_class" in str(coco_json_path):
        multi_class = True
        categories = dataset.class_names
    else:
        multi_class = False
        categories = ["fish_unknown"]
    # Create category entries - COCO categories start at 1
    for idx, cat in enumerate(categories, start=1):
        coco["categories"].append({
            "id": idx,
            "name": cat
        })

    img_id = 0
    ann_id = 0
    for i in tqdm(range(len(dataset)), desc="Converting to COCO", unit="items"):
        if debug and len(coco["images"]) == 2:
            break

        # skip images without labels
        if len(list(dataset.labels)[i]) > 0:
            img_id += 1

            image = cv2.imread(dataset.img_files[i])
            width = image.shape[1]
            height = image.shape[0]

            # Add image entry
            data_img = dict({
                "id": img_id,
                "file_name": dataset.img_files[i],
                "width": width,
                "height": height
            })
            coco["images"].append(data_img)

            # add annotation entry
            for lbl in dataset.labels[i]:
                ann_id += 1
                if multi_class:
                    cat_id = int(lbl[0]) + 1
                else:
                    cat_id = 1

                # Convert normalized YOLO to COCO pixel coordinates
                x_center, y_center, w, h = lbl[1:]
                x = int( (x_center - w / 2) * width )
                y = int( (y_center - h / 2) * height )
                w_px = int( w * width )
                h_px = int( h * height )

                data_anno = dict({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cat_id, # COCO IDs start at 1
                    "bbox": [x, y, w_px, h_px],
                    "area": w_px * h_px,
                    "iscrowd": 0
                })
                coco["annotations"].append(data_anno)

    with coco_json_path.open("w") as f:
        json.dump(coco, f)

    return