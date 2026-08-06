import cv2
import json
from datetime import datetime


def today_yyyymmdd():
    # Get current date and format it as YYYYMMDD
    return datetime.now().strftime("%Y%m%d")

def convert2coco(dataset, coco_json_path):
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


    categories = dataset.class_names
    # Create category entries - COCO categories start at 1
    for idx, cat in enumerate(categories, start=1):
        coco["categories"].append({
            "id": idx,
            "name": cat
        })

    for i in range(len(dataset)):
        if len(coco["images"]) == 2:
            break
        img_id = i + 1 # img_ids start at 1
        # skip images without labels
        if len(list(dataset.labels)[i]) > 0:
            image = cv2.imread(dataset.img_files[i])
            width = image.shape[1]
            height = image.shape[0]

            # Add image entry
            data_img = dict({
                "id": img_id,
                "file_name": dataset.img_files[img_id],
                "width": width,
                "height": height
            })
            coco["images"].append(data_img)

            # add annotation entry
            for j, lbl in enumerate(dataset.labels[img_id]):
                ann_id = j + 1 # ann_ids start at 1
                # Convert normalized YOLO to COCO pixel coordinates
                x_center, y_center, w, h = lbl[1:]
                x = int( (x_center - w / 2) * width )
                y = int( (y_center - h / 2) * height )
                w_px = int( w * width )
                h_px = int( h * height )

                data_anno = dict({
                    "id": ann_id,
                    "img_id": img_id,
                    "category_id": int(lbl[0]) + 1, # COCO IDs start at 1
                    "bbox": [x, y, w_px, h_px],
                    "area": w_px * h_px,
                    "iscrowd": 0
                })
                coco["annotations"].append(data_anno)
        else:
            pass

    with coco_json_path.open("w") as f:
        json.dump(coco, f)

    return