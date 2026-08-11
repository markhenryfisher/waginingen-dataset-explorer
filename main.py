import argparse
import math
from pathlib import Path
from matplotlib import pyplot as plt

from common.utils import show_random_image
from coco.utils import convert2coco
from detection.yolov3.utils.utils import load_classes
from detection.data_loader import FDFLoader

DATA_PATH = Path('S:/mhf/scratch/from_waginingen')
NEEDED_FOLDERS = ["fdf_images", "results"]
FISH_CLASSES_PATH = {
    "names_file": Path("detection") / "fish_classes.names",
}

def parse_args():
    parser = argparse.ArgumentParser(
        description='Fully Documented Fisheries dataset converter')
    parser.add_argument('--multi-class', action='store_true', help='convert multi-class dataset')
    args = parser.parse_args()
    return args

def check_dataset():
    # Check if all folders are correct
    if not all([(DATA_PATH / f).is_dir() for f in NEEDED_FOLDERS]):
        print(
            "Could not find all data folders! Did you extract both fdf_images.zip and results.zip in the data folder?")


def load_dataset():
    print("Loading dataset...")
    data_folder = DATA_PATH
    weights_folder = data_folder / "results/model_weights"
    dataset_folder = data_folder / "fdf_images"



    # Define hyperparams
    hyp = {
        "giou": 3.54,  # giou loss gain
        "cls": 37.4,  # cls loss gain
        "cls_pw": 1.0,  # cls BCELoss positive_weight
        "obj": 64.3,  # obj loss gain (*=img_size/320 if img_size != 320)
        "obj_pw": 1.0,  # obj BCELoss positive_weight
        "iou_t": 0.20,  # iou training threshold
        "lr0": 0.01,  # initial learning rate (SGD=5E-3, Adam=5E-4)
        "lrf": 0.0005,  # final learning rate (with cos scheduler)
        "momentum": 0.937,  # SGD momentum
        "weight_decay": 0.0005,  # optimizer weight decay
        "fl_gamma": 0.0,  # focal loss gamma (efficientDet default is gamma=1.5)
        "hsv_h": 0.0138,  # image HSV-Hue augmentation (fraction)
        "hsv_s": 0.678,  # image HSV-Saturation augmentation (fraction)
        "hsv_v": 0.36,  # image HSV-Value augmentation (fraction)
        "degrees": 1.98 * 0,  # image rotation (+/- deg)
        "translate": 0.05 * 0,  # image translation (+/- fraction)
        "scale": 0.05 * 0,  # image scale (+/- gain)
        "shear": 0.641 * 0,
    }

    # Define options
    opt = {
        "epochs": 100,  # Number of epochs. We did 800 epochs on complete dataset
        "batch_size": 8,
        "multi_scale": False,  # adjust (67%% - 150%%) img_size every 10 batches
        "img_size": [320, 640, 416],  # [min-train, max-train, test] image size
        "rect": False,  # rectangular training
        "resume": False,  # resume traning from last.pt
        "nosave": False,  # only save final epoch
        "notest": False,  # only test final epoch
        "evolve": False,  # evolve hyperparameters
        "bucket": "",  # gsutil bucket
        "cache_images": False,  # cache images for faster training
        "name": "",  # renames results.txt to results_name.txt if supplied
        "adam": False,  # use adam optimizer
        "single_cls": False,  # train as single-class dataset
        "freeze_layers": False,  # freeze non-output layers,
        "name": "FDF_training",
        "device": "",
    }

    # Check if files exists
    assert FISH_CLASSES_PATH["names_file"].is_file()
    assert dataset_folder.is_dir()

    classes = load_classes(FISH_CLASSES_PATH["names_file"])
    # Check and recalculate image size
    imgsz_min, imgsz_max, imgsz_test = opt["img_size"]

    # Image Sizes
    gs = 32  # (pixels) grid size
    assert math.fmod(imgsz_min, gs) == 0, "--img-size %g must be a %g-multiple" % (imgsz_min, gs)
    opt["multi_scale"] |= imgsz_min != imgsz_max  # multi if different (min, max)
    if opt["multi_scale"]:
        if imgsz_min == imgsz_max:
            imgsz_min //= 1.5
            imgsz_max //= 0.667
        grid_min, grid_max = imgsz_min // gs, imgsz_max // gs
        imgsz_min, imgsz_max = int(grid_min * gs), int(grid_max * gs)
    img_size = imgsz_max

    train_dataset = FDFLoader(
        dataset_folder,
        classes,
        subset_name="train",
        img_size=img_size,
        batch_size=opt["batch_size"],
        augment=True,
        hyp=hyp,
        rect=opt["rect"],
        cache_images=opt["cache_images"],
        single_cls=opt["single_cls"],
    )
    val_dataset = FDFLoader(
        dataset_folder,
        classes,
        subset_name="validation",
        img_size=imgsz_test,
        batch_size=min(opt["batch_size"], len(train_dataset)),
        hyp=hyp,
        rect=True,
        cache_images=opt["cache_images"],
        single_cls=opt["single_cls"],
    )
    test_dataset = FDFLoader(
        dataset_folder,
        classes,
        subset_name="test",
        img_size=imgsz_test,
        batch_size=min(opt["batch_size"], len(train_dataset)),
        hyp=hyp,
        rect=True,
        cache_images=opt["cache_images"],
        single_cls=opt["single_cls"],
    )

    return train_dataset, val_dataset, test_dataset




if __name__ == '__main__':
    args = parse_args()

    check_dataset()
    train_dataset, val_dataset, test_dataset = load_dataset()
    # Now, show a random training and testing image with annotations to make sure that we have loaded the dataset correctly:
    # f, axarr = plt.subplots(1, 2, figsize=(20,20))
    # show_random_image(train_dataset, axarr[0], load_classes(FISH_CLASSES_PATH["names_file"]))
    # show_random_image(test_dataset, axarr[1], load_classes(FISH_CLASSES_PATH["names_file"]))
    # plt.show()
    # create folder for COCO annotations
    data_folder = DATA_PATH
    dataset_folder = data_folder / "fdf_images"
    p = dataset_folder / "coco_annotations"
    p.mkdir(exist_ok=True)
    if args.multi_class:
        print("Multi-class training")
        fileroot = "multi_class_"
    else:
        print("Mono-class training")
        fileroot = "mono_class_"
    # convert annotations
    convert2coco(train_dataset, p / (fileroot + "debug.json"), debug=True)
    # convert2coco(train_dataset, p / (fileroot + "train.json"))
    # convert2coco(val_dataset, p / (fileroot + "val.json"))
    # convert2coco(test_dataset, p / (fileroot + "test.json"))
