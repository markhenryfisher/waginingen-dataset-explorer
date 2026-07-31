from pathlib import Path
from detection.yolov3.utils.utils import load_classes

DATA_PATH = Path('S:/mhf/scratch/from_waginingen')
NEEDED_FOLDERS = ["fdf_images", "results"]

def check_dataset():
    # Check if all folders are correct
    if not all([(DATA_PATH / f).is_dir() for f in NEEDED_FOLDERS]):
        print(
            "Could not find all data folders! Did you extract both fdf_images.zip and results.zip in the data folder?")


def load_dataset():
    data_folder = DATA_PATH
    weights_folder = data_folder / "results/model_weights"
    dataset_folder = data_folder / "fdf_images"

    # Define files
    paths = {
        "names_file": Path("detection") / "fish_classes.names",
    }

    # Check if files exists
    assert paths["names_file"].is_file()
    assert dataset_folder.is_dir()
    print("Loading dataset...")



if __name__ == '__main__':
    check_dataset()
    load_dataset()
