import fiftyone as fo
import fiftyone.zoo as foz

#
# Load 50 random samples from the validation split
#
# Only the required images will be downloaded (if necessary).
# By default, all label types are loaded
#

dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split="validation",
    label_types=["detections", "classifications"],
    classes=["Fedora", "Piano"],
    max_samples=50,
    shuffle=True,
)