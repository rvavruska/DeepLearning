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
    split='train',
    label_types=["detections"],
    classes=["Car"],
    max_samples=1000,
    shuffle=True,
)

session = fo.launch_app(dataset)


dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split='validation',
    label_types=["detections"],
    classes=["Car"],
    max_samples=300,
    shuffle=True,
)

session = fo.launch_app(dataset)


dataset = foz.load_zoo_dataset(
    "open-images-v6",
    split='test',
    label_types=["detections"],
    classes=["Car"],
    max_samples=100,
    shuffle=True,
)

session = fo.launch_app(dataset)

session.wait()