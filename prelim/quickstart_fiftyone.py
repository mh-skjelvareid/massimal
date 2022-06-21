import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
session = fo.launch_app(dataset, port=5151)

# Blocks execution until the App is closed
session.wait()
