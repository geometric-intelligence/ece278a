from kartezio.apps.instance_segmentation import create_instance_segmentation_model
from kartezio.dataset import read_dataset
from kartezio.endpoint import (
    EndpointEllipse,
    EndpointHoughCircle,
    EndpointLabels,
    EndpointWatershed,
    LocalMaxWatershed,
)
from kartezio.preprocessing import TransformToHED, TransformToHSV
from kartezio.training import train_model
from numena.io.drive import Directory

RUNS = 10
ITERATIONS = 20000
LAMBDA = 5
COLORS_SCALES = ["RGB", "HSV", "HED"]


if __name__ == "__main__":
    output = './results'
    dataset_path = './dataset/breast'
    run_number = 10
    endpoint_name = 'MCW'

    color_scale_index = (run_number - 1) // RUNS
    color_scale = COLORS_SCALES[color_scale_index]
    output_directory = Directory(output).next(endpoint_name).next(color_scale)

    outputs = 1
    if endpoint_name == "MCW":
        outputs = 2
        endpoint = EndpointWatershed()
    elif endpoint_name == "LMW":
        endpoint = LocalMaxWatershed()
    elif endpoint_name == "ELLIPSE":
        endpoint = EndpointEllipse(min_axis=10, max_axis=65)
    elif endpoint_name == "HCT":
        endpoint = EndpointHoughCircle(
            min_dist=15, p1=32, p2=16, min_radius=5, max_radius=32
        )
    elif endpoint_name == "LABELS":
        endpoint = EndpointLabels()

    preprocessing = None
    if color_scale == "HSV":
        preprocessing = TransformToHSV()
    elif color_scale == "HED":
        preprocessing = TransformToHED()

    model = create_instance_segmentation_model(
        ITERATIONS,
        LAMBDA,
        inputs=3,
        outputs=outputs,
        endpoint=endpoint,
    )
    dataset = read_dataset(dataset_path)
    elite, _ = train_model(
        model, dataset, str(output_directory._path), preprocessing=preprocessing
    )
