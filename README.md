# yolov8_crowdhuman
Train crowdhuman dataset using ultralytics yolov8

## Prepare Dataset

1. Download the dataset from [CrowdHuman](https://www.crowdhuman.org/download.html)

2. Extract the dataset to the `raw` directory

3. The train folder is divided into three, but it is combined into one.

    ```bash
    ./yolov8_crowdhuman/raw$ tree . -L 2
    .
    ├── annotation_train.odgt
    ├── annotation_val.odgt
    ├── CrowdHuman_train
    │   ├── Images
    │   └── labels
    └── CrowdHuman_val
        ├── Images
        └── label
    ```

4. Run python script. vbox is the label format of the `person` class, and it is converted to yolo format.

    ```bash
    python prepare_dataset/get_anno.py
    ```
