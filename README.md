# Deep Learning for Instance Segmentation of Agricultural Fields

This project delineates agricultural field parcels from satellite images using a deep learning instance segmentation approach. The methodology involves training a fully convolutional instance segmentation architecture (adapted from Li et al., 2016) on Sentinel-2 image data and corresponding agricultural field polygons from Denmark. The model aims to accurately predict field boundaries and, in a secondary experiment, crop types.

## Abstract
This thesis aims to delineate agricultural field parcels from satellite images via deep learning instance segmentation. Manual delineation is accurate but time-consuming, and many automated approaches with traditional image segmentation techniques struggle to capture the variety of possible field appearances. Deep learning has proven successful in various computer vision tasks, and might be a good candidate to enable accurate, performant, and generalizable delineation of agricultural fields. Here, a fully convolutional instance segmentation architecture (adapted from Li et al., 2016) was trained on Sentinel-2 image data and corresponding agricultural field polygons from Denmark. In contrast to many other approaches, the model operates on raw RGB images without significant pre- and post-processing. After training, the model proved successful in predicting field boundaries on held-out image chips. The results generalize across different field sizes, shapes, and other properties, but show characteristic problems in some cases. In a second experiment, the model was trained to simultaneously predict the crop type of the field instance. Performance in this setting was significantly worse. Many fields were correctly delineated, but the wrong crop class was predicted. Overall, the results are promising and prove the validity of the deep learning approach. Also, the methodology offers many directions for future improvement.

## Results
The model demonstrated success in predicting field boundaries across various field sizes and shapes, although some characteristic problems were noted. When tasked with predicting crop types alongside field boundaries, the model's performance decreased, with many fields being correctly delineated but incorrectly classified.

## Instructions

### 1. Installation of FCIS & MXNet
Install the FCIS model and MXNet framework using the instructions provided in the FCIS repository. The setup is optimized for an AWS EC2 P2 instance using the official AWS Deep Learning AMI (Ubuntu).

1. Clone the FCIS repository:
    ```bash
    git clone https://github.com/msracver/FCIS.git
    cd FCIS
    ```

2. Follow the installation instructions for MXNet and FCIS in the repository's README.

3. Verify the installation by running the FCIS demo:
    ```bash
    python FCIS/fcis/demo.py
    ```

### 2. Data Preprocessing
Prepare the Denmark LPIS field data and create the image chips and COCO format annotations.

1. Run the code in the Preprocessing Jupyter notebook.
2. Move the preprocessed vector folder to `.output/preprocessing/annotations`.
3. Move the image folder to `.output/preprocessing/images`.

### 3. Configuration
Configure the model for training and evaluation.

1. Place the configuration file `resnet_v1_101_coco_fcis_end2end_ohem.yaml` in `.FCIS/experiments/fcis/cfgs`.
2. Delete the annotations cache:
    ```bash
    rm -rf .FCIS/data/coco/annotations_cache/; rm -rf .FCIS/data/cache/COCOMask/
    ```

### 4. Model Evaluation
Evaluate the pre-trained model.

1. Move the folder containing the model to `FCIS/output/fcis/coco/resnet_v1_101_coco_fcis_end2end_ohem`.
2. Run the evaluation:
    ```bash
    python experiments/fcis/fcis_end2end_test.py --cfg experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml --ignore_cache
    ```

The resulting instance segmentation and object detection proposals will be saved to `FCIS/output/fcis/coco/resnet_v1_101_coco_fcis_end2end_ohem/val2016/detections_val2016_results.json`.

### 5. Custom Model Training
Train a custom model with new configurations or datasets.

1. Adjust the `PIXEL_MEANS` values in the configuration file to the RGB channels means of your dataset. The band means are saved to `.output/preprocessed/statistics.json` during preprocessing.
2. Delete existing model files:
    ```bash
    rm -rf /home/ubuntu/FCIS/output/fcis/coco/resnet_v1_101_coco_fcis_end2end_ohem/
    ```
3. Run the training task:
    ```bash
    python experiments/fcis/fcis_end2end_train_test.py --cfg experiments/fcis/cfgs/resnet_v1_101_coco_fcis_end2end_ohem.yaml
    ```

## Future Directions
The methodology offers several directions for future improvement, including:

- Enhancing the model architecture for better crop type classification.
- Incorporating additional data types (e.g., multi-spectral images) to improve segmentation accuracy.
- Implementing advanced pre- and post-processing techniques to refine results.

## References
- Li, Y., Qi, H., Dai, J., Ji, X., & Wei, Y. (2016). "Fully Convolutional Instance-aware Semantic Segmentation." arXiv preprint arXiv:1611.07709.
