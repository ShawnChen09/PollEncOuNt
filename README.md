# pollENcount

The purpose of this project is to count the number of **alive** and **dead** pollens in an image.

# Installation
1. Required Python Package Installation
```sh
pip install -r requirements.txt
```
2. Local package installs
```sh
pip install -e .
```

# Usage

### A pre-trained model can directly be used for prediction(path: `./model/best.pt`).
### A pre-writed data.yaml can directly be used for training (path: `./data.yaml`).

## Example

### Predict

```python
from plc import pl_predict

results = pl_predict(model_path='MODEL_PATH',
            img_dir='IMG_DIR_PATH',
            suffix='.jpg',
            save_img=True,
            save_csv=True,
            save_dir='SAVE_DIR_PATH',
            csv_name='results.csv')
```

`model_path (str)`: Path to the YOLO model.

`img_dir (str)`: Path to the directory containing images.

`suffix (str, optional)`: File suffix for images. Defaults to `.jpg`.

`save_img (bool, optional)`: Whether to save the predicted images. Defaults to `True`.

`save_csv (bool, optional)`: Whether to save the results to a CSV file. Defaults to `True`.

`save_dir (str, optional)`: Path to the directory to save the results. Required if `save_img` or `save_csv` is `True`.

`csv_name (str, optional)`: Name of the CSV file to save the results. Defaults to `results.csv`.


### Train
```python
from plc import pl_train

pl_train(data_path="DATA_PATH",
         save_dir="SAVE_DIR",
         model_path = "yolov8m.pt",
         mode = "detect",
         epochs = 100,
         device = "cpu")
```

`data_path (str)`: Path to the data .YAML file.

`save_dir (str)`: Path to the directory to save the trained model.

`model_path (str, optional)`: Path to the YOLO model. Defaults to `"yolov8m.pt"`.

`mode (str, optional)`: Mode for training. Defaults to `"detect"`.

`epochs (int, optional)`: Number of epochs for training. Defaults to `100`.

`device (str, optional)`: Device to use for training. Defaults to `"cpu"`.