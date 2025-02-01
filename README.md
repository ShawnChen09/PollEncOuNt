# PollEncOuNt <img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/Peon.png" width="40"/>

The purpose of this project is to count the number of **alive** and **dead** pollens in an image.

# Contents

- [Installation](#installation)
  
- [Outputs Demonstration](#outputs-demonstration)

- [Usage](#usage)

   - [GUI](#gui)

   - [Python API](#python-api)

# Installation

1. Package installs
```sh
pip install PollEncOuNt
```

# Outputs Demonstration

| **Raw**                                      | **Predicted**                              |
|---------------------------------------------|-------------------------------------------|
| <img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/pollen_raw.jpg" width="300"/>  | <img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/pollen_predicted.jpg" width="300"/> |

*blue squares detected are alive, green squares detected ard dead.*

`count_result.csv` summarizes the results after running the prediction process.

| Filename               | Alive | Dead |
|------------------------|-------|------|
| pollen_predicted.jpg   | 96    | 20   |

# Usage

## GUI

### Launching the GUI
Open your terminal (Mac/Linux) or command line (Windows).

Launch the GUI by typing:
```bash
% peon
```

The **PEON Main Menu** will appear.

<img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/main_menu.png" width="300"/>

### Main Menu

From the Actions menu, you can:

1. Select **Train** to open the Train GUI.

2. Select **Predict** to open the Predict GUI.

3. **Exit** the application.

<img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/menu_action.png" width="300"/>

### Predict GUI

#### A [Pre-trained Model](https://github.com/ShawnChen09/PollEncOuNt/raw/main/model/best.pt) can directly be used for prediction.

1. Open the Predict GUI:
 - From the main menu, go to `Actions > Predict`.

2. Configure Prediction Settings:
 - **Image Files**: Click the **Browse** button and select one or more image files for prediction.
 - **Model File**: Click the **Browse** button to load the trained model file (`.pt`, `.pth`, or `.onnx`).
 - **Project Save Directory**: Click the **Browse** button to select the directory where the prediction results will be saved.
 - **Pridect Settings**:
   - Check the **Save Images** box to save the images with predictions.
   - Check the **Save CSV** box to save the results in a `count_result.csv` file.

3. Start Prediction:
 - Click the **Start Prediction** button to begin inference.
 - Real-time log updates will appear in the **LOGS** section.

4. Reset Prediction:
 - Click the **Reset** button to clear inputs and logs.

<img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/predict_gui.png" width="400"/>

### Train GUI

1. Open the Train GUI:

 - From the main menu, go to `Actions > Train`.

2. Configure Training Settings:
 - **Data YAML File**: Click the **Browse** button and select a valid YAML file for training data.
 - **Project Save Directory**: Click the **Browse** button and choose the directory where training outputs (e.g., logs, model checkpoints) will be saved.
 - **Model Selection**:
   - For Pre-trained Models: Click the **Browse** button to select a `.pt`, `.pth`, or `.onnx` model file.
   - For YOLOv8 Models: Use the dropdown menu to choose one of the YOLO models (e.g., `yolov8n.pt`).
- **Training Settings**:
   - Set the number of **epochs** for training.
   - Select the device (`cpu` or `gpu`) from the dropdown menu.

3. Start Training:
 - Click the **Start Training** button to begin.
 - Monitor the real-time log updates in the **LOGS** section.

4. Reset Training:
 - To reset the form, click the **Reset** button. This clears all input fields and logs.

<img src="https://github.com/ShawnChen09/PollEncOuNt/raw/main/example/train_gui.png" width="400"/>

## Python API

### Predict

#### A pre-trained model can directly be used for prediction(path: `./model/best.pt`).

```python
from peon import peon_predict

results = peon_predict(img_files=["IMG1", "IMG2", ...],
                       model_path: "MODEL_PATH",
                       save_dir: "SAVE_DIR,
                       save_img = True,
                       save_csv = True,)
```

`img_files (list[str])`: List of image path to conduct prediction.

`model_path (str)`: Path to the YOLO model.

`save_dir (str, optional)`: Path to the directory to save the results. Required if `save_img` or `save_csv` is `True`.

`save_img (bool, optional)`: Whether to save the predicted images. Defaults to `True`.

`save_csv (bool, optional)`: Whether to save the results to a CSV file. Defaults to `True`.


### Train

```python
from peon import peon_train

peon_train(data_path="DATA_PATH",
           save_dir="SAVE_DIR",
           model_path = "yolov8m.pt",
           epochs = 100,
           device = "cpu",)
```

`data_path (str)`: Path to the data .YAML file.

`save_dir (str)`: Path to the directory to save the trained model.

`model_path (str, optional)`: Path to the YOLO model. Defaults to `"yolov8m.pt"`.

`epochs (int, optional)`: Number of epochs for training. Defaults to `100`.

`device (str, optional)`: Device to use for training. Defaults to `"cpu"`.