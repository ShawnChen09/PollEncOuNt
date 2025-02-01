import os
import re

import ultralytics

def peon_train(data_path,
               save_dir,
               model_path = "yolov8m.pt",
               epochs = 100,
               device = "cpu",):
    '''
    Trains a YOLO model using the specified parameters.

    Args:
        data_path (str): Path to the data .YAML file.
        save_dir (str): Path to the directory to save the trained model.
        model_path (str, optional): Path to the YOLO model. Defaults to "yolov8m.pt".
        epochs (int, optional): Number of epochs for training. Defaults to 100.
        device (str, optional): Device to use for training. Defaults to "cpu".
    '''
    yolo_models = [
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolov9t.pt", "yolov9s.pt", "yolov9m.pt", "yolov9c.pt", "yolov9e.pt",
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    ]
    try:
        assert model_path in yolo_models or os.path.isfile(model_path), \
            'model_path must be a valid YOLO model or a valid file'
        assert os.path.isfile(data_path), 'data_path must be a valid file'
        assert epochs > 0, 'epochs must be greater than 0'

        data_path = os.path.abspath(data_path)
        os.makedirs(save_dir, exist_ok=True)

        print("Training started.\n")
        model = ultralytics.YOLO(model_path)
        model.train(data=data_path,
                    epochs=epochs,
                    device=device,
                    project=save_dir)
        # model.export(format="onnx")
        print("\nTraining completed.")
    finally:
        _cleanup_yolo_model_files(model_path, yolo_models)

def _cleanup_yolo_model_files(model_path, yolo_models):
    """
    Removes all files matching the model_path pattern in the current directory.
    
    Args:
        model_path (str): Base name of the YOLO model files to be removed
    """
    if model_path not in yolo_models:
        return
    try:
        cwd = os.getcwd()
        pattern = f"{model_path}*"
        matching_files = [f for f in os.listdir(cwd) if re.match(pattern, f)]
        
        for file in matching_files:
            file_path = os.path.join(cwd, file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Failed to remove file {file}: {e}")
                
    except Exception as e:
        print(f"Error during cleanup of YOLO model files: {e}")