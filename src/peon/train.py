import ultralytics
import os

def pl_train(data_path, save_dir, model_path = "yolov8m.pt", epochs = 100, device = "cpu"):
    '''
    Trains a YOLO model using the specified parameters.

    Args:
        data_path (str): Path to the data .YAML file.
        save_dir (str): Path to the directory to save the trained model.
        model_path (str, optional): Path to the YOLO model. Defaults to "yolov8m.pt".
        epochs (int, optional): Number of epochs for training. Defaults to 100.
        device (str, optional): Device to use for training. Defaults to "cpu".
    '''
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