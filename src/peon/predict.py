import os

import pandas as pd
import ultralytics


def _get_result_count(verbose_results):
    try:
        verbose_results = list(filter(lambda w: w != "", verbose_results.split(", ")))
        results = {v.split()[1]: int(v.split()[0]) for v in verbose_results}
        return results
    except ValueError:
        print("Detect nothing.")
        return


def peon_predict(
    img_files: list[str],
    model_path: str,
    save_dir: str,
    save_img: bool = True,
    save_csv: bool = True,
    conf: float = 0.0,
    iou: float = 0.5,
    imgsz: int | tuple = 1280,
    max_det: int = 500,
    **kwargs,
):
    """
    Predicts objects in images using a YOLO model and saves the results to a CSV file.

    Args:
        img_files (list[str]): List of image path to conduct prediction.
        model_path (str): Path to the YOLO model.
        save_dir (str): Path to the directory to save the results.
        save_img (bool, optional): Whether to save the predicted images. Defaults to True.
        save_csv (bool, optional): Whether to save the results to a CSV file. Defaults to True.
        conf (float, optional): Confidence threshold for filtering detections. Defaults to 0.0.
        iou (float, optional): Intersection over Union threshold for NMS. Defaults to 0.5.
        imgsz (int | tuple, optional): Image size for inference as int or (height, width). Defaults to 1280.
        max_det (int, optional): Maximum number of detections per image. Defaults to 500.
        **kwargs: Additional arguments passed directly to the YOLO model's predict method.

    Returns:
        pandas.DataFrame: DataFrame containing the results.
    """
    assert os.path.isfile(model_path), "model_path must be a valid file"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    print("Start Prediction")
    model = ultralytics.YOLO(model_path)

    df = pd.DataFrame()
    for file in img_files:
        print(f"\nPredicting Image: {file}")

        if save_img or save_csv:
            assert save_dir is not None and os.path.isdir(save_dir), (
                "save_dir must be a valid directory"
            )

        res = model.predict(
            source=file,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            project=save_dir,
            name="image",
            show=False,
            save=save_img,
            show_labels=False,
            exist_ok=True,
            **kwargs,
        )

        result_count = _get_result_count(res[0].verbose())
        if result_count:
            df_extended = pd.DataFrame(result_count, index=[file])
            df = pd.concat([df, df_extended], ignore_index=False)
        else:
            pass

    df = df.fillna(0).astype(int)
    if save_csv:
        save_path = os.path.join(save_dir, "count_result.csv")
        df.to_csv(save_path)
        print(f"\nCount results saved to: {save_path}\n")

    print("Prediction completed.")
    return df
