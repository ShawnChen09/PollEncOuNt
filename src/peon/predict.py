import os

import pandas as pd
import ultralytics

def _get_result_count(verbose_results):
    verbose_results = list(filter(lambda w: w != "", verbose_results.split(", ")))
    results = {v.split()[1]: int(v.split()[0]) for v in verbose_results}
    return results

def peon_predict(img_files: list[str],
                 model_path: str,
                 save_dir: str,
                 save_img: bool = True,
                 save_csv: bool = True,):
    """
    Predicts objects in images using a YOLO model and saves the results to a CSV file.

    Args:
        img_files (list[str]): List of image path to conduct prediction.
        model_path (str): Path to the YOLO model.
        save_dir (str): Path to the directory to save the results.
        save_img (bool, optional): Whether to save the predicted images. Defaults to True.
        save_csv (bool, optional): Whether to save the results to a CSV file. Defaults to True.

    Returns:
        pandas.DataFrame: DataFrame containing the results.
    """
    assert os.path.isfile(model_path), 'model_path must be a valid file'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    print("Start Prediction")
    model = ultralytics.YOLO(model_path)

    df = pd.DataFrame()
    for file in img_files:
        print(f'\nPredicting Image: {file}')

        if save_img or save_csv:
            assert save_dir is not None and os.path.isdir(save_dir), 'save_dir must be a valid directory'

        res = model.predict(source=file,
                            show=False,
                            save=save_img,
                            conf=0.2,
                            imgsz=1280,
                            show_labels=False,
                            project=save_dir,
                            name="image",
                            exist_ok=True)

        result_count = _get_result_count(res[0].verbose())
        df_extended = pd.DataFrame(result_count, index=[file])
        df = pd.concat([df, df_extended], ignore_index=False)

    df = df.fillna(0).astype(int)
    if save_csv:
        save_path = os.path.join(save_dir, "count_result.csv")
        df.to_csv(save_path)
        print(f'\nCount results saved to: {save_path}\n')

    print("Prediction completed.")
    return df