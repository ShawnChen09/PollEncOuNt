import ultralytics
import os
import pandas as pd

def _get_result_count(verbose_results):
    results = {}
    verbose_results = list(filter(lambda w: w != "", verbose_results.strip().split()))
    for key, value in zip(map(lambda k: k.replace(",", ""), verbose_results[1::2]), map(lambda r: int(r), verbose_results[0::2])):
        key = key[:-1] if value > 1 else key # when more than 1 object detected, a 's' character is added at the end
        results[key] = int(value)

    return results

def pl_predict(model_path: str,
            img_dir: str,
            suffix: str = '.jpg',
            save_img: bool = True,
            save_csv: bool = True,
            save_dir: str = None,
            csv_name: str = 'results.csv'):
    """
    Predicts objects in images using a YOLO model and saves the results to a CSV file.

    Args:
        model_path (str): Path to the YOLO model.
        img_dir (str): Path to the directory containing images.
        suffix (str, optional): File suffix for images. Defaults to '.jpg'.
        save_img (bool, optional): Whether to save the predicted images. Defaults to True.
        save_csv (bool, optional): Whether to save the results to a CSV file. Defaults to True.
        save_dir (str, optional): Path to the directory to save the results. Required if save_csv is True.
        csv_name (str, optional): Name of the CSV file to save the results. Defaults to 'results.csv'.

    Returns:
        pandas.DataFrame: DataFrame containing the results.
    """
    assert os.path.isfile(model_path), 'model_path must be a valid file'
    assert os.path.isdir(img_dir), 'img_dir must be a valid directory'
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    model = ultralytics.YOLO(model_path)

    files = os.listdir(img_dir)
    df = pd.DataFrame()
    for file in files:
        if file.endswith(suffix):
            path = os.path.join(img_dir, file)
        else:
            continue
        
        print(f'Predicting {file}...')

        if save_img:
            assert save_dir is not None and os.path.isdir(save_dir), 'save_dir must be a valid directory'

        res = model.predict(source=path, show=False, save=save_img, conf=0.2, imgsz=1280, show_labels=False, project=save_dir, name="image", exist_ok=True)
        result_count = _get_result_count(res[0].verbose())
        df_extended = pd.DataFrame(result_count, index=[file])
        df = pd.concat([df, df_extended], ignore_index=False)

    df = df.fillna(0).astype(int)
    if save_csv:
        assert save_dir is not None and os.path.isdir(save_dir), 'save_dir must be a valid directory'

        save_path = os.path.join(save_dir, csv_name)
        df.to_csv(save_path)
        print(f'Results saved to {save_path}')

    return df