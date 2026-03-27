from BaguanTS import BaguanTS
import pandas as pd
import numpy as np
import yaml
import torch
import matplotlib.pyplot as plt


def linear_trend_toy():

    # build toy data
    start_date = '2022-12-03'
    end_date = '2023-01-17'
    dates = pd.date_range(start=start_date, end=end_date, freq='h')  # hour
    n_points = len(dates)


    trend = np.linspace(0, 100, n_points)  
    
    hour_of_day = dates.hour.values
    seasonal = 5 * np.sin(2 * np.pi * hour_of_day / 24)

    noise = np.random.normal(0, 1, n_points)

    ground_truth = trend + seasonal + noise

    
    split_idx = (dates >= '2023-01-15').argmax() 
    train_end = split_idx - 24  

    return dates, ground_truth[:train_end], ground_truth[train_end:]


def evaluate_BaguanTS(ckpt_path, model_cfg_path, hyper_path, device):
    model = BaguanTS(ckpt_path=ckpt_path, config_path=model_cfg_path, device=device)

    with open(hyper_path) as f:
        hyper_config = yaml.safe_load(f)

    # inference mode
    torch.manual_seed(42)
    max_sample = 10000
    mode = hyper_config.get("mode", "3D")

    # data 
    _, y_train, y_test = linear_trend_toy()
    running_idx = np.arange(len(y_train)+len(y_test))
    X_train, X_test = running_idx[:len(y_train)], running_idx[len(y_train):]
    if X_train.ndim == 1:
        X_train = X_train.reshape(-1, 1)
        X_test = X_test.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    # data shape
    # X_train: [context_seq_len, fea/cov_num] | X_test: [pred_seq_len, fea/cov_num]
    # Y_train: [context_seq_len, tar_num] | Y_test: [pred_seq_len, tar_num]

    if mode == '2D':
        # 2D inference
        X_train_2d = X_train[-max_sample:,:]
        y_train_2d = y_train[-max_sample:,:]
        X_test_2d = X_test
        forecast, forecast_q = model.predict(X_train_2d, 
                                            y_train_2d, 
                                            X_test_2d, 
                                            data_type='tabular', 
                                            mF=hyper_config.get("mF", 4),
                                            )
    else:
        # 3D inference
        X_train_3d = X_train
        y_train_3d = y_train
        X_test_3d = X_test
        forecast, forecast_q = model.predict(X_train_3d, y_train_3d, X_test_3d, 
                            data_type='TS-tabular',
                            context_len=hyper_config.get("context_len", 500),
                            K=hyper_config.get("K", 50),
                            period=hyper_config.get("RBfcst_period", 5), 
                            mF=hyper_config.get("mF", 4),
                            )


    
    return forecast, forecast_q


if __name__ == "__main__":
    ckpt_path = "PATH_TO_CKPT"
    model_cfg_path = "./configs/model_config.yml"
    hyper_path = "./configs/hyper_config.yml"
    device = "cuda:0"
    forecast, forecast_q = evaluate_BaguanTS(ckpt_path, model_cfg_path, hyper_path, device)
    print(forecast.shape)
    print(forecast_q.shape)





