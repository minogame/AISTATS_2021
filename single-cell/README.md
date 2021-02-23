# Tensor-Power Recurrent Models (TP-RNN)

### Paper

On the Memory Mechanism of Tensor-Power Recurrent Models

By Hejia Qiu, [Chao Li](chao.li@riken.jp), Ying Weng, Zhun Sun, Xingyu He, Qibin Zhao

### Requirements

- Python 3.6
- pytorch >= 1.6.0
- scipy >= 1.5.0
- sklearn >= 0.23.2
- pandas >= 0.24.2

### Usage



The file `new_train.py`  is used to execute a single run, the hyper-parameters can be specified as follow, which will train a `DORNN_HN1lCDx1Dh1Dp5` on the `arfima` dataset:
```
python3 new_train.py --dataset=arfima  --algorithm=DORNN_HN1lCDx1Dh1Dp3  --epochs=1000 --hidden_size=1 --input_size=1 --output_size=1 --train_size=2000 --validate_size=1200 --test_size=800 --seed=1234
```

If you want to carry out a serials of training with different seed, then run the `script1.py` with its seed parameter being specified as an iterable object, for example: 
`--seed="range(1,51,1)"` or `--seed="[123,456,789]"`
The results will be saved as a `.mat` file in the following format.
```
dataset : arfima
algorithm : DORNN_HN1lCDx1Dh1Dp5
epochs : 100
lr : 0.01
hidden_size : 1
input_size : 1
output_size : 1
train_size : 200
validate_size : 120
test_size : 80
look_back : 1
K : 100
patience : 100
final_train_loss : [0.00899 0.01003 0.01439 0.0087 ]
val_loss : [0.01342 0.01474 0.0212  0.0131 ]
test_rmse : [1.64249 1.95478 2.49251 1.63357]
test_mape : [ 74.62188  94.93149 120.77346  65.66157]
test_mae : [1.3328  1.61866 2.07411 1.31965]
seeds : [1, 2, 3, 4]
final_train_loss_mean_std_min : [0.0105275, 0.002284188860405374, 0.0087]
val_loss_mean_std_min : [0.015615, 0.0032825714005943574, 0.0131]
test_rmse_mean_std_min : [1.9308374999999998, 0.3491280165909776, 1.63357]
test_mape_mean_std_min : [88.99709999999999, 21.19049285617845, 65.66157]
test_mae_mean_std_min : [1.586305, 0.30592901059722993, 1.31965]
```

**Available datasets**

|Name|Identification|
|-|-|
|ARFIMA series| `arfima`|
|Dow Jones Industrial Average (DJI)| `DJI`|
|Metro interstate traffic volume| `traffic`|
|Tree ring| `tree7`|

**Available algorithms**

|Algorithm|Identification|
|-|-|
|vanilla RNN|`RNN`|
|vanilla LSTM|`LSTM`|
|Memory-augmented RNN with homogeneous d|`mRNN_fixD`|
|Memory-augmented RNN with dynamic d|`mRNN`|
|Memory-augmented LSTM with homogeneous d|`mLSTM_fixD`|
|Memory-augmented LSTM|`mLSTM`|
|Tensor-Power RNN of state size equaling 1 (N1) and no historic hidden state (Dh1) | `DORNN_HN1lCDx1Dh1Dp3`|




### Acknowledgement
- The code is modified based on the [Memory-augmented Recurrent Networks (mRNN-mLSTM)](https://github.com/huawei-noah/noah-research/tree/master/mRNN-mLSTM).
- The TRNN code is re-implentmented base on the [TensorRNN](https://github.com/yuqirose/TensorRNN).