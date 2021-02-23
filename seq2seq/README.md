# Tensor-Power Recurrent Models (TP-RNN)

### Paper

On the Memory Mechanism of Tensor-Power Recurrent Models

By Hejia Qiu, [Chao Li](chao.li@riken.jp), Ying Weng, Zhun Sun, Xingyu He, Qibin Zhao

### Requirements

* tensorflow >= r1.6
* Python >=3.0
* numpy >= 1.16
* matplotlib >= 3.0



### Usage

The file ` test.py`  is used to execute a single run, the hyper-parameters can be specified in `test.py` and `train_config.py` . 

```python
python test.py > log.txt
```

The log file will include program process, hyper parameters and loss. 

```
Flags configuration loaded ...
loading time series ...
input type  <class 'numpy.ndarray'> (1000, 100, 1)
normalize to (0-1)
----------------------------------------------------------------------------------------------------
model TP_LSTM |dataset| ./demo.npy |input steps| 15 |out steps| 54 |hidden size| 8 |hidden layer| 2 |learning rate| 0.001 |decay rate| 0.8 |rank val| |initial order| 1.0 [2] |batch size| 25
----------------------------------------------------------------------------------------------------
Training -->
          Create TFC Encoder ...
          Create TFC Decoder ...
Testing -->
          Create TFC Encoder ...
          Create TFC Decoder ...
Step 1, Minibatch Loss= 0.3683
Validation Loss: 0.3670053
......
Step 10000, Minibatch Loss= 0.0049
Validation Loss: 0.0077703693
Optimization Finished!
Testing Loss: 0.008367555
Model saved in file: ./log/test/
```



**Available datasets**

| Name                                                   | Identification |
| ------------------------------------------------------ | -------------- |
| Genz dataset                                           | `Genz`         |
| The Los Angeles County highway network traffic dataset | `traffic`      |
| NREL synthetic solar photovoltaic power dataset        | `solar`        |

**Available algorithms**

| Algorithm         | Identification |
| ----------------- | -------------- |
| vanilla RNN       | `RNN`          |
| vanilla LSTM      | `LSTM`         |
| HOT LSTM          | `TLSTM`        |
| Tensor-Power LSTM | `TP_LSTM`      |

### Acknowledgement

- The code is modified based on the [TensorRNN](https://github.com/yuqirose/TensorRNN).

