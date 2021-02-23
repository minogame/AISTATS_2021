#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under
#the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from copy import deepcopy
import math, time, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import models, dornn_simple as dornn, roseyu_trnn as trnn
from scipy.io import loadmat
import logging

# initialization
PARSER = argparse.ArgumentParser()
PARSER.add_argument('--seed', type=int, default=1)
PARSER.add_argument('--dataset', type=str, default='longarfima')
PARSER.add_argument('--algorithm', type=str, default='mLSTM')
PARSER.add_argument('--epochs', type=int, default=1000, help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=0.01)
PARSER.add_argument('--hidden_size', type=int, default=5)
PARSER.add_argument('--input_size', type=int, default=5)
PARSER.add_argument('--output_size', type=int, default=5)
PARSER.add_argument('--train_size', type=int, default=2000)
PARSER.add_argument('--validate_size', type=int, default=2000)
PARSER.add_argument('--test_size', type=int, default=2000)
PARSER.add_argument('--look_back', type=int, default=1)
PARSER.add_argument('--K', type=int, default=100)
PARSER.add_argument('--patience', type=int, default=100)
PARSER.add_argument('--gang_mode', type=bool, default=False)
FLAGS = PARSER.parse_args()

    # if FLAGS.dataset == 'tree7':
    #     train_size = 2500
    #     validate_size = 1000
    # if FLAGS.dataset == 'DJI':
    #     train_size = 2500
    #     validate_size = 1500
    # if FLAGS.dataset == 'traffic':
    #     train_size = 1200
    #     validate_size = 200
    # if FLAGS.dataset == 'arfima':
    #     train_size = 2000
    #     validate_size = 1200

if FLAGS.gang_mode:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler('tons_of_logs/{}_{}.log'.format(FLAGS.algorithm, int(time.time()))),
        ]
    )
else:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler('tons_of_logs/{}_{}.log'.format(FLAGS.algorithm, int(time.time()))),
            logging.StreamHandler()
        ]
    )
    

for key in FLAGS.__dict__:
    logging.info('{} = {}'.format(key, FLAGS.__dict__[key]))

train_size, validate_size, test_size = FLAGS.train_size, FLAGS.validate_size, FLAGS.test_size
batch_size=1
look_back=FLAGS.look_back
scaler = MinMaxScaler(feature_range=(0, 1))
data = loadmat('data/time_series_prediction/{}.mat'.format(FLAGS.dataset))['x'].reshape(-1, FLAGS.input_size)
data = scaler.fit_transform(data)

starter_point = 0
train_x, train_y = data[starter_point:starter_point+train_size,:], data[starter_point+look_back:starter_point+train_size+look_back,:]
starter_point += train_size
validate_x, validate_y = data[starter_point:starter_point+validate_size,:], data[starter_point+look_back:starter_point+validate_size+look_back,:]
starter_point += validate_size
test_x, test_y = data[starter_point:starter_point+test_size,:], data[starter_point+look_back:starter_point+test_size+look_back,:]

# reshape input to be [time steps,samples,features]
train_x, train_y = np.expand_dims(train_x, 1), np.expand_dims(train_y, 1)
validate_x, validate_y = np.expand_dims(validate_x, 1), np.expand_dims(validate_y, 1)
test_x, test_y = np.expand_dims(test_x, 1), np.expand_dims(test_y, 1)

seed = FLAGS.seed
torch.manual_seed(seed)

rmse_list = []
mae_list = []

if FLAGS.algorithm == 'RNN':
    model = models.RNN(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size)
elif FLAGS.algorithm == 'LSTM':
    model = models.LSTM(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size)
elif FLAGS.algorithm == 'mRNN_fixD':
    model = models.MRNNFixD(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size, k=FLAGS.K)
elif FLAGS.algorithm == 'mRNN':
    model = models.MRNN(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size, k=FLAGS.K)
elif FLAGS.algorithm == 'mLSTM_fixD':
    model = models.MLSTMFixD(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size, k=FLAGS.K)
elif FLAGS.algorithm == 'mLSTM':
    model = models.MLSTM(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size, k=FLAGS.K)
elif FLAGS.algorithm.startswith('DORNN'):
    model = dornn.DORNN(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size, prefix=FLAGS.algorithm[5:])
elif FLAGS.algorithm.startswith('TRNN'):
    model = trnn.TRNN(input_size=FLAGS.input_size, hidden_size=FLAGS.hidden_size, output_size=FLAGS.output_size, prefix=FLAGS.algorithm[4:])
else:
    print('Algorithm selection ERROR!!!')


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=FLAGS.lr)
best_loss = np.infty
best_train_loss = np.infty
stop_criterion = 1e-5
rec = np.zeros((FLAGS.epochs, 3))
val_loss = -1
train_loss = -1
cnt = 0

def train(do_test=False):
    model.train()
    optimizer.zero_grad()
    target = torch.from_numpy(train_y).float()
    output, hidden_state = model(torch.from_numpy(train_x).float())
    with torch.no_grad():
        # val
        val_y, _ = model(torch.from_numpy(validate_x).float(), hidden_state)
        target_val = torch.from_numpy(validate_y).float()
        val_loss = criterion(val_y, target_val)

        # test
        if do_test:
            test_predict, _ = model(torch.from_numpy(test_x).float(), hidden_state)
            test_predict = test_predict.detach().numpy()
            test_predict_r = scaler.inverse_transform(test_predict[:, 0, :])
            test_y_r = scaler.inverse_transform(test_y[:, 0, :])

            test_rmse = math.sqrt(mean_squared_error(test_y_r[:, 0], test_predict_r[:, 0]))
            test_mape = (abs((test_predict_r[:, 0] - test_y_r[:, 0]) / test_y_r[:, 0])).mean()
            test_mae = mean_absolute_error(test_predict_r[:, 0], test_y_r[:, 0])
        else:
            test_rmse, test_mape, test_mae = 0,0,0

    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss, val_loss, test_rmse, test_mape, test_mae

start_time = time.time()
for e in range(FLAGS.epochs):
    if (e+1)%10 == 0:
        loss, val_loss, test_rmse, test_mape, test_mae = train(True)
        time_elapsed = time.time() - start_time
        start_time = time.time()

        logging.info(' =========== epoch {:4d}/{:4d} ({:3.2f}s) =========== '.format(e+1, FLAGS.epochs, time_elapsed))
        logging.info('train_loss = {:2.5f} | val_loss = {:2.5f}'.format(loss.item(), val_loss.item()))
        logging.info('test_rmse = {:2.5f} | test_mape = {:2.5f} | test_mae = {:2.5f}'.format(test_rmse, test_mape, test_mae))
        logging.info('\n')
    else:
        train(False)
else:
    print ('{:2.5f},{:2.5f},{:2.5f},{:2.5f},{:2.5f}'.format(loss.item(), val_loss.item(), test_rmse, test_mape, test_mae))