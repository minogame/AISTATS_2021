import subprocess
from subprocess import Popen
import numpy as np
from scipy.io import savemat
import argparse, copy, time

PARSER = argparse.ArgumentParser()
PARSER.add_argument('--dataset', type=str, default='arfima')
PARSER.add_argument('--algorithm', type=str, default='RNN')
PARSER.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
PARSER.add_argument('--lr', type=float, default=0.01)
PARSER.add_argument('--hidden_size', type=int, default=1)
PARSER.add_argument('--input_size', type=int, default=1)
PARSER.add_argument('--output_size', type=int, default=1)
PARSER.add_argument('--train_size', type=int, default=1000)
PARSER.add_argument('--validate_size', type=int, default=300)
PARSER.add_argument('--test_size', type=int, default=300)
PARSER.add_argument('--look_back', type=int, default=1)
PARSER.add_argument('--K', type=int, default=100)
PARSER.add_argument('--patience', type=int, default=100)
PARSER.add_argument('--log_file', type=str, default=None)
PARSER.add_argument('--seed', type=str, default='[432,123]')
FLAGS = PARSER.parse_args()

seeds = eval(FLAGS.seed)
seeds = list(seeds)

scripts = []
for seed in seeds:
	script = ['python3','new_train.py']
	script.append('--seed={}'.format(seed))

	for key in FLAGS.__dict__:
		if key in ['seed', 'log_file']:
			continue
		script.append('--{}={}'.format(key, FLAGS.__dict__[key]))

	script.append('--gang_mode=True')
	scripts.append(script)

mat_dict = copy.deepcopy(FLAGS.__dict__)
mat_dict.pop('seed', None); mat_dict.pop('log_file', None)

if FLAGS.log_file is None:
	log_file = 'mats/{}.mat'.format(int(time.time()))
else:
	log_file = 'mats/{}.mat'.format(FLAGS.log_file)

popens = [ Popen(script, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True) for script in scripts ]

results = []
seed_alive = []
for popen,seed in zip(popens, seeds):
	try:
		output, error = popen.communicate()
		results.append([float(s) for s in output.split(',')])
		seed_alive.append(seed)
	except:
		print ('=== Seed {} raises an error, excluded. ==='.format(seed))
		print (error.split('\n')[-2])
		continue
if len(seed_alive) == 0:
	print ('\nAll seeds have been exclued, no result will be kept.')
	exit()

results = np.array(results)
mat_dict['final_train_loss'] = results[:,0]
mat_dict['val_loss']				 = results[:,1]
mat_dict['test_rmse']				 = results[:,2]
mat_dict['test_mape']				 = results[:,3]
mat_dict['test_mae']				 = results[:,4]
mat_dict['seeds']						 = seed_alive

mat_dict['final_train_loss_mean_std_min'] = [np.mean(results[:,0]), np.std(results[:,0]), np.min(results[:,0])]
mat_dict['val_loss_mean_std_min']				 	= [np.mean(results[:,1]), np.std(results[:,1]), np.min(results[:,1])]
mat_dict['test_rmse_mean_std_min']				= [np.mean(results[:,2]), np.std(results[:,2]), np.min(results[:,2])]
mat_dict['test_mape_mean_std_min']				= [np.mean(results[:,3]), np.std(results[:,3]), np.min(results[:,3])]
mat_dict['test_mae_mean_std_min']					= [np.mean(results[:,4]), np.std(results[:,4]), np.min(results[:,4])]


savemat(log_file, mat_dict)
for attribute, value in mat_dict.items():
  print('{} : {}'.format(attribute, value))