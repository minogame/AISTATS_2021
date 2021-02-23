from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import random_seed
import tensorflow as tf
from tensorflow.contrib import rnn
from reader import read_data_sets
from model_seq2seq import *
from trnn import *
import numpy 
from train_config import *

flags = tf.flags
flags.DEFINE_string("model", "TP_LSTM",
          "Model used for learning.")
flags.DEFINE_string("data_path", "./demo.npy",
          "Data input directory.")
flags.DEFINE_string("save_path", "./log/test/",
          "Model output directory.")
flags.DEFINE_bool("use_sched_samp", False,
                  "Use scheduled sampling in training")
flags.DEFINE_integer("inp_steps", 15, "burn in steps")
flags.DEFINE_integer("out_steps", 54, "test steps")
flags.DEFINE_integer("hidden_size", 8, "hidden layer size")
flags.DEFINE_float("learning_rate", 1e-3, "learning rate")
flags.DEFINE_float("decay_rate", 0.8, "learning rate")
flags.DEFINE_float("order", 1.0, "num of initial order")
flags.DEFINE_integer("rank", 2, "rank for tt decomposition")
flags.DEFINE_integer("layer", 2, "num of hidden layers")
flags.DEFINE_string('f', '', 'kernel')

FLAGS = flags.FLAGS
print('Flags configuration loaded ...')

# Training Parameters
config = TrainConfig()
config.hidden_size = FLAGS.hidden_size
config.learning_rate = FLAGS.learning_rate
config.decay_rate = FLAGS.decay_rate
config.rank_vals = [FLAGS.rank]
config.num_orders = FLAGS.order
config.num_layers = FLAGS.layer
config.use_sched_samp = FLAGS.use_sched_samp

# Scheduled sampling [optional]
if config.use_sched_samp:
    config.sample_prob = tf.get_variable("sample_prob", shape=(), initializer=tf.zeros_initializer())
sampling_burn_in = 400

# Training Parameters
training_steps = config.training_steps
batch_size = config.batch_size
display_step = 20
inp_steps = FLAGS.inp_steps
out_steps = FLAGS.out_steps

# Read Dataset
dataset, stats = read_data_sets(FLAGS.data_path, True, inp_steps, out_steps)

# Network Parameters
num_input = stats['num_input']  # dataset data input (time series dimension: 3)
num_steps = stats['num_steps']

if out_steps is None:
    # Forecast for the rest if horizon is not set
    out_steps = num_steps - inp_steps

# Print training config
print('-'*100)
print('model',FLAGS.model,'|dataset|',FLAGS.data_path,'|input steps|', inp_steps, '|out steps|', out_steps ,'|hidden size|',config.hidden_size, '|hidden layer|', config.num_layers, '|learning rate|',
     config.learning_rate, '|decay rate|', config.decay_rate,'|rank val|','|initial order|', config.num_orders, config.rank_vals, '|batch size|', batch_size)
print('-'*100)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Model = globals()[FLAGS.model]

# tf Graph input
X = tf.placeholder("float", [None, inp_steps, num_input])
Y = tf.placeholder("float", [None, out_steps, num_input])

# Decoder output
Z = tf.placeholder("float", [None, out_steps, num_input])

Model = globals()[FLAGS.model]
with tf.name_scope("Train"):
    with tf.variable_scope("Model", reuse=None):
        train_pred = Model(X, Y, True,  config)
        
with tf.name_scope("Test"):
    with tf.variable_scope("Model", reuse=True):
        test_pred = Model(X, Y, False,  config)

# Define loss and optimizer
train_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(train_pred, Z)))
test_loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(test_pred, Z)))

# Exponential learning rate decay 
global_step = tf.Variable(0, trainable=False)
starter_learning_rate = config.learning_rate
learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           2000, config.decay_rate, staircase=True)
optimizer = tf.train.RMSPropOptimizer(learning_rate)
train_op = optimizer.minimize(train_loss,global_step=global_step)

# Scheduled sampling params
eps_min = 0.1 # minimal prob

# Write summary
train_summary = tf.summary.scalar('train_loss', train_loss)
valid_summary = tf.summary.scalar('test_loss', test_loss)
lr_summary = tf.summary.scalar('learning_rate', learning_rate)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Saver for the model and loss
saver = tf.train.Saver()
hist_loss =[]

# Start training
with tf.Session() as sess:
    # Merge all the summaries and write them out to /log/tlstm (by default)
    merged = tf.summary.merge_all()
    summary_writer_train = tf.summary.FileWriter(FLAGS.save_path + '/train',sess.graph)
    summary_writer_test = tf.summary.FileWriter(FLAGS.save_path + '/test',sess.graph)
    
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y, batch_z = dataset.train.next_batch(batch_size)
        
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, Z:batch_z})
    
        if step % display_step == 0 or step == 1:
            
            # Calculate train loss 
            summary, loss = sess.run([merged,train_loss], feed_dict={X: batch_x,Y: batch_y, Z:batch_z})
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            summary_writer_train.add_run_metadata(run_metadata, 'step%03d' % step)
            summary_writer_train.add_summary(summary, step)
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) )
            
            # Print variables
            check_result = []
            check_v_name = [v.name for v in tf.trainable_variables() ]
            values = sess.run(check_v_name)
            for n, v in zip(check_v_name, values):
                check_result.append([n, v])
            #print(check_result)
            
            # Calculate validation loss
            valid_enc_inps = dataset.validation.enc_inps.reshape((-1, inp_steps, num_input))
            valid_dec_inps = dataset.validation.dec_inps.reshape((-1, out_steps, num_input))
            valid_dec_outs = dataset.validation.dec_outs.reshape((-1, out_steps, num_input))
            va_sum, va_loss = sess.run([valid_summary,test_loss], \
                                       feed_dict={X: valid_enc_inps, Y: valid_dec_inps, Z: valid_dec_outs})
            summary_writer_test.add_summary(va_sum, step)
            print("Validation Loss:", va_loss)
            
            # Check if overfit
            hist_loss.append(va_loss)
            
            #Update sampling prob
            if FLAGS.use_sched_samp and step > sampling_burn_in:
                sample_prob = max(eps_min, 1.0-step/(2*training_steps))
                sess.run(tf.assign(config.sample_prob, sample_prob))
                print('Sampling prob:', sample_prob)

    print("Optimization Finished!")
    plt.yscale('log')
    plt.plot(hist_loss)
    plt.savefig('Validation.png')
    # Calculate accuracy for test datasets
    test_enc_inps = dataset.test.enc_inps.reshape((-1, inp_steps, num_input))
    test_dec_inps = dataset.test.dec_inps.reshape((-1, out_steps, num_input))
    test_dec_outs = dataset.test.dec_outs.reshape((-1, out_steps, num_input))

    # Fetch the predictions 
    fetches = {
        "true":Z,
        "pred":test_pred,
        "loss":test_loss
    }
    test_vals = sess.run(fetches, feed_dict={X: test_enc_inps, Y: test_dec_inps, Z: test_dec_outs})
    print("Testing Loss:", test_vals["loss"])

    # Save the variables to disk.
    save_path = saver.save(sess, FLAGS.save_path)
    print("Model saved in file: %s" % save_path)
                                            
    # Save predictions (in .npy)
    numpy.save(save_path+"predict.npy", (test_vals["true"], test_vals["pred"]))
                                            
    # Save config file
    with open(save_path+"config.out", 'w') as f:
        f.write('hidden_size:'+ str(config.hidden_size)+'\t'+ 'learning_rate:'+ str(config.learning_rate)+ '\n')
        f.write('train_error:'+ str(loss) +'\t'+ 'valid_error:' + str(va_loss) + '\t'+ 'test_error:'+ str(test_vals["loss"]) +'\n')

plt.cla()
y_true = test_vals["true"]
y_pred = test_vals["pred"]
plt.plot(y_true[0:10,:,0].T,':')
plt.plot(y_pred[0:10,:,0].T,'-')
plt.savefig('sampleResult.png')