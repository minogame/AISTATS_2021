class TrainConfig(object):
  """Tiny config, for testing."""
  init_scale = 0.1
  learning_rate = 1e-2
  decay_rate = 0.8
  max_grad_norm = 10
  hidden_size = 8
  num_layers = 2
  inp_steps =  15
  horizon = 1
  num_lags = 2
  num_orders = 3
  rank_vals= [2]
  num_freq = 2
  #below are modified in config
  training_steps = int(1e4)
  keep_prob = 1.0
  sample_prob = 0.0 
  batch_size = 25
  use_sched_samp = False
  is_branched = True
  is_weighted = True
  r_se = 2
