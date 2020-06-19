import tensorflow as tf

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# ============
def multivariate_data(dataset, target, start_index, end_index, history_size,
                      target_size, step, single_step=False):
  data = []
  labels = []

  start_index = start_index + history_size
  if end_index is None:
    end_index = len(dataset) - target_size

  for i in range(start_index, end_index):
    indices = range(i-history_size, i, step)
    data.append(dataset[indices])

    if single_step:
      labels.append(target[i+target_size])
    else:
      labels.append(target[i:i+target_size])

  return np.array(data), np.array(labels)

# ============
def create_time_steps(length):
  return list(range(-length, 0))

# ============
def baseline(history):
  return np.mean(history)

# =============
colnames = ['year','month','val']

pna_all = pd.read_csv('norm.pna.monthly.b5001.current.ascii', sep='\s+', header=None, names=colnames)  

ao_all = pd.read_csv('monthly.ao.index.b50.current.ascii', sep='\s+', header=None, names=colnames)

soi_all = pd.read_csv('monthly.soi.current.ascii', sep='\s+', header=None, names=colnames)

eawr_all = pd.read_csv('monthly.eawr.current.ascii', sep='\s+', header=None, names=colnames)

# ============
time_colnames = ['time']

time_index_all = pd.read_csv('time_index.ascii', sep='\s+', header=None, names=time_colnames)

print(time_index_all['time'])

# ===========
#print(pna_all['year'])
#print(pna_all['month'])
#print(pna_all['val'])

pna_mean = np.mean(pna_all['val'])
pna_std  = np.std(pna_all['val'])

print(pna_mean)
print(pna_std)

#print(np.min(pna_all['val']))

pna = (pna_all['val'].values - pna_mean)/pna_std

# ===============
ao_mean = np.mean(ao_all['val'])
ao_std  = np.std(ao_all['val'])

#print(ao_all['year'])
#print(ao_all['month'])
#print(ao_all['val'])

print(ao_mean)
print(ao_std)

ao = (ao_all['val'].values - ao_mean)/ao_std

# ======
soi_mean = np.mean(soi_all['val'])
soi_std  = np.std(soi_all['val'])

print(soi_mean)
print(soi_std)

soi = (soi_all['val'].values - soi_mean)/soi_std

ntimes_all = len(soi)

print('ntimes_all = ', ntimes_all)

# ======
eawr_mean = np.mean(eawr_all['val'])
eawr_std  = np.std(eawr_all['val'])

print(eawr_mean)
print(eawr_std)

eawr = (eawr_all['val'].values - eawr_mean)/eawr_std

# ====
#dataset = np.empty((ntimes_all, 3))
dataset = np.empty((ntimes_all, 4))

print('dataset.shape', dataset.shape)
#dataset.shape =  (420551, 3)

dataset[:,0] = ao[:]
dataset[:,1] = pna[:]
dataset[:,2] = soi[:]
dataset[:,3] = eawr[:]

print(dataset)

# ===========
TRAIN_SPLIT = 500

tf.random.set_seed(13)

past_history = 6
future_target = 3
STEP = 1

x_train_single, y_train_single = multivariate_data(dataset, dataset[:, 1], 0,
                                                   TRAIN_SPLIT, past_history,
                                                   future_target, STEP,
                                                   single_step=True)
x_val_single, y_val_single = multivariate_data(dataset, dataset[:, 1],
                                               TRAIN_SPLIT, None, past_history,
                                               future_target, STEP,
                                               single_step=True)

print ('Single window of past history : {}'.format(x_train_single[0].shape))


print('y_train_single.shape', y_train_single.shape)

# ===========
BATCH_SIZE = 256
BUFFER_SIZE = 500

train_data_single = tf.data.Dataset.from_tensor_slices((x_train_single, y_train_single))
train_data_single = train_data_single.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_single = tf.data.Dataset.from_tensor_slices((x_val_single, y_val_single))
val_data_single = val_data_single.batch(BATCH_SIZE).repeat()

# =============
single_step_model = tf.keras.models.Sequential()
single_step_model.add(tf.keras.layers.LSTM(32,
                                           input_shape=x_train_single.shape[-2:]))
single_step_model.add(tf.keras.layers.Dense(1))

single_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss='mae')

for x, y in val_data_single.take(1):
  print(single_step_model.predict(x).shape)

# =============
EVALUATION_INTERVAL = 6
EPOCHS = 10

single_step_history = single_step_model.fit(train_data_single, epochs=EPOCHS,
                                            steps_per_epoch=EVALUATION_INTERVAL,
                                            validation_data=val_data_single,
                                            validation_steps=50)

ncount = 494 - 1

mae = 0.
ncount_mae = 0.

for x, y in val_data_single.take(200):
   ncount = ncount + 1
   #print(pna[ncount], single_step_model.predict(x)[0])
   mae = mae + abs(pna[ncount]-single_step_model.predict(x)[0])
   ncount_mae = ncount_mae + 1.

print('avg_mae =' , mae/ncount_mae)
