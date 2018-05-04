from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

#Funzione che parsa il csv
def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 5 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(5,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features,label

num_epochs = 201
num_batch = 32
to_skip = 2

train_dataset_fp = ["data/tesla.csv"]
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(to_skip)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(num_batch)
train_dataset = train_dataset.repeat()


iterator = train_dataset.make_one_shot_iterator()
features,label = iterator.get_next()

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(100):
    	print(i)
    	ft,l = sess.run((features,label))
    	print(ft,l)