from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.contrib import rnn

#Funzione che parsa il csv
def parse_csv(line):
  example_defaults = [[0.], [0.], [0.], [0.], [0.], [0]]  # sets field types
  parsed_line = tf.decode_csv(line, example_defaults)
  # First 5 fields are features, combine into single tensor
  features = tf.reshape(parsed_line[:-1], shape=(5,))
  # Last field is the label
  label = tf.reshape(parsed_line[-1], shape=())
  return features,label

#Runtime..

#1. Preparare training set
#train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

#train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
#print("Local copy of the dataset file: {}".format(train_dataset_fp))
#train_dataset_fp = "/home/mirkpul/Desktop/fca.csv"
#train_dataset_fp = "data/fca_normalizzazione.csv"
train_dataset_fp = "data/tesla.csv"

#define constants
#unrolled through 5 time steps
time_steps=5
#hidden LSTM units
num_units=128
#total 5 inputs
n_input=5
#learning rate for adam
learning_rate=0.001
#mnist is meant to be classified in 10 classes(0-9).
n_classes=2
#size of batch
batch_size=10
#epoch
num_epoch=800
#to skip
to_skip = 2


train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(to_skip)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(batch_size*time_steps)
train_dataset = train_dataset.repeat()

iterator = train_dataset.make_one_shot_iterator()
features,label = iterator.get_next()

#2. Select the model. Sequential e' una rete neurale fully connected

#weights and biases of appropriate shape to accomplish above task
out_weights=tf.Variable(tf.random_normal([num_units,n_classes]))
out_bias=tf.Variable(tf.random_normal([n_classes]))

#defining placeholders
#input placeholder
x=tf.placeholder("float",[None,time_steps,n_input])
#input label placeholder
y=tf.placeholder("int32",[None,n_classes])

#processing the input tensor from [batch_size,n_steps,n_input] to "time_steps" number of [batch_size,n_input] tensors
input=tf.unstack(x ,time_steps,1)

#defining the network
lstm_layer=rnn.BasicLSTMCell(num_units,forget_bias=1)
outputs,_=rnn.static_rnn(lstm_layer,input,dtype="float32")

#converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction=tf.matmul(outputs[-1],out_weights)+out_bias

#3. Creazione ottimizzatore, discesa del gradiente
#loss_function
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
#optimization
opt=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

#model evaluation
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

#4. Training time
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

#initialize variables
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    iter=1
    while iter<num_epoch:
        #batch_x,batch_y=sess.run([features, label])
        batch_x,batch_y = sess.run((features,label));
        print(batch_x)
        print("before reshaping")

        #batch_x=batch_x.reshape((batch_size,time_steps,n_input))
        tf.reshape(batch_x,[batch_size,time_steps,n_input])
        print("batch x:",batch_x)
        print("batch y:",batch_y)
        print("after reshaping")

        print("doing epoch", iter)

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        print("done epoch", iter)

        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={x:batch_x,y:batch_y})
            los=sess.run(loss,feed_dict={x:batch_x,y:batch_y})
            print("For iter ",iter)
            print("Accuracy ",acc)
            print("Loss ",los)
            print("__________________")

        iter=iter+1

    # Wait for threads to finish.
    coord.join(threads)
    sess.close()
'''
    while True:
      try:
        print("doing..")
        example_data, country_name = sess.run([features, label])
        print(example_data, country_name)
      except tf.errors.OutOfRangeError:
        print("out of range")
        break
'''