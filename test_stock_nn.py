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
  return features, label

#Funione loss
def loss(model, x, y):
  y_ = model(x)
  return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

#Funzione gradiente
def grad(model, inputs, targets):
  with tfe.GradientTape() as tape:
    loss_value = loss(model, inputs, targets)
  return tape.gradient(loss_value, model.variables)

#Runtime..
#1. Abilitare esecuzione diretta
tf.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

#2. Scaricare training set
#train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"

#train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url), origin=train_dataset_url)
#train_dataset_fp = "/home/mirkpul/Desktop/fca.csv"
train_dataset_fp = "data/fca_normalizzazione.csv"
train_dataset_fp = "data/tesla.csv"
#train_dataset_fp = "/home/mirkpul/Desktop/fca_standardizzazione.csv"


print("Local copy of the dataset file: {}".format(train_dataset_fp))

#3. Creare il train dataset
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(2)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)

# View a single example entry from a batch
features, label = tfe.Iterator(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])

#4. Select the model. Sequential e' una rete neurale fully connected
model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation="relu", input_shape=(5,)),  # input shape required
  tf.keras.layers.Dense(10, activation="relu"),
  tf.keras.layers.Dense(2)
])

#5. Creazione ottimizzatore, discesa del gradiente
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)

#6. Training time
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 201

for epoch in range(num_epochs):
  epoch_loss_avg = tfe.metrics.Mean()
  epoch_accuracy = tfe.metrics.Accuracy()

  # Training loop - using batches of 32
  for inputs, target in tfe.Iterator(train_dataset):
    # Optimize the model
    grads = grad(model, inputs, target)
    #zipped = zip(grads, model.variables)
    #print(zipped)
    optimizer.apply_gradients(zip(grads, model.variables))

    # Track progress
    epoch_loss_avg(loss(model, inputs, target))  # add current batch loss
    # compare predicted label to actual label
    epoch_accuracy(tf.argmax(model(inputs), axis=1, output_type=tf.int32), target)

  # end epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 50 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))