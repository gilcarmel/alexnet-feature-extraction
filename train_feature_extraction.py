import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import numpy as np
from tensorflow.contrib.learn.python.learn.datasets import mnist

# TODO: Load traffic signs data.
nb_classes = 43

training_file = "traffic-signs-data/train.p"
testing_file = "traffic-signs-data/test.p"

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# number of training examples
n_train = len(X_train)

# number of testing examples
n_test = len(X_test)

y_train = mnist.dense_to_one_hot(y_train, nb_classes)
y_test = mnist.dense_to_one_hot(y_test, nb_classes)


print("done loading")

# TODO: Split data into training and validation sets.

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, 32, 32, 3))
# TODO: Resize the images so they can be fed into AlexNet.
# HINT: Use `tf.image.resize_images` to resize the images
resized = tf.image.resize_images(x, (227, 227))

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)

# TODO: Add the final layer for traffic sign classification.
shape = (fc7.get_shape().as_list()[-1], nb_classes)
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros(nb_classes))
logits = tf.nn.xw_plus_b(fc7, fc8W, fc8b)
probs = tf.nn.softmax(logits)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.
y_true = tf.placeholder(tf.float32, shape=[None, nb_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=probs,
                                                        labels=y_true)
y_pred_cls = tf.argmax(probs, dimension=1)
cost = tf.reduce_mean(cross_entropy)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Use Adam optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)



# TODO: Train and evaluate the feature extraction model.
import time
from datetime import timedelta


def shuffle_training_set():
    global current_index, X_train, y_train
    current_index = 0
    perm = np.arange(n_train)
    np.random.shuffle(perm)
    X_train = X_train[perm]
    y_train = y_train[perm]


# Gets the next batch of images (randomize order at the beginning of each epoch)
def next_batch(batch_size):
    global current_index
    assert current_index + batch_size < n_train
    batch_x, batch_y = X_train[current_index:current_index + batch_size], y_train[
                                                                          current_index:current_index + batch_size]
    current_index += batch_size
    return batch_x, batch_y


# Optimize one epoch
def optimize():
    # Start-time used for printing time-usage below.
    start_time = time.time()

    dropout = 0.55
    shuffle_training_set()

    train_batch_size = 256
    batch_num = 0
    while current_index + train_batch_size < n_train:
        x_batch, y_true_batch = next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        batch_num = batch_num + 1


    #Print the accuracy of the last batch
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    msg = "Optimization batch: {0:>6}, Training Accuracy: {1:>6.1%}"
    print(msg.format(batch_num, acc))

    #Print epoch training time
    end_time = time.time()
    time_dif = end_time - start_time
    print("Epoch time usage: " + str(timedelta(seconds=int(round(time_dif)))))



import os.path

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
checkpoint_name = "model.ckpt"

num_epochs = 2

# Split the test-set into smaller batches of this size.
test_batch_size = 256


def print_test_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):
    num_test_images = len(X_test)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test_images, dtype=np.int)

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test_images:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test_images)

        # Get the images from the test-set between index i and j.
        images = X_test[i:j, :]

        # Get the associated labels.
        labels = y_test[i:j, :]

        # Create a feed-dict with these images and labels.
        feed_dict = {x: images,
                     y_true: labels}

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j

    # Convenience variable for the true class-numbers of the test-set.
    cls_true = np.argmax(y_test[:len(cls_pred)], axis=1)

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test_images

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test_images))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        # plot_example_errors(cls_pred, cls_true, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        # plot_confusion_matrix(cls_pred, cls_true)


with tf.Session() as session:

    session.run(tf.initialize_all_variables())
    for i in range(num_epochs):
        print ("Epoch {}:".format(i))
        optimize()
        save_path = saver.save(session, checkpoint_name)
        print("Model saved in file: %s" % save_path)
        if i%5==0:
            print_test_accuracy(True,True)
    print_test_accuracy(True,True)

