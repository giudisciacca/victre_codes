import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import h5py
from dataClass import *


batch_size = 200;
bSize = 200;
nfeat = 16;
loadname ='';


def fcnl(w,b,x):
    return tf.nn.relu(tf.matmul(w,x))+b
def weight_var(shape,name):
    W = tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer(uniform = False))
    return W

def hinge_loss(x,y):
    l = tf.reduce_sum(tf.maximum(tf.zeros_like(x), tf.ones_like(x)- tf.multiply(x,y)))
    l = tf.reduce_sum(tf.square(tf.ones_like(x)- tf.multiply(x,y)));
    return l
def cross_loss(x,y):
    l = -tf.reduce_sum(tf.multiply(x, tf.log(y)))

    return l

def FCN(x0):
    x = tf.expand_dims(x0,axis=2)
    base = 2;
    W1 = weight_var([ nfeat*base,nfeat],'W1')
    b1 = weight_var([1],'b1')
    W2 = weight_var([ nfeat*base*2, nfeat*base],'W2')
    b2 = weight_var([1],'b2')
    W3 = weight_var([ nfeat*base*4,nfeat*base*2],'W3')
    b3 = weight_var([1],'b3')
    W4 = weight_var([ nfeat*base*2,nfeat*base*4],'W4')
    b4 = weight_var([1],'b4')
    W5 = weight_var([ nfeat*base, nfeat*base*2],'W5')
    b5 = weight_var([1],'b5')
    Wf = weight_var([ 1,nfeat*base],'Wf')
    bf = weight_var([1],'bf')
    l1 = fcnl(W1,b1,x);
    l2 = fcnl(W2,b2,l1);
    l3 = fcnl(W3, b3, l2);
    l4 = fcnl(W4, b4, l3);
    l5 = fcnl(W5, b5, l4);
    lf = (tf.matmul(Wf,l5)+bf)
    #model_output = tf.nn.sigmoid(lf);
    return lf,b5

def SVM(x):
    # Gaussian (RBF) kernel
    gamma = tf.constant(-10.0)
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM Model
    model_output = tf.matmul(b, my_kernel)

    return loss, model_output



sess = tf.Session()

features = tf.placeholder(shape=[None, nfeat], dtype=tf.float32)
labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)
#with tf.name_scope('Net'):
#    # net

out,b1 = FCN(features);

with tf.name_scope('loss'):
    # loss
    #loss = (tf.compat.v1.losses.hinge_loss(out[:,:,0], labels))
    rightclass = tf.abs(tf.math.round(out[:,:,0]) - labels);
    loss = tf.reduce_sum(tf.abs((out[:,:,0]) - labels));
    #loss= hinge_loss(out[:,0,0],labels[:,0])
    #loss = cross_loss(out[:,0,0],labels[:,0])
    #bce =tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #loss = bce( labels,out[:,:,0])
    regularizer = 0;
#optimizer
with tf.name_scope('training'):
    learningRate = tf.constant(1e-3)
    train_step = tf.train.AdamOptimizer(learningRate).minimize(loss + regularizer)

# load dataset
sess.run( tf.global_variables_initializer())
loadname = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/CLASSIFICATION_538_truth'
dataset = fullDataset(loadname);
saver = tf.train.Saver();
# training
lVal = 0.000001;
error_val_old = 1;
for i in range(0,200000):

    batch = dataset.train.next_batch(bSize)
    #      test1 = batch[0]
    #      test2 = batch[1]
    feed_train = {features: batch[0], labels: batch[1], learningRate: lVal}
    sess.run(train_step, feed_dict=feed_train)
    _, train_result, good = sess.run([train_step, loss, rightclass], feed_dict=feed_train)

    #        if i % 500 == 0:
    #            lVal = lVal*1.5

    if i % 20 == 0:
        # train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
        # testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})

        feed_test = {features: dataset.test.features, labels: dataset.test.labels, learningRate: lVal}

        test_result,b = sess.run([loss,b1],  feed_dict=feed_test)
        print(b)

        print('iter={},  losstrain={}, losstest={}'.format(i, train_result, test_result))

        # run for all samples

        if i > 400:
            # check mean validation error
            sumok = 0;
            array_val = [None] * (np.shape(dataset.valid.features)[0]-bSize+1)

            feed_val = {features: dataset.valid.features,
                         labels: dataset.valid.labels}
            loss_val, classok = sess.run([loss,rightclass], feed_dict=feed_val)

            if loss_val < error_val_old:
                error_val_old = loss_val
                print('NEW CHAMPION SAVING')#+%f GOOD = %d' % (loss_val, classok))

                #saved_path = saver.save(sess, OutName[0][0:-4] + 'best_binaryInput')

## gen set for space
#cated = list()
#cated = [np.linspace(np.min(dataset.test.features[:,i]),np.max(dataset.test.features[:,i]),3) for i in range(0, dataset.test.features.shape[1])]
#f = np.meshgrid(*cated)
#feat_domain = np.c_()
#feat_domain=
feed_domain = {features: feat_domain, labels: lab_domain}






## visualisation PCA decomposition


'''''

first_term = tf.reduce_sum(b)
b_vec_cross = tf.matmul(tf.transpose(b), b)
y_target_cross = tf.matmul(y_target, tf.transpose(y_target))
second_term = tf.reduce_sum(tf.multiply(my_kernel, tf.multiply(b_vec_cross, y_target_cross)))
loss = tf.negative(tf.subtract(first_term, second_term))

iris = datasets.load_iris()
x_vals = np.array([[x[0], x[3]] for x in iris.data])
y_vals = np.array([1 if y==0 else -1 for y in iris.target])
class1_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==1]
class1_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==1]
class2_x = [x[0] for i,x in enumerate(x_vals) if y_vals[i]==-1]
class2_y = [x[1] for i,x in enumerate(x_vals) if y_vals[i]==-1]

plt.scatter(x_vals[:, 0], x_vals[:, 1],c=y_vals, s=50, cmap='autumn');
plt.show()




batch_size = 50
x_data = tf.placeholder(shape=[None, 2], dtype=tf.float32)
y_target = tf.placeholder(shape=[None, 1], dtype=tf.float32)
prediction_grid = tf.placeholder(shape=[None, 2], dtype=tf.float32)
b = tf.Variable(tf.random_normal(shape=[1,batch_size]))




rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1),[-1,1])
rB = tf.reshape(tf.reduce_sum(tf.square(prediction_grid), 1),[-1,1])
pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data,
tf.transpose(prediction_grid)))), tf.transpose(rB))
pred_kernel = tf.exp(tf.multiply(gamma, tf.abs(pred_sq_dist)))
prediction_output = tf.matmul(tf.multiply(tf.transpose(y_target),b),pred_kernel)
prediction = tf.sign(prediction_output-tf.reduce_mean(prediction_output))
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.squeeze(prediction),tf.squeeze(y_target)), tf.float32))


my_opt = tf.train.GradientDescentOptimizer(0.01)
train_step = my_opt.minimize(loss)
init = tf.initialize_all_variables()
sess.run(init)



loss_vec = []
batch_accuracy = []
for i in range(1000):
    rand_index = np.random.choice(len(x_vals), size=batch_size)
    X = x_vals[rand_index]
    Y = np.transpose([y_vals[rand_index]])
    sess.run(train_step, feed_dict={x_data: X, y_target:Y})
    temp_loss = sess.run(loss, feed_dict={x_data: X, y_target: Y})
    loss_vec.append(temp_loss)
    acc_temp = sess.run(accuracy, feed_dict={x_data: X,y_target: Y,prediction_grid:X})
    batch_accuracy.append(acc_temp)

# Create a mesh to plot points in
x_vals = x_vals.astype(np.float)
x_min, x_max = x_vals[:, 0].min() - 1, x_vals[:, 0].max() + 1
y_min, y_max = x_vals[:, 1].min() - 1, x_vals[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_predictions = sess.run(prediction_grid, feed_dict={x_data: x_vals,
                                                     y_target: np.transpose([y_vals]),
                                                     prediction_grid: grid_points})
grid_predictions = grid_predictions.reshape(xx.shape)

# Plot points and grid
plt.contourf(xx, yy, grid_predictions, cmap=plt.cm.Paired, alpha=0.8)
plt.plot(class1_x, class1_y, 'ro', label='I. setosa')
plt.plot(class2_x, class2_y, 'kx', label='Non setosa')
plt.title('Gaussian SVM Results on Iris Data')
plt.xlabel('Petal Length')
plt.ylabel('Sepal Width')
plt.legend(loc='lower right')
plt.ylim([-0.5, 3.0])
plt.xlim([3.5, 8.5])
plt.show()

# Plot batch accuracy
plt.plot(batch_accuracy, 'k-', label='Accuracy')
plt.title('Batch Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Plot loss over time
plt.plot(loss_vec, 'k-')
plt.title('Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Loss')
plt.show()

'''