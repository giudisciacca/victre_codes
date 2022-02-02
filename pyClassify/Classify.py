import tensorflow as tf
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
import h5py
from dataClass import *

NORMALISE =True;
LOGNORM=True;
batch_size = 200;
bSize = 121;
nfeat = 16;
loadname ='';



def quantify(pred,lab):

    lab = 1-lab;
    pred = 1-pred;

    TP = np.sum(np.multiply(lab,pred));
    FP = np.sum(np.multiply(1-lab,pred));
    FN = np.sum(np.multiply(lab,1-pred));
    TN = np.sum(np.multiply(1-lab,1-pred));
    prec = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = (2*(prec*rec))/(prec+rec)
    acc = 1- (np.sum(np.abs(lab-pred))/np.shape(lab)[0])
    print('acc = {}'.format(acc))
    print('prec = {}'.format(prec))
    print('rec = {}'.format(rec))
    print('f1 = {}'.format(f1))
    return acc,prec,rec,f1


def fcnl(w,b,x, outnum = 4):
    out =tf.contrib.layers.fully_connected(
        x,
        outnum,
        activation_fn=tf.nn.relu,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(uniform = False),
        weights_regularizer=None,
        biases_initializer=tf.zeros_initializer(),
        biases_regularizer=None,
        reuse=None,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope=None
    )

    out = tf.nn.relu(tf.matmul(w,x))+b
    return out

def weight_var(shape,name):
    W =tf.get_variable(name,shape=shape,initializer=tf.contrib.layers.xavier_initializer(uniform = True))
    return W

def hinge_loss(x,y):
    l = tf.reduce_sum(tf.maximum(tf.zeros_like(y), (tf.subtract(tf.ones_like(y), tf.multiply(2*x-1,2*y-1)))))
    #l = tf.reduce_sum(tf.maximum(tf.zeros_like(y), tf.square(tf.subtract(tf.ones_like(y), tf.multiply(2 * x - 1, 2 * y - 1)))))
    #l = tf.reduce_sum(tf.square(tf.ones_like(x)- tf.multiply(x,y)));
    return l
def cross_loss(x,y):
    l = -tf.reduce_sum(tf.multiply(x, tf.log(y)))

    return l

def FCN(x0):
    x = tf.expand_dims(x0,axis=2)
    #x = tf.transpose(x0, perm=[1,0])
    #x = tf.expand_dims(x0, axis=0)
    base = 3
    W1 = weight_var([ nfeat*base*1,nfeat],'W1')
    b1 = weight_var([1],'b1')
    W2 = weight_var([ nfeat*base, nfeat*base*1],'W2')
    b2 = weight_var([1],'b2')
    '''''
    W3 = weight_var([ nfeat*base,nfeat*base],'W3')
    b3 = weight_var([1],'b3')
    W4 = weight_var([ nfeat*base*4,nfeat*base*4],'W4')
    b4 = weight_var([1],'b4')
    W5 = weight_var([ nfeat*base*3, nfeat*base*4],'W5')
    b5 = weight_var([1],'b5')
    W6 = weight_var([ nfeat*base*2, nfeat*base*3],'W6')
    b6 = weight_var([1],'b6')
    W7 = weight_var([ nfeat*base, nfeat*base*2],'W7')
    b7= weight_var([1],'b7')
    '''
    Wf = weight_var([ 1,nfeat*base],'Wf')
    bf = weight_var([1],'bf')

    l1 = fcnl(W1,b1,x,outnum=16*2);
    print(l1)
    l2 = fcnl(W2,b2,l1,outnum=16);
    '''''
    l3 = fcnl(W3, b3, l2,outnum=8);
    l4 = fcnl(W4, b4, l3,outnum=1);
    l5 = fcnl(W5, b5, l4);
    l6 = fcnl(W6, b6, l5);
    l7 = fcnl(W7, b7, l6);
    
    print(l7)
    '''
    lf = tf.nn.relu(tf.matmul(Wf,l2)+bf)
    model_output = tf.nn.sigmoid(lf);
    return lf,b1

def SVM(x):
    b = tf.tile(weight_var([1,16],'SVMw'))
    # Gaussian (RBF) kernel
    gamma = tf.constant(-10.0)
    dist = tf.reduce_sum(tf.square(x), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, tf.multiply(2., tf.matmul(x, tf.transpose(x)))), tf.transpose(dist))
    my_kernel = tf.exp(tf.multiply(gamma, tf.abs(sq_dists)))

    # Compute SVM Model
    model_output = tf.matmul(b, my_kernel)

    return model_output,gamma



sess = tf.Session()

features = tf.placeholder(shape=[None, nfeat], dtype=tf.float32)
labels = tf.placeholder(shape=[None, 1], dtype=tf.float32)
#with tf.name_scope('Net'):
#    # net

#out,b1 = SVM(features);

out,b1 = FCN(features);
with tf.name_scope('loss'):
    # loss
    #loss = (tf.compat.v1.losses.hinge_loss(out[:,:,0], labels))

    rightclass = tf.abs(tf.clip_by_value(tf.math.round(out[:,:,0]) ,0,1)- labels);
    b1 = rightclass;
    print(out)
    #loss = tf.reduce_sum(tf.abs(tf.subtract(out[:,0,0] , labels[:,0])));
    loss= hinge_loss(out[:,0,0],(labels[:,0]))
    out_res = tf.clip_by_value(tf.math.round(out[:, :, 0]), 0, 1)
    lab_res = labels[:,0]
    ######
    '''''
    rightclass = tf.abs(tf.math.round(out[:,:]) - labels);
    b1 = rightclass;
    loss = tf.reduce_sum(tf.abs((out[:,0]) - labels));
    loss= hinge_loss(out[:,0],labels[:,0])
    '''''
    #loss = cross_loss(out[:,0,0],labels[:,0])
    #bce =tf.keras.losses.BinaryCrossentropy(from_logits=True)
    #loss = bce( labels,out[:,:,0])
    #regularizer = tf.nn.l2_loss(weights);
    vars = tf.trainable_variables()
    regularizer = tf.add_n([tf.nn.l2_loss(v) for v in vars]) *0* 0.01
#optimizer
with tf.name_scope('training'):
    learningRate = tf.constant(1e-3)
    train_step = tf.train.AdamOptimizer(learningRate).minimize(loss + regularizer)

# load dataset
sess.run( tf.global_variables_initializer())
loadname = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/JacFD_DTsepWave_coeffs3_706'
dataset = fullDataset(loadname)
saver = tf.train.Saver();



if LOGNORM==True:
    #dataset.train._features = np.log(1-np.min(dataset.train.features)+dataset.train.features)
    #dataset.valid._features =np.log(1-np.min(dataset.valid.features)+dataset.valid.features)
    #dataset.test._features=np.log(1-np.min(dataset.test.features)+dataset.test.features)
    dataset.train._features = np.log(dataset.train.features)
    dataset.valid._features =np.log(dataset.valid.features)
    dataset.test._features=np.log(dataset.test.features)
    #dataset.train._features = np.divide(dataset.train.features - np.mean(dataset.train.features,axis=0),np.std(dataset.train.features,axis=0))
    #dataset.test._features = np.divide(dataset.test.features - np.mean(dataset.test.features,axis=0),np.std(dataset.test.features,axis=0))
    #dataset.valid._features = np.divide(dataset.valid.features - np.mean(dataset.valid.features,axis=0),np.std(dataset.valid.features,axis=0))
if NORMALISE==True:
    concd = np.concatenate((dataset.train.features,dataset.valid.features ,dataset.test.features), axis =0 )
    nmean = np.mean(concd, axis=0)
    nstd= np.std( concd, axis=0)
    dataset.train._features = np.divide(dataset.train.features - nmean,nstd)
    dataset.test._features = np.divide(dataset.test.features - nmean,nstd)
    dataset.valid._features = np.divide(dataset.valid.features - nmean,nstd)
# training
dict_param ={"typ":['JacFD_sepWave_coeffs_724','JacFD_sepWave_coeffs_724_truth'],
             "lVal":[500e-6,],
             "decay":[5000,],
             "regu":[0,],
             "LOG":[True,False]}
lVal = 20e-6#20e-6;
decay=40000#25000;
error_val_old = 1000;
champ_mis_old  = 2;
champ_mis = 2;
for i in range(0,30000):

    batch = dataset.train.next_batch(bSize)
    #      test1 = batch[0]
    #      test2 = batch[1]
    feed_train = {features: batch[0], labels: batch[1], learningRate: lVal}
    feed_train = {features: dataset.train.features, labels: dataset.train.labels, learningRate: lVal}
    #sess.run(train_step, feed_dict=feed_train)
    _, train_result, good = sess.run([train_step, loss, rightclass], feed_dict=feed_train)

    #        if i % 500 == 0:
    #            lVal = lVal*1.5
    if i % decay ==0:
        lVal = lVal/2;
    if i % 10 == 0:
        # train_accuracy = accuracy.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})
        # testPosit = testPos.eval(feed_dict={imag: dataDbar.test.images[0:16], true: dataDbar.test.true[0:16]})

        feed_test = {features: dataset.test.features, labels: dataset.test.labels, learningRate: lVal}

        test_result,b = sess.run([loss,b1],  feed_dict=feed_test)
        #print(b)
        if i %60:
            print('iter={},  losstrain={}, losstest={}, class={}, classperc={} champ = {}, champclass={}'.format(i, train_result, test_result,np.sum(b),
                                                                                   np.sum(b)/np.shape(dataset.test.features)[0],error_val_old, champ_mis))

        # run for all samples

        if i > 20:
            # check mean validation error
            sumok = 0;
            array_val = [None] * (np.shape(dataset.valid.features)[0]-bSize+1)

            feed_val = {features: dataset.valid.features,
                         labels: dataset.valid.labels}
            loss_val, classok = sess.run([loss,rightclass], feed_dict=feed_val)
            print(loss_val)
            if loss_val < error_val_old:
                error_val_old = loss_val


            champ_mis_val = np.sum(classok) / np.shape(dataset.valid.features)[0];
            if champ_mis_val < champ_mis_old:
                champ_mis_old = champ_mis_val
                champ_mis = np.sum(b) / np.shape(dataset.test.features)[0];
                print('NEW CHAMPION SAVING')#+%f GOOD = %d' % (loss_val, classok))
                pred,lab = sess.run([out_res, lab_res], feed_dict=feed_test)
quantify(pred[:,0], lab)
                #saved_path = saver.save(sess, OutName[0][0:-4] + 'best_binaryInput')

## gen set for space
#cated = list()
#cated = [np.linspace(np.min(dataset.test.features[:,i]),np.max(dataset.test.features[:,i]),3) for i in range(0, dataset.test.features.shape[1])]
#f = np.meshgrid(*cated)
#feat_domain = np.c_()
#feat_domain=
#feed_domain = {features: feat_domain, labels: lab_domain}






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