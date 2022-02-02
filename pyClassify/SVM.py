#import tensorflow as tf
import numpy as np
import sklearn
from sklearn import svm as sksvm
from sklearn.linear_model import LogisticRegression as logReg
from sklearn.model_selection import GridSearchCV
#import sklearn
#import matplotlib.pyplot as plt
import h5py
from dataClass import *
NORMALISE = True;
LOGNORM=False;

batch_size = 200;
bSize = 200;
nfeat = 16;

def bilog(x):
    return np.multiply(np.sign(x),np.log(np.abs(x)+1))
def log(x):
    return np.log(x)


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
    acc = 1- (np.sum(np.abs(lab-pred))/np.shape(lab)[1])
    print('acc = {}'.format(acc))
    print('prec = {}'.format(prec))
    print('rec = {}'.format(rec))
    print('f1 = {}'.format(f1))
    return acc,prec,rec,f1

def combData(x, test=1):
    if test == 1:

        return np.concatenate((x.valid.features, x.test.features), axis=0),np.concatenate((x.valid.labels, x.test.labels), axis=0)

    else:
        #np.concatenate((x.train.features,x.valid.features,x.test.features),axis=0)
        return np.concatenate((x.train.features,x.valid.features,x.test.features),axis=0),np.concatenate((x.train.labels,x.valid.labels,x.test.labels),axis=0)
# load dataset
loadname = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/JacFD_DTsepWave_coeffs3_706'
#loadname = '/cs/research/medim/gdisciac/SOLUS/example/VICTRE_PARADIGM/CLASSIFICATION_538'

dataset = fullDataset(loadname);



if LOGNORM==True:
    posit = np.abs(np.min(combData(dataset,test=0)[0]))+1;#0.000000001;
    dataset.train._features = np.log(dataset.train.features)
    dataset.valid._features =np.log(dataset.valid.features)
    dataset.test._features=np.log(dataset.test.features)
    #dataset.train._features = bilog(posit+dataset.train.features)
    #dataset.valid._features =bilog(posit+dataset.valid.features)
    #dataset.test._features=bilog(posit+dataset.test.features)

#print(dataset.train.features[0,:])
if NORMALISE==True:
    concd = np.concatenate((dataset.train.features,dataset.valid.features ,dataset.test.features), axis =0 )
    nmean = np.mean(concd, axis=0)
    nstd= np.std( concd, axis=0)
    dataset.train._features = np.divide(dataset.train.features - nmean,nstd)
    dataset.test._features = np.divide(dataset.test.features - nmean,nstd)
    dataset.valid._features = np.divide(dataset.valid.features - nmean,nstd)
#if PCA == True:
#    sklearn.decomposition.PCA

X = dataset.train.features;#combData(dataset)[0]
Y = np.array(dataset.train.labels, dtype=np.int32)#combData(dataset)[1],dtype=np.int32)#

parameters = {'kernel':['rbf','poly','sigmoid'], 'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.25,1.5,2.5,3,4,5,6,7,8,9,10,20,40,50,75,100],'max_iter':[6000]}
#clf = sksvm.SVC(C = 1,gamma='auto', kernel='rbf',probability=False,tol = 0.0000001)#class_weight='balanced'
clf = GridSearchCV(estimator=sksvm.SVC(), param_grid=parameters)
clf.fit(X,np.ravel(Y))

#lgr = logReg(random_state=0,max_iter=300, C=50).fit(X, Y)
parameters = {'C':[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,1.5,2.5,3,4,5,6,7,8,9,10,20,40,50,75,100],'max_iter':[6000]}
lgr = GridSearchCV(estimator=logReg(), param_grid=parameters)
lgr.fit(X,np.ravel(Y))
trmis = np.sum(np.abs(clf.predict(X) -Y.T))
tstmis = np.sum(np.abs(clf.predict(dataset.test.features)-dataset.test.labels.T))
valmis = np.sum(np.abs(clf.predict(dataset.valid.features)-dataset.valid.labels.T))
print('SVM')
print(trmis/X.shape[0])
print(tstmis/dataset.test.labels.shape[0])
print(valmis/dataset.valid.labels.shape[0])
print(1-(tstmis+valmis)/(dataset.valid.labels.shape[0]+dataset.test.labels.shape[0]))

quantify(clf.predict(combData(dataset)[0]),combData(dataset)[1].T)
quantify(lgr.predict(combData(dataset)[0]),combData(dataset)[1].T)

trmis = np.sum(np.abs(lgr.predict(X) -Y.T))
tstmis = np.sum(np.abs(lgr.predict(dataset.test.features)-dataset.test.labels.T))
valmis = np.sum(np.abs(lgr.predict(dataset.valid.features)-dataset.valid.labels.T))

print('Logistic regression')
print(trmis/X.shape[0])
print(tstmis/dataset.test.labels.shape[0])
print(valmis/dataset.valid.labels.shape[0])

print( 1- (tstmis+valmis)/(dataset.valid.labels.shape[0]+dataset.test.labels.shape[0]))

print(dataset.valid.labels.shape[0]+dataset.test.labels.shape[0]+dataset.train.labels.shape[0])

print('Q clf')
quantify(clf.predict(combData(dataset)[0]),combData(dataset)[1].T)
print('Q lgr')
quantify(lgr.predict(combData(dataset)[0]),combData(dataset)[1].T)
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