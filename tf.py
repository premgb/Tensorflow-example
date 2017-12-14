############################################################################################
# Classification problem having 3 o/p classes - Solved using 1 hidden layer Neural Network #
############################################################################################
#
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt 

#generate data - 1000 samples per class
Nclass= 1000
D=2 # dimensionality of input
M=3 # hidden layer size
K=3 # number of classes

X1 = np.random.randn(Nclass,2) + np.array([0,-2.5])
X2 = np.random.randn(Nclass,2) + np.array([2.5,2.5])
X3 = np.random.randn(Nclass,2) + np.array([-2.5,2.5])
X= np.vstack([X1, X2, X3]).astype(np.float32)
Y=np.array([0]*Nclass + [1]*Nclass +[2]*Nclass)

plt.scatter(X[:,0],X[:,1],c=Y,s=100,alpha=0.5)
plt.show()

#create indicator variable for targets
N=len(Y)
T=np.zeros((N,K))
for i in xrange(N):
	T[i,Y[i]]=1

def forward(X,W1,b1,W2,b2):
	Z=tf.nn.sigmoid(tf.matmul(X,W1)+b1)
	#softmax is not returned but activation 
	return tf.matmul(Z,W2)+b2

#create tensorflow placeholders - placeholder for data
tfX = tf.placeholder(tf.float32,shape=[None,D])
tfY = tf.placeholder(tf.float32,shape=[None,K])

#initialize weights
W1=tf.Variable(tf.truncated_normal([D,M]))
b1=tf.Variable(tf.zeros([M]))
W2=tf.Variable(tf.truncated_normal([M,K]))
b2=tf.Variable(tf.zeros([K]))

#output variable -this has no value yet
py_x=forward(tfX,W1,b1,W2,b2)

loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=tfY))
#train funcion
train_op= tf.train.GradientDescentOptimizer(0.5).minimize(loss)
predict_op=tf.argmax(py_x,axis=1)

sess=tf.Session()
sess.run(tf.global_variables_initializer())

for i in xrange(1000):
	#perform gradient descent
	_, cost, pred = sess.run([train_op, loss, predict_op], feed_dict={tfX:X, tfY:T})
	if i%50==0:
		print np.mean(Y==pred)
