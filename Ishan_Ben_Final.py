
# coding: utf-8

# # Stock Market Prediction
# ## Ben Welkie & Ishan Jain

# In[33]:


import edward as ed
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from edward.models import Categorical, Normal, PointMass
import warnings
from hmmlearn.hmm import GaussianHMM
from statsmodels.tsa import ar_model
warnings.filterwarnings('ignore')

data = pd.read_csv('/Downloads/Google.csv')
data = data['Close']
data = data.as_matrix()
plt.plot(data)
plt.show()

# Format data as percent change
for i in reversed(range(1, data.size)):
    data[i] = round((data[i]-data[i-1])/data[i-1], 2)
data[0] = 0

plt.plot(data)
plt.show()
N = data.size
print("Number of data points: {}".format(N))

timelen = 50
# Chain of stocks for ~1 month
numhidden = 3
# States are increasing stock, decreasing stock, stable stock
numobs = np.unique(data).size
print(np.sort(np.unique(data)))
print(numobs)

p_init = Categorical(probs=tf.fill([numhidden], 1.0 / numhidden))
# Transition Matrix
Trans = tf.nn.softmax(tf.Variable(tf.zeros([numhidden, numhidden])), dim=0)
# Emission Matrix
Emiss = tf.nn.softmax(tf.Variable(tf.zeros([numobs, numhidden])), dim=0)
# HMM model
x = []
y = []
for t in range(timelen):
    x_tmp = x[-1] if x else p_init
    x_i = Categorical(probs=Trans[:, x_tmp])
    y_i = Categorical(probs=Emiss[:, x_i])
    x.append(x_i)
    y.append(y_i)

qf = [Categorical(probs=tf.nn.softmax(tf.Variable(tf.ones(numhidden))))
      for t in range(timelen)]

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    inf_KLqp = ed.KLqp(dict(zip(x, qf)), dict(zip(y, data)))
    inf_KLqp.run(n_iter=5000, n_print=5000/10)
    print(sess.run(Trans))
    print(sess.run(Emiss))

data = np.column_stack(data)
data = np.transpose(data)
print("fitting to HMM and decoding ...")

# Make an HMM instance and execute fit
print(data.shape[0])
model = GaussianHMM(n_components=3, covariance_type="full",
                    n_iter=1000).fit(data)

# Predict the optimal sequence of internal hidden state
hidden_states = model.predict(data)

print("done")

print("Transition matrix")
print(model.transmat_)
print()

print("Means and vars of each hidden state")
for i in range(model.n_components):
    print("{0}th hidden state".format(i))
    print("mean = ", model.means_[i])
    print("var = ", np.diag(model.covars_[i]))
    print()

mu = 0.
beta_true = np.array([0.7, 0.25])
noise_obs = 0.1
T = 128
p = 2
# Generate synthetic data
x_true = np.random.randn(T+1)*noise_obs
for t in range(p, T):
    x_true[t] += beta_true.dot(x_true[t-p:t][::-1])
plt.plot(x_true)
plt.show()

mu = Normal(loc=0.0, scale=10.0)
beta = [Normal(loc=0.0, scale=2.0) for i in range(p)]
noise_proc = tf.constant(0.1)
# InverseGamma(alpha=1.0, beta=1.0)
noise_obs = tf.constant(0.1)
# InverseGamma(alpha=1.0, beta=1.0)

x = [0] * T
for n in range(p):
    x[n] = Normal(loc=mu, scale=10.0)  # fat prior on x
for n in range(p, T):
    mu_ = mu
    for j in range(p):
        mu_ += beta[j] * x[n-j-1]
    x[n] = Normal(loc=mu_, scale=noise_proc)

print("setting up distributions")
qmu = PointMass(params=tf.Variable(0.))
qbeta = [PointMass(params=tf.Variable(0.)) for i in range(p)]
print("constructing inference object")
vdict = {mu: qmu}
vdict.update({b: qb for b, qb in zip(beta, qbeta)})
inference = ed.MAP(vdict, data={xt: xt_true for xt, xt_true in zip(x, x_true)})
print("running inference")
inference.run()

print("parameter estimates:")
print("beta: ", [qb.value().eval() for qb in qbeta])
print("mu: ", qmu.value().eval())

print("setting up variational distributions")
qmu = Normal(loc=tf.Variable(0.), scale=tf.nn.softplus(tf.Variable(0.)))
qbeta = [Normal(loc=tf.Variable(0.), scale=tf.nn.softplus(tf.Variable(0.)))
         for i in range(p)]
print("constructing inference object")
vdict = {mu: qmu}
vdict.update({b: qb for b, qb in zip(beta, qbeta)})
inference_vb = ed.KLqp(vdict, data={xt: xt_true for xt,
                       xt_true in zip(x, x_true)})
print("running inference")
inference_vb.run()

print("parameter estimates:")
for j in range(p):
    print("beta[%d]: " % j, qbeta[j].mean().eval(),)
print("mu: ", qmu.variance().eval())

ar2_sm = ar_model.AR(x_true)
res = ar2_sm.fit(maxlag=2, ic=None, trend='c')

print("statsmodels AR(2) params: ", res.params)
