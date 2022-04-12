import pickle
import numpy as np
import subprocess
import matplotlib.pyplot as plt
import tensorflow as tf

# depth = 3
# labels = tf.constant([1, 1, 2])
# sample = tf.one_hot(labels, depth)

# with tf.Session() as sess:
#     print(sess.run(sample))
a = {'alex': [1, 2, 3.4], 'bob': [1.2, 3.1]}
b = {'alex': False, 'bob': True}

c, d = list(a.values()), list(b.values())
print(c)
print(d)

# t = np.arange(-0.9, 10, 0.1)
# y = (1 + t) * np.log(1 + t) - t - 0.5 * (t**2)/(1 + t/3)
# plt.plot(t, y)

# plt.savefig('test.png')
# for start in range(0, 1024, 1024):
#     print(start)
# saver = "./policy/"
# exp_name = "simple"
# command = "mkdir " + saver + exp_name
# error_message = subprocess.call(command, shell=True)
# a = np.array([[1, 2, 3, 4], [4, 5, 6, 7], [7, 8, 9, 10]])
# c = a[:, 3]
# print("c:")
# print(c)
# b = np.delete(a, 3, axis=1)
# print("b")
# print(b)
# b = np.sum(b, axis = 1)
# print("b")
# print(b)

# b = c + 0.1 * b
# print("b")
# print(b)

# a = np.arange(1, 10)
# b = 0.01 / a
# c = (2 / (a + 1))
# d = b ** c
# f = 2.71 ** d - 1
# print(np.exp(2))

# a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# director = "./learning_curves/"
# experiment_name = "simple"
# file_name = director + experiment_name + '_agrewards.pkl'
# with open(file_name, 'wb') as fp:
#     pickle.dump(a, fp)