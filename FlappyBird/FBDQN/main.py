# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import cv2
import sys
sys.path.append('game/')
import wrapped_flappy_bird as fb

ACTIONS = 2
IMAGE_SIZE = 80

sess = tf.Session()
sess.run(tf.global_variables_initializer())

saver = tf.train.import_meta_graph('./flappy_bird_dqn-8500000.meta')
saver.restore(sess, tf.train.latest_checkpoint('./'))
graph = tf.get_default_graph()

S = graph.get_tensor_by_name('S:0')
Q = graph.get_tensor_by_name('Q/BiasAdd:0')

game_state = fb.GameState()

do_nothing = np.zeros(ACTIONS)
do_nothing[0] = 1
img, reward, terminal = game_state.frame_step(do_nothing)
img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
_, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
S0 = np.stack((img, img, img, img), axis=2)

while True:
    Qv = sess.run(Q, feed_dict={S: [S0]})[0]
    Av = np.zeros(ACTIONS) 
    Av[np.argmax(Qv)] = 1

    img, reward, terminal = game_state.frame_step(Av)
    img = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2GRAY)
    _, img = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    img = np.reshape(img, (IMAGE_SIZE, IMAGE_SIZE, 1))
    S0 = np.append(S0[:, :, 1:], img, axis=2)