#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @author: Linlifang
# @file: tf_srnn.py
# @time: 18-8-30上午10:35
import datetime

from utils import *
import tensorflow as tf


class SRNNModel(object):
	def __init__(self, num_step=512, num_class=5, num_units=100, embedding=None, srnn=True):
		self.num_step = num_step
		self.num_class = num_class
		self.num_units = num_units
		self.clip = 5
		self.lr = 0.05
		self.x = tf.placeholder(dtype=tf.int32, shape=[None, self.num_step])
		self.y = tf.placeholder(dtype=tf.float32, shape=[None, self.num_class])
		self.keep_prob = tf.placeholder(dtype=tf.float32)

		self.embedding = tf.get_variable(dtype=tf.float32, initializer=embedding, name='embed')
		self.input_embed = tf.nn.embedding_lookup(self.embedding, self.x)

		if srnn:
			self.input_embed = tf.reshape(self.input_embed, [-1, 8, self.input_embed.get_shape()[-1]])
			# batch*64 num_units
			state_first = self.rnn_layer_first()
			self.state_first = tf.reshape(state_first, [-1, 8, self.num_units])

			# batch*8 num_units
			state_second = self.rnn_layer_second()
			self.state_second = tf.reshape(state_second, [-1, 8, self.num_units])
			# batch num_units
			self.state = self.rnn_layer_threed()
		else:
			self.state = self.rnn_layer_first()
		self.output = self.full_connect_layer()

		# 计算准确率
		self.correct_prediction = tf.equal(tf.argmax(self.y, axis=1), tf.argmax(self.output, axis=1))
		self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, 'float'))  # 计算正确率

		# 损失函数 交叉熵出现loss=nan 改成 二次代价函数
		# self.loss = tf.reduce_mean(tf.square(tf.subtract(self.label, self.output)))
		self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.output), reduction_indices=[1]))
		# 优化器
		self.train_op = self.optimize()

	def rnn_layer_first(self):
		with tf.variable_scope("first_layer"):
			cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
			output, (c_state, h_state) = tf.nn.dynamic_rnn(cell, inputs=self.input_embed, dtype=tf.float32)
		return h_state

	def rnn_layer_second(self):
		with tf.variable_scope("second_layer"):
			cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
			output, (c_state, h_state) = tf.nn.dynamic_rnn(cell, inputs=self.state_first, dtype=tf.float32)
		return h_state

	def rnn_layer_threed(self):
		with tf.variable_scope("third_layer"):
			cell = tf.nn.rnn_cell.LSTMCell(num_units=self.num_units)
			cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.keep_prob)
			output, (c_state, h_state) = tf.nn.dynamic_rnn(cell, inputs=self.state_second, dtype=tf.float32)
		return h_state

	def full_connect_layer(self):
		"""
		全连接层
		:return:
		"""
		w = tf.get_variable(name='weight', shape=[self.num_units, self.num_class])
		b = tf.get_variable(name='bias', shape=[self.num_class])

		# output = tf.matmul(self.state, w) + b
		output = tf.nn.softmax(tf.nn.xw_plus_b(self.state, w, b))
		return output

	def optimize(self):
		"""
		优化器
		:return:
		"""
		optimize = tf.train.AdagradOptimizer(learning_rate=self.lr)
		param = tf.trainable_variables()
		gradients = tf.gradients(self.loss, param)
		clip_grad, clip_norm = tf.clip_by_global_norm(gradients, self.clip)
		train_op = optimize.apply_gradients(zip(clip_grad, param))
		# global_step = tf.Variable(0, name="global_step", trainable=False)
		# optimizer = tf.train.AdamOptimizer(self.config.lr)
		# grads_and_vars = optimizer.compute_gradients(self.loss)
		# train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
		return train_op

	def train(self):
		X_train = x_train_padded_seqs
		Y_train = y_train
		X_val = x_val_padded_seqs
		Y_val = y_val
		X_test = x_test_padded_seqs
		Y_test = y_test
		batch_size = 100

		config = tf.ConfigProto(allow_soft_placement=True)
		with tf.Session(config=config) as sess:
			sess.run(tf.global_variables_initializer())
			for epoch in range(20):
				start = datetime.datetime.now()
				loss, acc = self.run_epoch(sess, X_train, Y_train, batch_size, True)
				val_loss, val_acc = self.predict(sess, X_val, Y_val, batch_size)
				end = datetime.datetime.now()
				print('epoch\t{}\tloss\t{}\t准确率\t{}\tval 准确率\t{}\t耗时\t{}\n'.format(epoch, loss, acc, val_acc,
																				   end - start))

	def predict(self, sess, data, y, batch_size):
		total_loss = 0
		batch_accuracy = []
		for batch_data, batch_y in self.get_batch(data, y, batch_size):
			feed_dict = self.add_feed_dict(batch_data, batch_y, False)
			loss, correct = sess.run([self.loss, self.correct_prediction], feed_dict)
			batch_accuracy.append(correct)
		acc = metric(batch_accuracy)
		return round(total_loss, 4), acc

	def get_batch(self, data, y, batch_size):
		total_length = data.shape[0]
		num_batch = total_length // batch_size + 1
		for kk in range(num_batch):
			start = kk * batch_size
			end = min((kk + 1) * batch_size, total_length)
			batch_data = data[start:end, :]
			batch_y = y[start:end, :]
			yield batch_data, batch_y

	def run_epoch(self, sess, data, y, batch_size, is_train=True):
		total_loss = 0
		total_batch = len(data) // batch_size + 1
		batch_accuracy = []
		for batch_data, batch_y in self.get_batch(data, y, batch_size):
			feed_dict = self.add_feed_dict(batch_data, batch_y, is_train)
			outputs = [self.train_op, self.loss, self.correct_prediction]
			train_op, loss, correct = sess.run(outputs, feed_dict)
			total_loss += loss / total_batch
			batch_accuracy.append(correct)
		acc = metric(batch_accuracy)
		return round(total_loss, 4), acc

	def add_feed_dict(self, batch_data, batch_y, is_train=False):
		"""
		添加feed_dict
		:param is_train:
		:return:
		"""
		feed_dict = {self.x: batch_data, self.keep_prob: 1.0, self.y: batch_y}
		if is_train:
			feed_dict[self.keep_prob] = 0.5
		return feed_dict


def metric(batch_accuracy):
	total = 0
	correct = 0
	for batch in batch_accuracy:
		(m,) = batch.shape
		total += m
		for i in range(m):
			if batch[i]:
				correct += 1
	return round(correct / total, 4)


def main():
	model = SRNNModel(embedding=embedding_matrix, srnn=False)
	model.train()


if __name__ == '__main__':
	main()
