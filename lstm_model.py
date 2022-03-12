# ------------------------- IMPORTS ------------------------- #
import numpy 					as np
import tensorflow 				as tf
import tensorflow.contrib.rnn 	as rnn

# ------------------------- MODEL FUNCTIONS ------------------------- #
# Weights and biases
def weight_variable(shape, name): # Init variables with glorot uniform
	return tf.get_variable(name=name, shape=shape, initializer=tf.contrib.layers.xavier_initializer())

def bias_variable(shape, name):
	return tf.Variable(tf.constant(0.1, shape=shape), name=name)

def model(num_sequences, n_features, n_output, num_cells):

	print '\nBuilding model...'

	# ---------------------------------------- Input Layers ---------------------------------------- #
	with tf.name_scope('inputs'):
		x  = tf.placeholder(tf.float32, shape=[None, num_sequences, n_features], name='x')
		y_ = tf.placeholder(tf.float32, shape=[None, n_output], name='y_')
		keep_prob = tf.placeholder(tf.float32, name='keep_prob')
		print 'Input Size: ',  x.get_shape().as_list()
		print 'Output Size: ', y_.get_shape().as_list()

	with tf.name_scope('unstack'):
		unstacked_x = tf.unstack(value=x, num=num_sequences, axis=1, name='unstacked_x')
		print 'Sequences: ', len(unstacked_x)
		print 'Shape of each sequence: ', unstacked_x[-1].get_shape().as_list()

	# ---------------------------------------- Recurrent Layers ---------------------------------------- #
	with tf.name_scope('lstm'):

		# Forward of the LSTM
		with tf.variable_scope('forward_lstm'):
			lstm_cell_fw = rnn.LSTMCell(num_units=num_cells, forget_bias=1.0)
			lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=keep_prob)

		# Backward of the LSTM
		with tf.variable_scope('backward_lstm'):
			lstm_cell_bw = rnn.LSTMCell(num_units=num_cells, forget_bias=1.0)
			lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=keep_prob)

		# The recurrent part
		with tf.variable_scope('rnn'):
			out, out_state_fw, out_state_bw = rnn.static_bidirectional_rnn(	cell_fw=lstm_cell_fw,
																			cell_bw=lstm_cell_bw,
																			inputs=unstacked_x,
																			dtype=tf.float32)

		print 'LSTM Output of each sequence: ',  out[-1].get_shape().as_list()
		print 'out_state_fw: ', out_state_fw[-1].get_shape().as_list()
		print 'out_state_bw: ', out_state_bw[-1].get_shape().as_list()

	# ---------------------------------------- Output Layers ---------------------------------------- #
	with tf.name_scope('out'):
		W = weight_variable([2*num_cells, n_output], name='W')
		b = bias_variable([n_output], name='b')
		y = tf.matmul(out[-1], W) + b
		print 'Final -> Input:', n_features ,'- Output:', n_output

	# ---------------------------------------- Loss ---------------------------------------- #
	# L2 Regularization
	# Add l2 loss only if it is a Weight (name contains W)
	with tf.name_scope('l2_loss'):
		l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'W' in v.name]) * 0.001

	# Loss function (Cross Entropy)
	# tf.nn.softmax_cross_entropy_with_logits expects unscaled logits, since
	# it performs a softmax on logits internally for efficiency. Do not call
	# this op with the output of softmax, as it will produce incorrect results.
	with tf.name_scope('loss'):
		loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) + l2_loss

	# Accuracy
	with tf.name_scope('accuracy'):
		correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# Score (for the ROC Curve)
	with tf.name_scope('scores'):
		scores = tf.nn.softmax(logits=y)

	return {'x': x, 'y_': y_, 'keep_prob': keep_prob, 'scores': scores, 'loss': loss, 'accuracy': accuracy}
