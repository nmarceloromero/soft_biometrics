# ------------------------- IMPORTS ------------------------- #
import numpy        	 as np
import tensorflow   	 as tf
import matplotlib.pyplot as plt
import os, sys, timeit, getopt, cv2, imutils, pickle

from tqdm 						import tqdm
from lstm_model 				import model
from scipy 						import interp
from sklearn.metrics 			import roc_auc_score
from sklearn.metrics 			import confusion_matrix
from sklearn.metrics 			import classification_report
from sklearn.metrics 			import roc_curve, auc
from sklearn.metrics 			import precision_recall_curve
from binary_evaluation			import Sensitivity, Specificity
from binary_evaluation			import SexSp, Accuracy

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ------------------------- METHODS ------------------------- #
def load_extracted_features(file_name):
	pickle_file = file_name + '.pickle'
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		features = save['features']
		names  	 = save['video_names']
		info 	 = save['info']
		del save
	return features, names, info

# Load pickle.
def load_crossval(file_name):
	pickle_file = file_name + '.pickle'
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_fold = save['train_fold']
		test_fold  = save['test_fold']
		del save
	return train_fold, test_fold

# Labels names.
def labels_names():
	labels_names = []
	labels_names.append('0_Gender')
	labels_names.append('1_Superior')
	labels_names.append('2_Inferior')
	labels_names.append('3_Superior_Color')
	labels_names.append('4_Inferior_Color')
	labels_names.append('5_Hat')
	labels_names.append('6_Object_Hand')
	labels_names.append('7_Backpack')
	labels_names.append('8_Glasses')
	labels_names.append('9_Sunglasses')
	labels_names.append('10_Long_Hair')
	labels_names.append('11_Hair_Color')
	labels_names.append('12_Noise')
	return labels_names

# Labels classes.
def lab_classes():
    return np.array([['Woman', 'Man'], ['Long', 'Short'], ['Long', 'Short'], ['No', 'Yes'], ['No', 'Yes']])

# Notice that this function returns only mini-batches of size `batchsize`.
# If the size of the data is not a multiple of `batchsize`, it will not
# return the last (remaining) mini-batch.
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	assert len(inputs) == len(targets)
	if shuffle:
			indices = np.arange(len(inputs))
			np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx+batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# ------------------------- PARAMETERS ------------------------- #
index_label 	= 0
epochs			= 100
batch_size		= 512

num_sequences 	= 2
n_output 		= 2
num_hidden 		= 512

# ----------------------- READ INPUT PARAMETERS -----------------------#
try:
	opts, args = getopt.getopt(sys.argv[1:],"h:i:n:s:",["help", "indexlabel=", "numhidden=", "sequences="])
except getopt.GetoptError:
	print 'train_lstm_model.py -i <indexlabel> -n <numhidden> -s <sequences>'
	sys.exit(2)

for opt, arg in opts:
	if opt in ("-h", "--help"):
		print 'train_lstm_model.py -i <indexlabel> -n <numhidden> -s <sequences>'
		sys.exit()
	elif opt in ("-i", "--indexlabel"):
		index_label = int(arg)
	elif opt in ("-n", "--numcells"):
		num_hidden = int(arg)
	elif opt in ("-s", "--sequences"):
		num_sequences = int(arg)

# ----------------------- LOAD AND NORMALIZE THE DATASET ----------------------- #
os.system('clear')
print 'Loading dataset...'
features, names, info = load_extracted_features(file_name='features_extracted_with_inception')

# Create a new set containing number of frames that is multiple of num_sequences for each video.
for i in range(len(features)):
	if (len(features[i]) < num_sequences):
		print 'The number of sequences is too large! There is a video with less number of frames'
		sys.exit()
	else:
		rest = len(features[i]) % num_sequences
		if (rest != 0):
			features[i] = features[i][:-rest]
n_features = features[0][0].shape[3]


# Get the name of the label that is going to be used.
label_name = str(labels_names()[index_label])
print '\nSelected attribute: ', label_name

# Load pickle containing the cross-val indexes and labels (you will find an example in the labels_crossval folder).
train_fold, test_fold = load_crossval('./labels_crossval/' + labels_names()[index_label])

# Global evaluation metrics
accuracies 			= []
SexSpses			= []
auc_rocses 			= []
Seses 				= []
Spses 				= []
Y_true 				= [] # Saves labels at the order they were used in the CrossVal.
Y_pred 				= [] # Saves prediction of the samples the order they appeared in the CrossVal.
Y_scores 			= [] # Saves scores of the samples the order they appeared in the CrossVal.
tprs 				= [] # For the ROC Curve in case we use the strategy == 1.
aucs 				= [] # For the ROC Curve in case we use the strategy == 1.
mean_fpr = np.linspace(0, 1, 100) # For the ROC Curve in case we use the strategy == 1.

# Open log file
file_log = open('./logs/' + label_name + '_numhidden' + str(num_hidden) + '_numseq' + str(num_sequences) + '.dat', 'w')
file_log.write('Test Info - Label: ' + label_name + '\n')

# Loop over the 10 folds of the cross-validation.
counter_train = 0
counter_test  = 0
for fold in range(10):

	# Print message.
	print '\nProcessing fold %d...' % fold

	# ----------------- TRAIN SET ----------------- #
	X_train = []
	y_train = []

	while train_fold[counter_train,0] == fold:
		video_name = train_fold[counter_train,1]
		label = train_fold[counter_train,2]
		counter_train += 1

		# Get the list of features for each frame.
		features_video = features[names.index(str(video_name))]

		temp_array = []
		for i in range(len(features_video)):
			temp_array.append(np.asarray(features_video[i]).flatten())
			if ((i+1) % num_sequences == 0):
				X_train.append(temp_array)
				y_train.append(label)
				temp_array = []

		# Update counter and current_fold.
		if (counter_train >= train_fold.shape[0]):
			break

	X_train = np.asarray(X_train)
	y_train = np.asarray(y_train)
	y_train = (np.arange(2) == y_train[:,None]).astype(np.float32) # Labels to one-hot encoding.

	print 'Train dataset shape: ', X_train.shape
	print 'Train labels shape:  ', y_train.shape

	# ---------------------------------- TRAIN THE MODEL ---------------------------------- #
	# Start session for the inception model.
	sess = tf.Session()

	# Load classifier.
	with tf.device('/gpu:0'):
		net = model(num_sequences=num_sequences,n_features=n_features,n_output=n_output,num_cells=num_hidden)
		optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(net['loss'])

	# Init variables.
	sess.run(tf.global_variables_initializer())

	# Train the network.
	print '\nTraining LSTM...'
	tic_total = timeit.default_timer()
	for epoch in range(epochs):
		tic = timeit.default_timer()
		train_loss    = 0.0
		train_acc     = 0.0
		train_batches = 0.0
		print '\nEpoch %d' % (epoch + 1)
		pbar = tqdm(total=X_train.shape[0]/batch_size) # Create progress bar.
		for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
			x, y_ = batch
			sess.run(optimizer, feed_dict={net['x']: x, net['y_']: y_, net['keep_prob']: 0.5})
			train_loss    += sess.run(net['loss'],     feed_dict={net['x']: x, net['y_']: y_, net['keep_prob']: 1.0})
			train_acc     += sess.run(net['accuracy'], feed_dict={net['x']: x, net['y_']: y_, net['keep_prob']: 1.0})
			train_batches += 1.0
			pbar.update(1)
		train_loss = train_loss / train_batches
		train_acc  = train_acc  / train_batches
		toc = timeit.default_timer()
		pbar.close()
		# Print information of the epoch.
		print("\nEpoch {} of {} took {:.3f}s".format(epoch + 1, epochs, toc-tic))
		print("Training loss: {:.6f}".format(train_loss))
		print("Training accuracy: {:.6f} %".format(train_acc * 100))
		if (train_acc * 100 == 100):
			break
	toc_total = timeit.default_timer()

	# ---------------------------------- TEST THE MODEL ---------------------------------- #
	y_true   = []
	y_pred 	 = []
	y_scores = []

	while test_fold[counter_test,0] == fold:
		video_name = test_fold[counter_test,1]
		label = test_fold[counter_test,2]
		counter_test += 1

		# Get the list of features for each frame.
		features_video = features[names.index(str(video_name))]
		X_test = []
		c = 0
		temp_array = []
		for i in range(len(features_video)):
			temp_array.append(np.asarray(features_video[i]).flatten())
			if ((i+1) % num_sequences == 0):
				X_test.append(temp_array)
				temp_array = []

		X_test = np.asarray(X_test)
		scores = []
		for i in range(X_test.shape[0]):
			test = X_test[i,:].reshape([-1, num_sequences, n_features])
			scores.append(sess.run(net['scores'], feed_dict={net['x']: test, net['keep_prob']: 1.0}).reshape(2))
		scores = np.asarray(scores)

		# Save ground truth and predictions for this fold.
		y_true.append(label)
		y_pred.append(np.argmax(np.mean(scores, axis=0)))
		y_scores.append(np.mean(scores, axis=0)[1])

		# Save global ground truth and predictions.
		Y_true.append(label)
		Y_pred.append(np.argmax(np.mean(scores, axis=0)))
		Y_scores.append(np.mean(scores, axis=0)[1])

		if (counter_test >= test_fold.shape[0]):
			break


	# Evaluation for this fold.
	acc 		= Accuracy(np.asarray(y_true), np.asarray(y_pred))
	sexsp 		= SexSp(np.asarray(y_true), np.asarray(y_pred))
	Se 			= Sensitivity(np.asarray(y_true), np.asarray(y_pred))
	Sp 			= Specificity(np.asarray(y_true), np.asarray(y_pred))
	report_test = classification_report(np.asarray(y_true), np.asarray(y_pred))
	conf_matrix = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))

	# Global evaluation.
	accuracies.append(acc)
	SexSpses.append(sexsp)
	Seses.append(Se)
	Spses.append(Sp)

	# Compute ROC curve and area the curve.
	fpr, tpr, thresholds = roc_curve(np.asarray(y_true).ravel(), np.asarray(y_scores).ravel(), pos_label=1)
	tprs.append(interp(mean_fpr, fpr, tpr))
	tprs[-1][0] = 0.0
	roc_auc = auc(fpr, tpr)
	aucs.append(roc_auc)
	plt.figure(0)
	plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.3f)' % (fold, roc_auc))

	# Save results.
	file_log.write('\n# --------------------- Fold: ' + str(fold) + ' --------------------- #')
	file_log.write('\nAccuracy: '     	   + str(acc*100)   + '\n')
	file_log.write('Se: ' + str(Se) + ' - Sp: ' + str(Sp) + ' -> SexSp: ' + str(sexsp) + '\n')
	file_log.write('\nClassification Report' + '\n')
	file_log.write(str(report_test))
	file_log.write('\nConfusion Matrix' + '\n')
	file_log.write(str(conf_matrix) + '\n')

	# For the ROC Curve.
	roc_weighted = roc_auc_score(np.asarray(y_true).ravel(), np.asarray(y_scores).ravel(), average='weighted')
	roc_micro = roc_auc_score(np.asarray(y_true).ravel(), np.asarray(y_scores).ravel(), average='micro')
	roc_macro = roc_auc_score(np.asarray(y_true).ravel(), np.asarray(y_scores).ravel(), average='macro')
	auc_rocses.append(roc_macro)
	file_log.write('\nROC AUC' + '\n')
	file_log.write('Weighted: '	+ str(roc_weighted)	+ '\n')
	file_log.write('Micro: ' 	+ str(roc_micro)  	+ '\n')
	file_log.write('Macro: ' 	+ str(roc_macro)  	+ '\n')

	tf.reset_default_graph()


# For the ROC Curve.
plt.figure(0)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", fontsize=7)
plt.savefig('logs/' + label_name + '_numhidden' + str(num_hidden) +  '_numseq_' + str(num_sequences) + '_roc.eps', format='eps', dpi=1000)

# For the ROC Curve.
plt.figure(1)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='k', alpha=.8)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.3f $\pm$ %0.3f)' % (mean_auc, std_auc), lw=2, alpha=.8)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
plt.xlim([-0.05, 0.55])
plt.ylim([0.45, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", fontsize=7)
plt.savefig('logs/' + label_name + '_numhidden' + str(num_hidden) + '_numseq_' + str(num_sequences) + '_roc_zoomed.eps', format='eps', dpi=1000)

# For the global Precision-Recall curve.
precision, recall, _ = precision_recall_curve(np.asarray(Y_true).ravel(), np.asarray(Y_scores).ravel(), pos_label=1)
plt.figure(2)
plt.step(recall, precision, color='0.4', alpha=0.2, where='post')
plt.fill_between(recall, precision, step='post', alpha=0.2, color='0.6')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.savefig('logs/' + label_name + '_numhidden' + str(num_hidden) + '_numseq_' + str(num_sequences) + '_PR.eps', format='eps', dpi=1000)

# Save statistics of the folds (mean and standard deviation).
avg_accuracy 	= np.mean(np.asarray(accuracies))
stv_accuracy 	= np.std(np.asarray(accuracies))

avg_sexsp 		= np.mean(np.asarray(SexSpses))
stv_sexsp    	= np.std(np.asarray(SexSpses))

avg_se 			= np.mean(np.asarray(Seses))
stv_se		   	= np.std(np.asarray(Seses))

avg_sp 			= np.mean(np.asarray(Spses))
stv_sp		   	= np.std(np.asarray(Spses))

avg_auc_roc  	= np.mean(np.asarray(auc_rocses))
stv_auc_roc  	= np.std(np.asarray(auc_rocses))

final_report 	= classification_report(Y_true, Y_pred)
final_cfmatrix 	= confusion_matrix(Y_true, Y_pred)

file_log.write('\n# --------------------- FINAL RESULTS --------------------- #\n')
file_log.write('\nACCURACY\n')
file_log.write('Mean Score: ' + str(avg_accuracy*100)  + '\n')
file_log.write('Std. Dev.: '  + str(stv_accuracy*100)  + '\n')
file_log.write('\nSE\n')
file_log.write('Mean Score: ' + str(avg_se)  + '\n')
file_log.write('Std. Dev.: '  + str(stv_se)  + '\n')
file_log.write('\nSP\n')
file_log.write('Mean Score: ' + str(avg_sp)  + '\n')
file_log.write('Std. Dev.: '  + str(stv_sp)  + '\n')
file_log.write('\nSExSP\n')
file_log.write('Mean Score: ' + str(avg_sexsp)  + '\n')
file_log.write('Std. Dev.: '  + str(stv_sexsp)  + '\n')

file_log.write('\nROC AUC (Macro Average)\n')
file_log.write('Mean Score: ' + str(avg_auc_roc)  + '\n')
file_log.write('Std. Dev.: '  + str(stv_auc_roc)  + '\n')

file_log.write('\nClassification Report' + '\n')
file_log.write(str(final_report))
file_log.write('\nConfusion Matrix' + '\n')
file_log.write(str(final_cfmatrix))
file_log.close()