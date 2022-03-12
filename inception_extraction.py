# ------------------------- IMPORTS ------------------------- #
from tqdm 	import tqdm
from scipy 	import misc
import numpy        	 as np
import tensorflow   	 as tf
import SSD_detection  	 as ssd
import os, sys, timeit, getopt, cv2, imutils, pickle

# ------------------------- CONSTANT VALUES ------------------------- #
INCEPTION_MODEL_PATH = './pre_trained_models/inception/classify_image_graph_def.pb'
FRAMES_PATH = './datasets/videos_frames/' # It should contain folders, each of them containing frames ordered by name (e.g. 1.png, 2.png, 3.png, etc).

# ------------------------- FUNCTIONS ------------------------- #
def save_extracted_features(features_dictionary, file_name):
	pickle_file_name = file_name + '.pickle'
	try:		
		f = open(pickle_file_name, 'wb')
		pickle.dump(features_dictionary, f, pickle.HIGHEST_PROTOCOL)
		f.close()
	except Exception as e:
		print('Unable to save data:', e)
		raise

def load_inception_model(graph_name):
	with tf.gfile.FastGFile(graph_name, 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')

def forward_inception(image, tf_session):
	pool_tensor = tf_session.graph.get_tensor_by_name('pool_3:0')
	return tf_session.run(pool_tensor, {'DecodeJpeg:0': image})

# -------------------------------------------------- MAIN -------------------------------------------------- #
if __name__ == '__main__':

	# Start session to use the pre-trained Inception model.
	sess = tf.Session()

	# Load classifier.
	# You can download the model in: https://github.com/taey16/tf/tree/master/imagenet
	print '\nLoading Inception model...'
	load_inception_model(INCEPTION_MODEL_PATH)

	# Load folder with videos.
	features = []
	names 	 = []

	# Extract features from videos contained in a folder.
	# Note that Here we get the feature of a person regardless of whether it is the right person or not (it can be a person from the background for example),
	# so the idea is that the model can automatically identify this as noise and ignore it, giving more weight to the other frames.
	#for i, file_name in enumerate(os.listdir('../Videos/')):
	for _, folder in enumerate(glob(FRAMES_PATH + '*/', recursive=False)):
		
		# Get names of the files inside the folder.
		files = sorted(os.listdir(str(folder) + '/'))

		# Read each frame and extract features.
		for frame in files:
			frame_path = folder + str(frame)
			image = misc.imread(frame_path)
			try:
				features_single_video.append(forward_inception(image=image, tf_session=sess))
			except:
				print f"Error extracting features for frame '{frame_path}', continuing to next frame..."
				continue

		# Save list of names and features.
		names.append(frame_path)
		features.append(features_single_video)


	# Save features and the name of the videos.
	file_name = 'features_extracted_with_inception'
	dict_info = 'The dictionary contains a list of lists for the features, with each element of the first list representing a video '
	dict_info += '(the name of each video is defined in names) and the sublist of each video contains a list of features of each frame of the video.'
	features_dictionary = {'features': features, 'video_names': names, 'info': dict_info}
	save_extracted_features(features_dictionary, file_name)