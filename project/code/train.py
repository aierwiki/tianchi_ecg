import numpy as np
import pandas as pd
import glob
import math
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, MaxPooling1D, Dropout, Dense, Activation, Flatten, Embedding
from keras.layers.core import Lambda
from keras import backend as K
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import regularizers
import os


ARRYTHMIA_FILE_PATH = "../data/hf_round1_arrythmia.txt"
LABEL_FILE_PATH = "../data/hf_round1_label.txt"
ECG_FILE_PATH = "../data/train"
LABEL_FILE_PATH_TEST_A = "../data/hefei_round1_ansA_20191008.txt" 
ECG_FILE_PATH_TEST_A = "../data/testA"
LABEL_FILE_PATH_TEST_B = "../data/hf_round1_subB_noDup_rename.txt" 
ECG_FILE_PATH_TEST_B = "../data/testB_noDup_rename"
MODEL_SAVE_PATH = '../user_data/'


def load_arrythmia(arrythmia_file):
	df_arrythmia = pd.read_csv(arrythmia_file, header = None)
	df_arrythmia.columns = ["Arrythmia-Type"]
	df_arrythmia = df_arrythmia.reset_index().set_index('Arrythmia-Type')
	return df_arrythmia


def cast_str_to_int(age_str):
	if len(age_str.strip()) == 0:
		return 0
	else:
		return int(age_str)
	
	
def load_label(label_file, test_flag = False):
	df_label = pd.read_csv(label_file, header = None)
	df_label.columns = ['Origin_Data']
	df_label['EEG-File'] = df_label['Origin_Data'].map(lambda x : x.split('\t')[0])
	df_label['Age'] = df_label['Origin_Data'].map(lambda x : cast_str_to_int(x.split('\t')[1]))
	df_label['Gender'] = df_label['Origin_Data'].map(lambda x : 'UNKNOWN' if x.split('\t')[2] == "" else x.split('\t')[2])
	if not test_flag:
		df_label['Arrythmia-Types'] = df_label['Origin_Data'].map(lambda x : x.strip().split('\t')[3:])
	return df_label.drop('Origin_Data', axis = 1)


def load_electrocarddiogram(eeg_path):
	df_eeg = pd.DataFrame(columns = ['EEG-File', 'I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'])
	txt_files = glob.glob(os.path.join(eeg_path, "*.txt"))
	for file in txt_files:
		pd_one_eeg = pd.read_csv(file, sep = ' ')
		pd_one_row = pd.DataFrame(columns = df_eeg.columns, index = [0])
		pd_one_row['EEG-File'] = os.path.split(file)[-1]
		for col in df_eeg.columns:
			if col is not 'EEG-File':
				pd_one_row.loc[0, col] = pd_one_eeg[col].values.tolist()
		df_eeg = df_eeg.append(pd_one_row)
	return df_eeg.reset_index(drop = True)


def trans_label(df_label, df_arrythmia):
	df_label['Arrythmia-Types'] = df_label['Arrythmia-Types'].map(lambda x : [df_arrythmia.loc[xx][0] for xx in x])
	return df_label


def load_data(arrythmia_file, label_file, eeg_path, test_flag = False):
	df_arrythmia = load_arrythmia(arrythmia_file)
	df_label = load_label(label_file, test_flag = test_flag)
	df_eeg = load_electrocarddiogram(eeg_path)

	if not test_flag:
		df_label = trans_label(df_label, df_arrythmia)
	
	df_eeg_label = pd.merge(left = df_label, right = df_eeg, on = 'EEG-File')
	
	return df_eeg_label, df_arrythmia


def flat_arrythmia_type(df_label, type_num):
	for i in range(type_num):
		df_label['Arrythmia-Types-' + str(i)] = df_label['Arrythmia-Types'].map(lambda x : 1 if i in x else 0)
	return df_label


def get_train_data_from_label_eeg(df_eeg_label, test_flag = False):
	if not test_flag:
		df_eeg_label = flat_arrythmia_type(df_eeg_label, 55)
	df_eeg_label['Age'] = df_eeg_label['Age'].map(lambda x : 8 if x >= 80 else x // 10)
	df_eeg_label['Gender'] = df_eeg_label['Gender'].map(lambda x : {'FEMALE' : 0, 'MALE' : 1, 'UNKNOWN' : 2}[x])

	cols = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
	if not test_flag:
		y = df_eeg_label.filter(regex = 'Arrythmia-Types-').values
	X = np.array(df_eeg_label[cols].values.tolist())
	X = X.swapaxes(1, 2)
	X = np.concatenate([X, np.zeros((X.shape[0], math.ceil(X.shape[1] / 256) * 256 - X.shape[1], X.shape[2]))], axis = 1)
	X_age = df_eeg_label[['Age']].values
	X_gender = df_eeg_label[['Gender']].values

	if test_flag:
		return X, X_age, X_gender
	else:
		return X, X_age, X_gender, y


def load_train_data():
	df_eeg_label, df_arrythmia = load_data(ARRYTHMIA_FILE_PATH, LABEL_FILE_PATH, ECG_FILE_PATH)	
	X, X_age, X_gender, y = get_train_data_from_label_eeg(df_eeg_label)	
	np.save(os.path.join(MODEL_SAVE_PATH, 'X.npy'), X)
	np.save(os.path.join(MODEL_SAVE_PATH, 'X_age.npy'), X_age)
	np.save(os.path.join(MODEL_SAVE_PATH, 'X_gender.npy'), X_gender)
	np.save(os.path.join(MODEL_SAVE_PATH, 'y.npy'), y)
	return X, X_age, X_gender, y


def load_test_a_data():
	df_eeg_label, df_arrythmia = load_data(ARRYTHMIA_FILE_PATH, LABEL_FILE_PATH_TEST_A, ECG_FILE_PATH_TEST_A)	
	X, X_age, X_gender, y = get_train_data_from_label_eeg(df_eeg_label)	
	np.save(os.path.join(MODEL_SAVE_PATH, 'X_testA.npy'), X)
	np.save(os.path.join(MODEL_SAVE_PATH, 'X_age_testA.npy'), X_age)
	np.save(os.path.join(MODEL_SAVE_PATH, 'X_gender_testA.npy'), X_gender)
	np.save(os.path.join(MODEL_SAVE_PATH, 'y_testA.npy'), y)
	return X, X_age, X_gender, y


def zeropad(x):
	y = K.zeros_like(x)
	return K.concatenate([x, y], axis = 2)


def zeropad_output_shape(input_shape):
	shape = list(input_shape)
	assert len(shape) == 3
	shape[2] *= 2
	return tuple(shape)


def resnet_block(layer, num_filters, subsample_length, block_index, conv_increase_channels_at, conv_num_skip):
	shortcut = MaxPooling1D(pool_size = subsample_length)(layer)
	zero_pad = (block_index % conv_increase_channels_at) == 0 and block_index > 0
	
	if zero_pad is True:
		shortcut = Lambda(zeropad, output_shape = zeropad_output_shape)(shortcut)
		
	for i in range(conv_num_skip):
		if not (block_index == 0 and i == 0):
			layer = BatchNormalization()(layer)
			layer = Activation('relu')(layer)
			layer = Dropout(0.2)(layer)
			
		layer = Conv1D(filters = num_filters, kernel_size = 16, strides = subsample_length if i == 0 else 1,
					  padding = 'same',
					  kernel_initializer = 'he_normal')(layer)
	layer = Add()([shortcut, layer])
	return layer


def calc_recall_score(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def calc_precision_score(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision


def calc_f1_score(y_true, y_pred):
	precision = calc_precision_score(y_true, y_pred)
	recall = calc_recall_score(y_true, y_pred)
	return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def get_focal_loss(alpha = 0.25, gamma = 2):
	def focal_loss(y_true, y_pred):
		fl = -alpha * y_true * K.log(y_pred) * ((1.0 - y_pred) ** gamma) - (1.0 - alpha) * (1.0 - y_true) * K.log(1.0 - y_pred) * (y_pred ** gamma)
		fl_sum = K.mean(fl, axis = -1)
		return fl_sum
	return focal_loss


def build_model():
	inputs = Input(shape = [5120, 8], dtype = 'float32', name = 'inputs')
	
	# add resnet layer
	layer = Conv1D(filters = 32, kernel_size = 16, strides = 1, padding = 'same', kernel_initializer = 'he_normal')(inputs)
	layer = BatchNormalization()(layer)
	layer = Activation('relu')(layer)
	
	conv_subsample_lengths = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]
	for index, subsample_length in enumerate(conv_subsample_lengths):
		num_filters = 2 ** (index // 4) * 32
		layer = resnet_block(layer, num_filters, subsample_length, index, 4, 2)
		
	layer = BatchNormalization()(layer)
	layer = Activation('relu')(layer)
	
	layer = Flatten()(layer)
	layer = Dense(32, activation = 'relu')(layer)

	# Age and Gender
	inputs_age = Input(shape = (1,), dtype = 'int32', name = 'age_input')
	layer_age = Embedding(output_dim = 32, input_dim = 9, input_length = 1)(inputs_age)
	layer_age = Flatten()(layer_age)

	inputs_gender = Input(shape = (1,), dtype = 'int32', name = 'gender_input')
	layer_gender = Embedding(output_dim = 32, input_dim = 3, input_length = 1)(inputs_gender)
	layer_gender = Flatten()(layer_gender)
	
	# Concat all layers
	layer = keras.layers.concatenate([layer, layer_age, layer_gender])

	# add output layer
	layer = Dense(32, activation = 'relu', kernel_regularizer = regularizers.l2(0.01))(layer) 
	#layer = Dropout(0.1)(layer)
	outputs = Dense(55, activation = 'sigmoid')(layer)
	
	model = Model(inputs = [inputs, inputs_age, inputs_gender], outputs = [outputs])
	
	optimizer = Adam(lr = 0.001, clipnorm = 1)
	model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = [calc_precision_score, calc_recall_score, calc_f1_score])
	#model.compile(loss = get_focal_loss(0.75, 0.75), optimizer = optimizer, metrics = [calc_precision_score, calc_recall_score, calc_f1_score])
	
	return model
	

def train():
	import time
	start = time.time()
	X, X_age, X_gender, y = load_train_data()
	print('load train data ok!')
	X_testA, X_age_testA, X_gender_testA, y_testA = load_test_a_data()
	print('load test_a data ok!')
	X = np.concatenate([X, X_testA])
	X_age = np.concatenate([X_age, X_age_testA])
	X_gender = np.concatenate([X_gender, X_gender_testA])
	y = np.concatenate([y, y_testA])
	print(X.shape, X_age.shape, X_gender.shape, y.shape)
	end = time.time()
	print("load data : {}".format(end - start))
	X_train, X_dev, X_age_train, X_age_dev, X_gender_train, X_gender_dev, y_train, y_dev = train_test_split(X, X_age, X_gender, y, test_size = 0.15)
	#X_train, X_dev, X_age_train, X_age_dev, X_gender_train, X_gender_dev, y_train, y_dev = train_test_split(X, X_age, X_gender, y, test_size = 0.33, random_state = 31)
	model = build_model()
	stopping = keras.callbacks.EarlyStopping(patience = 8)
	reduce_lr = keras.callbacks.ReduceLROnPlateau(factor = 0.1, patience = 2, min_lr = 0.001 * 0.001)
	checkpointer = keras.callbacks.ModelCheckpoint(
		filepath=os.path.join(MODEL_SAVE_PATH, 'ecg.model'),
		save_best_only=False)
	
	hist = model.fit([X_train, X_age_train, X_gender_train], y_train, batch_size = 32, epochs = 50, 
				 validation_data = [[X_dev, X_age_dev, X_gender_dev], y_dev], callbacks = [reduce_lr, stopping, checkpointer])
	
	print("training model finished!")
	end = time.time()	
	print("train model : {}".format(end - start))

def main():
	train()


if __name__ == '__main__':
	main()
