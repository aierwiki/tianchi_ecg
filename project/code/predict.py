import numpy as np
import keras
from train import calc_precision_score, calc_recall_score, calc_f1_score
from train import load_data, get_train_data_from_label_eeg
import os
import pandas as pd


ARRYTHMIA_FILE_PATH = "../data/hf_round1_arrythmia.txt"
LABEL_FILE_PATH_TEST_B = "../data/hf_round1_subB_noDup_rename.txt" 
ECG_FILE_PATH_TEST_B = "../data/hf_round1_testB_noDup_rename"
MODEL_SAVE_PATH = '../user_data/'
RESULT_SAVE_PATH = '../prediction_result/'


def load_test_b_data():
	df_eeg_label, df_arrythmia = load_data(ARRYTHMIA_FILE_PATH, LABEL_FILE_PATH_TEST_B, ECG_FILE_PATH_TEST_B, test_flag = True)	
	X, X_age, X_gender = get_train_data_from_label_eeg(df_eeg_label, test_flag = True)	
	return X, X_age, X_gender 


def load_arrythmia_for_sub(arrythmia_file):
	df_arrythmia = pd.read_csv(arrythmia_file, header = None)
	df_arrythmia.columns = ["Arrythmia-Type"]
	return df_arrythmia


def make_submit_file(y, label_file, arrythmia_file, submit_file):
	df_arrythmia = load_arrythmia_for_sub(arrythmia_file)
	label_lines = []
	with open(label_file) as f:
		label_lines = f.readlines()
		
	assert len(y) == len(label_lines)
	
	new_label_lines = []
	for i in range(len(y)):
		label_idxs = np.argwhere(y[i] == 1).reshape(-1)
		labels = df_arrythmia.iloc[label_idxs]['Arrythmia-Type'].values.tolist()
		new_label_line = '\t'.join([label_lines[i].strip('\n')] + labels)
		new_label_lines.append(new_label_line)
	
	with open(submit_file, 'w') as f:
		f.write('\n'.join(new_label_lines))


def predict():
	X_test, X_age_test, X_gender_test = load_test_b_data()
	model = keras.models.load_model(os.path.join(MODEL_SAVE_PATH, 'ecg.model'), custom_objects = {'calc_precision_score' : calc_precision_score, 'calc_recall_score' : calc_recall_score, 'calc_f1_score' : calc_f1_score})
	pred_sub = model.predict([X_test, X_age_test, X_gender_test])
	make_submit_file(np.round(pred_sub), LABEL_FILE_PATH_TEST_B, ARRYTHMIA_FILE_PATH, os.path.join(RESULT_SAVE_PATH, "result.txt"))
	print("make submit file finished!")
	

def main():
	predict()


if __name__ == '__main__':
	main()
