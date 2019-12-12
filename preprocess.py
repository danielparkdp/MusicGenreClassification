import librosa
import tarfile
import soundfile as sf
import io
import os
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd
import csv

genre_ids = {"rock": 0, "reggae": 1, "pop": 2, "metal": 3, "jazz": 4, "hiphop": 5, "disco" : 6, "country": 7, "classical": 8, "blues": 9}
total_frames = 661794
total_mfcc_frames = 1293
# Given 30 second audio clip, how many datapoints to break up into
divisor = 12
frame_portion = 1.0 / divisor
parsed_frames = int(total_frames * frame_portion)
parsed_mfcc_frames = int(total_mfcc_frames * frame_portion)
test_split = 0.8

def get_rnn_data(file_name):
    header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
    for i in range(1, 21):
        header += f' mfcc{i}'
    header += ' label'
    header = header.split()
    if not (os.path.isfile('data2.csv')):
        file = open('data2.csv', 'w', newline='')
        with file:
            writer = csv.writer(file)
            writer.writerow(header)
        genres = 'blues classical country disco hiphop jazz metal pop reggae rock'.split()
        for g in genres:
            for filename in os.listdir(f'./data/genres/{g}'):
                songname = f'./data/genres/{g}/{filename}'
                y, sr = librosa.load(songname, mono=True, duration=30)
                chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                #print("chroma: ", np.shape(chroma_stft))
                rms = librosa.feature.rms(y=y)
                #print("rms: ", np.shape(rms))
                spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                #print("spec_cent: ", np.shape(spec_cent))
                spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                #print("spec_bw: ", np.shape(spec_bw))
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                #print("rolloff: ", np.shape(rolloff))
                zcr = librosa.feature.zero_crossing_rate(y)
                #print("zcr: ", np.shape(zcr))
                mfcc = librosa.feature.mfcc(y=y, sr=sr)
                lenny = int(len(chroma_stft)/divisor)
                for i in range(divisor):
                    c = chroma_stft[i*lenny:(i+1)*lenny]
                    r =  rms[i*lenny:(i+1)*lenny]
                    # print("chroma: ", chroma_stft[i*lenny:(i+1)*lenny], len(c))
                    # print("rms: ", rms[i*lenny:(i+1)*lenny], len(r))
                    to_append = f'{filename} {np.mean(chroma_stft[i*lenny:(i+1)*lenny])} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                    for e in mfcc:
                        to_append += f' {np.mean(e[i*lenny:(i+1)*lenny])}'
                    to_append += f' {g}'
                    file = open('data2.csv', 'a', newline='')
                    with file:
                        writer = csv.writer(file)
                        writer.writerow(to_append.split())
                # songname = f'./data/genres/{g}/{filename}'
                # y, sr = librosa.load(songname, mono=True, duration=30)
                # chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                # rms = librosa.feature.rms(y=y)
                # spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                # spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                # rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                # zcr = librosa.feature.zero_crossing_rate(y)
                # mfcc = librosa.feature.mfcc(y=y, sr=sr)
                # to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rms)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                # for e in mfcc:
                #     to_append += f' {np.mean(e)}'
                # to_append += f' {g}'
                # file = open('data.csv', 'a', newline='')
                # with file:
                #     writer = csv.writer(file)
                #     writer.writerow(to_append.split())


    # reading dataset from csv

    data = pd.read_csv('data2.csv')
    data.head()

    # Dropping unneccesary columns
    data = data.drop(['filename'],axis=1)
    data = data.drop(['rmse'], axis=1)
    data = data.drop(['spectral_bandwidth'], axis=1)
    data = data.drop(['rolloff'], axis=1)
    data = data.drop(['zero_crossing_rate'], axis=1)
    data = data.drop(['spectral_centroid'], axis=1)
    data.head()

    genre_list = data.iloc[:, -1]
    encoder = LabelEncoder()
    y = encoder.fit_transform(genre_list)
    print(y)

    # normalizing
    scaler = StandardScaler()
    X = scaler.fit_transform(np.array(data.iloc[:, :-1], dtype = float))

    # spliting of dataset into train and test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("x train size: ", X_train.shape)
    print("y train size: ", y_train.shape)
    return X_train, y_train, X_test, y_test




def get_data(file_name):
    '''
    Takes in a file_name: tar file.
    '''
    inputs = []
    labels = []
    tar = tarfile.open(file_name)
    for member in tar.getmembers():
        if ".wav" in member.name:
            f=tar.extractfile(member)
            if f != None:
                # Default number of frames is 661794 (~30 seconds)
                # For ~5 seconds, number of frames is 110299
                # Converter: mfcc or chromagram
                # Each converted_mfcc is 20 by num_frames
                # num_frames is 216 for ~5 seconds and 1293 for ~30 seconds
                buff = io.BytesIO(f.read())
                time_series, sample_rate = sf.read(buff)
                converted_mfcc  = librosa.feature.mfcc(time_series)
                label_id = genre_ids[member.name.split('/')[1]]
                for i in range(divisor):
                    spliced_datapoint = np.average(converted_mfcc[:, (i * parsed_mfcc_frames): ((i + 1) * parsed_mfcc_frames)], axis=1)
                    # cov = np.cov(spliced_datapoint)
                    # print(cov)
                    inputs.append(spliced_datapoint)
                    labels.append(label_id)
    tar.close()

    num_points = len(labels)
    split_index = int(test_split * num_points)
    indices = range(0, num_points)

    indices = tf.random.shuffle(indices)

    shuffled_inputs = tf.gather(inputs, indices)
    shuffled_labels = tf.gather(labels, indices)

    return shuffled_inputs[:split_index], shuffled_labels[:split_index], shuffled_inputs[split_index:], shuffled_labels[split_index:]
