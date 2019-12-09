import librosa
import tarfile
import soundfile as sf
import io
import numpy as np
import tensorflow as tf

genre_ids = {"rock": 0, "reggae": 1, "pop": 2, "metal": 3, "jazz": 4, "hiphop": 5, "disco" : 6, "country": 7, "classical": 8, "blues": 9}
total_frames = 661794
divisor = 6
frame_portion = 1.0 / divisor
parsed_frames = int(total_frames * frame_portion)
test_split = 0.8


# Preprocessing takes 90 seconds with all frames
# Takes 30 seconds with reduced frames (5 second windows)
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
                buff = io.BytesIO(f.read())
                for i in range(0, divisor):
                    # Should experiment with sf.read
                    # Default number of frames is 661794 (~30 seconds)
                    # For ~5 seconds, number of frames is 110299
                    time_series, sample_rate = sf.read(buff, frames=parsed_frames, start=(i*parsed_frames))
                    # Converter: mfcc or chromagram
                    # Each converted_mfcc is 20 by num_frames
                    # num_frames is 216 for ~5 seconds and 1293 for ~30 seconds
                    converted_mfcc  = librosa.feature.mfcc(time_series)
                    #print(converted_mfcc.shape)
                    inputs.append(converted_mfcc.flatten())
                    labels.append(genre_ids[member.name.split('/')[1]])
                    
                # # Should experiment with sf.read
                # # Default number of frames is 661794 (~30 seconds)
                # # For ~5 seconds, number of frames is 110299
                # time_series, sample_rate = sf.read(buff, frames=parsed_frames)
                # # Converter: mfcc or chromagram
                # # Each converted_mfcc is 20 by num_frames
                # # num_frames is 216 for ~5 seconds and 1293 for ~30 seconds
                # converted_mfcc  = librosa.feature.mfcc(time_series)
                # #print(converted_mfcc.shape)
                # inputs.append(converted_mfcc.flatten())
                # labels.append(genre_ids[member.name.split('/')[1]])
    tar.close()

    num_points = len(labels)
    split_index = int(test_split * num_points)
    indices = range(0, num_points)

    indices = tf.random.shuffle(indices)

    shuffled_inputs = tf.gather(inputs, indices)
    shuffled_labels = tf.gather(labels, indices)

    return shuffled_inputs[:split_index], shuffled_labels[:split_index], shuffled_inputs[split_index:], shuffled_labels[split_index:]
