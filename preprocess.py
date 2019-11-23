import librosa
import tarfile
import soundfile as sf
import io
import numpy as np



genre_ids = {"rock": 0, "reggae": 1, "pop": 2, "metal": 3, "jazz": 4, "hiphop": 5, "disco" : 6, "country": 7, "classical": 8, "blues": 9}

# Preprocessing takes 90 seconds
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
                # Should experiment with sf.read
                # Default number of frames is 661794 (~30 seconds)
                time_series, samplerate = sf.read(buff)
                # Converter: mfcc or chromagram
                # Each converted_mfcc is 20 by 1293, where for 1293 timestamps we get 20 coefficients
                converted_mfcc  = librosa.feature.mfcc(time_series)
                inputs.append(converted_mfcc)
                labels.append(genre_ids[member.name.split('/')[1]])

    tar.close()
    return inputs, labels