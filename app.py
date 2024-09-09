### Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import librosa as librosa
import statistics as st
import pickle
import io
import base64
from flask import Flask, request, jsonify, render_template



def predict_genre(a):
    
    ### Taking Audio Sample 
    # a = "believer.wav" #put your audio name here

    yt, sr = librosa.load(a, sr = 22050)
    y, index = librosa.effects.trim(yt)

    length = len(y)
    length_in_sec = length/sr
    n_segments = int(length_in_sec/3)

    y = librosa.util.fix_length(y, n_segments*3*sr)


    # def features(a,b,master_feapinture_list):
        


    master_feature_list = []

    for i in range(n_segments):
        trimmed = librosa.util.fix_length(y, 3*sr*(i+1))
        trimmed = np.flip(trimmed)
        
        trimmed = librosa.util.fix_length(trimmed, 3*sr)
        trimmed = np.flip(trimmed)
        
        #sf.write('stereo_file' + str(i) + '.wav', trimmed, 22050)
        a= trimmed
        b = sr
        # features(trimmed, sr, master_feature_list)
        feature_list = []
        
        feature_list.append("test_audio_part" + str(i) + ".wav")
        feature_list.append(length)
        
        chroma_stft = librosa.feature.chroma_stft(a,b)
        feature_list.append(np.mean(chroma_stft))
        feature_list.append(np.var(chroma_stft))

        rms = librosa.feature.rms(a)
        feature_list.append(np.mean(rms))
        feature_list.append(np.var(rms))

        spectral_centroid = librosa.feature.spectral_centroid(a,b)
        feature_list.append(np.mean(spectral_centroid))
        feature_list.append(np.var(spectral_centroid))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(a,b)
        feature_list.append(np.mean(spectral_bandwidth))
        feature_list.append(np.var(spectral_bandwidth))

        spectral_rolloff = librosa.feature.spectral_rolloff(a,b)
        feature_list.append(np.mean(spectral_rolloff))
        feature_list.append(np.var(spectral_rolloff))

        zero_crossing_rate = librosa.feature.zero_crossing_rate(a)
        feature_list.append(np.mean(zero_crossing_rate))
        feature_list.append(np.var(zero_crossing_rate))
        
        y_harmonic, y_perceptr = librosa.effects.hpss(a)
        feature_list.append(np.mean(y_harmonic))
        feature_list.append(np.var(y_harmonic))
        feature_list.append(np.mean(y_perceptr))
        feature_list.append(np.var(y_perceptr))
        
        onset_env = librosa.onset.onset_strength(y=a, sr=b)
        tempo = librosa.beat.tempo(onset_envelope=onset_env,sr = b)
        feature_list.append(float(tempo))
        
        mfccs = librosa.feature.mfcc(a, sr=b, n_mfcc=20)
        for i in range(20):
            feature_list.append(np.mean(mfccs[i]))
            feature_list.append(np.var(mfccs[i]))
        
        master_feature_list.append(feature_list)
        #feature function till here

    
    headers = ["filename", "length", "chroma_stft_mean", "chroma_stft_var", "rms_mean", "rms_var", "spectral_centroid_mean", "spectral_centroid_var", "spectral_bandwidth_mean", "spectral_bandwidth_var", "rolloff_mean", "rolloff_var", "zero_crossing_rate_mean", "zero_crossing_rate_var", "harmony_mean", "harmony_var", "perceptr_mean", "perceptr_var", "tempo", "mfcc1_mean", "mfcc1_var", "mfcc2_mean", "mfcc2_var", "mfcc3_mean", "mfcc3_var", "mfcc4_mean", "mfcc4_var", "mfcc5_mean", "mfcc5_var", "mfcc6_mean", "mfcc6_var", "mfcc7_mean", "mfcc7_var", "mfcc8_mean", "mfcc8_var", "mfcc9_mean", "mfcc9_var", "mfcc10_mean", "mfcc10_var", "mfcc11_mean", "mfcc11_var", "mfcc12_mean", "mfcc12_var", "mfcc13_mean", "mfcc13_var", "mfcc14_mean", "mfcc14_var", "mfcc15_mean", "mfcc15_var", "mfcc16_mean", "mfcc16_var", "mfcc17_mean", "mfcc17_var", "mfcc18_mean", "mfcc18_var", "mfcc19_mean", "mfcc19_var", "mfcc20_mean", "mfcc20_var" ]
    audio_df = pd.DataFrame(master_feature_list, columns=headers)
    f = audio_df.columns

    ### Normalizing The Audio Feature File 

    with open('Scaler.pkl', 'rb') as handle:
        scaler = pickle.load(handle)

    audio_df_norm = scaler.transform(audio_df.loc[:,f[2:]])
    audio_df_norm = pd.DataFrame(audio_df_norm,columns=f[2:])
    audio_df_norm = scaler.transform(audio_df.loc[:,f[2:]])
    audio_df_norm = pd.DataFrame(audio_df_norm,columns=f[2:])


    ### Predicting through saved Knn Model

    with open('Knn_model.pkl', 'rb') as handle:
        knn = pickle.load(handle)

    y_pred = knn.predict(audio_df_norm)

    with open('label_decode.pkl', 'rb') as handle:
        dic = pickle.load(handle)

    ### Making plot

    genre_name = []
    frequency = []

    for i in range(10):
        frequency.append(list(y_pred).count(i)*10)
        genre_name.append(dic[i])

    genre_label = str(dic[st.mode(y_pred)]).title()
    return genre_label

app = Flask(__name__)

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict_genre', methods=['POST'])
def process_audio():
    # Check if audio file exists
    if 'audio' not in request.files:
        return jsonify(error='No audio file uploaded')

    audio_file = request.files['audio']

    # Check if file is empty
    if audio_file.filename == '':
        return jsonify(error='Empty file uploaded')

    # Call the predict_genre function to get the genre label
    genre_label = predict_genre(audio_file)

    # Prepare the response
    response = {
        'text': f'Predicted Genre: {genre_label}'
    }

    return jsonify(response)


if __name__ == '__main__':
    app.debug = True
    app.run()
