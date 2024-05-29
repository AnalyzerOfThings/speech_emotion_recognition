import soundfile
import librosa
import numpy as np
import pandas as pd
import glob, os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.manifold import MDS
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.preprocessing import StandardScaler

def extract_feature(file_name, mfcc, chroma, mel, contrast, tonnetz):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        result = np.array([])
        if chroma:
            stft=np.abs(librosa.stft(X))
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=X,
                                               sr=sample_rate,
                                               n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, 
                                                       sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate).T,
                        axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, 
                                                        sr=sample_rate).T,
                               axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), 
                                                      sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))    
    return result

emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

observed_emotions={'fearful', 'happy', 'neutral', 'disgust',
                   'sad','angry','surprised'}

def load_data(test_size=0.2):
    x,y=[],[]
    for file in glob.glob('/home/karn/Desktop/ravdess-data/Actor_*/*.wav'):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file, 
                                  mfcc=True, 
                                  chroma=True, 
                                  mel=True,
                                  contrast=True,
                                  tonnetz=True)
        x.append(feature)
        y.append(emotion)
    return x,y

def is_active(data):
    ACTIVE_EMOTIONS = {'angry', 'fear', 'happy', 'surprised'}
    emotion_list=[]
    for i in range(len(data)):
        if data[97].iloc[i] in ACTIVE_EMOTIONS: # 180 contains labels
            emotion_list.append('active')
        else:
            emotion_list.append('passive')
    data['is_active'] = emotion_list
    return data

def is_emotion(emotion, data):
    emotion_list=[]
    for i in range(len(data)):
        if data[97].iloc[i]==emotion: 
            emotion_list.append(emotion)
        else:
            emotion_list.append('not '+emotion)
    data['is_'+emotion] = emotion_list
    return data

def this_or_that(this, that, data):
    emotion_list = []
    for i in range(len(data)):
        if data[97].iloc[i] == this:
            emotion_list.append(this)
        elif data[97].iloc[i] == that:
            emotion_list.append(that)
        else:
            emotion_list.append(np.nan)
    data[this+'_or_'+that] = emotion_list
    return data

def train_MLPC(data, target):
    features = [str(i) for i in range(97)]
    param_grid = {'activation':['identity', 'logistic', 'tanh', 'relu'],
                  'solver':['lbfgs', 'sgd', 'adam'],
                  'hidden_layer_sizes':[(i,) for i in range(12)]}
    model = HalvingGridSearchCV(MLPClassifier(max_iter=10000), param_grid, cv=3, 
                         scoring='accuracy')
    x_train, x_test, y_train, y_test = train_test_split(data[features], 
                                                        data[target])
    model.fit(x_train, y_train)
    
    preds = model.predict(x_test)
    return model, confusion_matrix(y_test, preds)

data,y = load_data()
data = MDS(n_components=97).fit_transform(data)
data = pd.DataFrame(data)
y=pd.Series(y)
data = pd.concat([data,y], axis=1, ignore_index=True)


data = is_active(data)
data = is_emotion('fearful',data)
data = is_emotion('angry',data)
data = is_emotion('neutral',data)
data = this_or_that('happy', 'surprised', data)
data = this_or_that('disgust', 'sad', data)

#%%
data.columns = data.columns.astype(str)

features = [str(i) for i in range(97)]
data[features] = StandardScaler().fit_transform(data[features])


is_active_classifier, active_cm = train_MLPC(data, 'is_active')
is_fearful_classifier, fear_cm = train_MLPC(data, 'is_fearful')
is_neutral_classifier, neutral_cm = train_MLPC(data, 'is_neutral')
is_angry_classifier, angry_cm = train_MLPC(data, 'is_angry')
#%%
happy_or_surprise_classifier, hs_cm = train_MLPC(data.loc[data['97'].isin(['happy', 'surprised'])],
                                                'happy_or_surprised')
disgust_or_sad_classifier, ds_cm = train_MLPC(data.loc[data['97'].isin(['disgust', 'sad'])],
                                             'disgust_or_sad')

data['active_preds'] = np.nan * 1248
data['fearful_preds'] = np.nan * 1248
data['angry_preds'] = np.nan * 1248
data['neutral_preds'] = np.nan * 1248
data['happy_surprised_preds'] = np.nan * 1248
data['disgust_sad_preds'] = np.nan * 1248

features = [str(i) for i in range(97)]

active_preds = is_active_classifier.predict(data[features])
data['active_preds'] = active_preds

fearful_preds = is_fearful_classifier.predict(data.loc[data['active_preds']=='active'][features])
data.loc[data['active_preds']=='active','fearful_preds'] = fearful_preds

neutral_preds = is_neutral_classifier.predict(data.loc[data['active_preds']=='passive'][features])
data.loc[data['active_preds']=='passive','neutral_preds'] = neutral_preds

angry_preds = is_angry_classifier.predict(data.loc[data['fearful_preds']=='not fearful'][features])
data.loc[data['fearful_preds']=='not fearful', 'angry_preds'] = angry_preds

happy_surprised_preds = happy_or_surprise_classifier.predict(data.loc[data['angry_preds']=='not angry'][features])
data.loc[data['angry_preds']=='not angry', 'happy_surprised_preds'] = happy_surprised_preds

disgust_sad_preds = disgust_or_sad_classifier.predict(data.loc[data['neutral_preds']=='not neutral'][features])
data.loc[data['neutral_preds']=='not neutral', 'disgust_sad_preds'] = disgust_sad_preds

final_preds = ['' for i in range(1248)]

for i in range(1248):
    if data['fearful_preds'].iloc[i] == 'fearful':
        final_preds[i] = 'fearful'
        continue
    elif data['angry_preds'].iloc[i] == 'angry':
        final_preds[i] = 'angry'
        continue
    elif data['neutral_preds'].iloc[i] == 'neutral':
        final_preds[i] = 'neutral'
        continue
    elif data['disgust_sad_preds'].iloc[i] == 'disgust':
        final_preds[i] = 'disgust'
        continue
    elif data['disgust_sad_preds'].iloc[i] == 'sad':
        final_preds[i] = 'sad'
        continue
    elif data['happy_surprised_preds'].iloc[i] == 'happy':
        final_preds[i] = 'happy'
        continue
    else:
        final_preds[i] = 'surprised'

data['final_preds'] = final_preds
print(accuracy_score(data['97'], data['final_preds']))
