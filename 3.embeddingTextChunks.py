import requests
import pandas as pd
import json
import os
import joblib

# converting text into vectors through bge-m3 model in ollama
def embedText(promptList):
    response = requests.post('http://localhost:11434/api/embed',
                         json={
                             'model' : 'bge-m3',
                             'input' : promptList
                         }).json()

    return response

dataFrameList = []

jsonFiles = os.listdir('audioToJsons')
for jsonFile in jsonFiles:
    with open (f'audioToJsons/{jsonFile}','r') as f:
        jsonFileData = json.load(f)
        for chunkData in jsonFileData['chunks']:
            dataFrameList.append(chunkData)

df = pd.DataFrame(dataFrameList)    

# calling the embedding function to make a new column of text's vectors
df['text-vectors'] = embedText([data['text'] for data in dataFrameList])['embeddings']

# adding new indexes just in case
df = df.reset_index().rename(columns={'index':'chunkId'})

# dumping all of the json files that contain video data (title,name, chunks and their id,timestamps with vectorized texts
joblib.dump(df,'allJsonsWithEmbeddsDataFrame.joblib')
print('joblib file created')