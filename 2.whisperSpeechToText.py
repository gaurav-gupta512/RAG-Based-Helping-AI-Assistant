import whisper
import json
import os

model = whisper.load_model('base')

audioFiles = os.listdir('audios')

for audioFile in audioFiles:
    
    result = model.transcribe(audio=f'audios/{audioFile}',
                              language='English',
                            word_timestamps=False
                            )

    chunks = []
    for segmentData in result['segments']: # segments are audio segments
        # creating chunks where each has the video's name,number,chunk's id,start-time,end-time,and the text + at last the complete text (unsegmented)
        chunks.append({'videoNumber': audioFile.split('.')[0],
                       'videoName' : audioFile.split('.')[1],
                       'id': segmentData['id'],
                    'start': segmentData['start'],
                    'end': segmentData['end'],
                    'text': segmentData['text']
                    })
        
    jsonFileName = f'{audioFile.split(".")[0]}.{audioFile.split(".")[1]}.json'
    jsonFileData = {'chunks' : chunks,',completeText' : result['text']}

    with open(f'audioToJsons/{jsonFileName}','w') as f:
        json.dump(jsonFileData,f)

    print(f'file {jsonFileName} done')