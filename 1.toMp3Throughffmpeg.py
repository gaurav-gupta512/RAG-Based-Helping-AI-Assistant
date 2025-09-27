import os,subprocess

videoFiles = os.listdir('videos') # fetching video names
for videoFile in videoFiles:
    videoFileNumber = videoFile.split('.')[0] # fetching video number
    videoFileName = videoFile.split('.')[1] # fetching video name
    subprocess.run(['ffmpeg', # running ffmpeg
                    '-i', 
                    f'videos/{videoFile}', # input videos
                    f'audios/{videoFileNumber}.{videoFileName}.mp3' # output audios
                    ])
