from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import requests
import joblib

df = joblib.load('allJsonsWithEmbeddsDataFrame.joblib')

# dynamic query
query = input('Ask about the course')

# embedder for query vectorization
def embedText(prompt):
    response = requests.post('http://localhost:11434/api/embeddings',
                         json={
                             'model' : 'bge-m3',
                             'prompt' : prompt
                         }).json()

    return response

promptEmbed = embedText(query)['embedding']

# creating a 2d array of text vectors so that cosine can accept it as an argument
cosineInputOne = np.vstack(df['text-vectors'].values)

#making it a 2d array by putting inside a list ( creating 2nd dimension )
cosineInputTwo = [promptEmbed]

comparision = cosine_similarity(cosineInputOne,cosineInputTwo).flatten()

# getting top 3 values, first we sort args. best one goes to last so we reverse the array then get first 25 indices
topResults = comparision.argsort()[::-1][:25]

# new dataframe to feed information to LLM
newDf = df.loc[topResults]
dataForLLM = newDf[["chunkId","videoNumber","videoName","text","start","end"]].to_json(orient='records')

llmPrompt = f'''
So i have this data about the tkinter course from Bro Code on youtube. i have some necessary data to process a user's prompt and get them the relevant answers. i used cosine similarity to find best chunks of the videos and show them to the user along with timestamps ( start,end ) , video name and video number. the data contains - video number, video name, chunk id, texts fetched for user responses ( top 25 ) . here's the data : {dataForLLM}. use this and get the user best possible result.

this is user's query : {query}. if user asks anything unrelated. tell them its out of your scope. don't ask extra questions anytime.

your response should only contain response to the user and not what you are supposed to do, how you think, etc.
'''
# used the phi3 model (not recommended for production use, as better models can be used for better responses)
def inferLLM(prompt):
    response = requests.post('http://localhost:11434/api/generate',
                        json={
                            'model' : 'phi3',
                            'prompt' : prompt,
                            'stream': False
                        })

    return response.json()

llmResponse = inferLLM(llmPrompt)['response']
print(llmResponse)

