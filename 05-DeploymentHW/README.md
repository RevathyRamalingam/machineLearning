#This shows how ML models can be deployed in Docker 

It uses uvicorn to package all the python dependencies such as scikit-learn, fastapi
The pipeline(DictVectorizer+Model) is saved and loaded via pickle
There is a python script that loads the pipeline and predicts the lead conversion probability