import requests
import pickle
import logging
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
import numpy as np
import azure.functions as func


def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')
    request_json=req.get_json()
    blob_service_client = BlobServiceClient.from_connection_string("connection-string-for-container")
    clsblob =BlobClient.from_connection_string(conn_str="connection-string-for-container", 
    container_name="mlcontainer", blob_name="classifier.pickle")
    scblob =BlobClient.from_connection_string(conn_str="connection-string-for-container", 
    container_name="mlcontainer", blob_name="sc.pickle")
    downloader_cls=clsblob.download_blob(0)
    downloader_sc=scblob.download_blob(0)
    f1=downloader_cls.readall()
    f2=downloader_sc.readall()

    serverless_classifier=pickle.loads(f1)
    serverless_sc=pickle.loads(f2)
    age=request_json['age']
    salary=request_json['salary']
    pred_proba=serverless_classifier.predict_proba(serverless_sc.transform(np.array([[age,salary]])))[:,1]
    print(pred_proba)
    print(age)
    print(salary)
    return func.HttpResponse(f"The prediction probability is, {pred_proba}. This HTTP triggered function executed successfully.")
