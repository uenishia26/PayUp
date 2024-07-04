import os
import json 
import requests
from google.oauth2 import service_account
from google.cloud import documentai_v1 as documentai
import io #Input/output operations 
import itertools

# Set the environment variable for the service account key
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = './dataentryautomation-793ea24c158f.json'

# Initialize the Document AI client
client = documentai.DocumentProcessorServiceClient()

# Specify the project and location (e.g., 'us' or 'eu')
project_id = 'dataentryautomation'
location = 'us'  # Format is 'us' or 'eu'
processor_id = 'fd575fe82416e643'  # The ID of your processor

# The full resource name of the processor
name = f'projects/{project_id}/locations/{location}/processors/{processor_id}'

# Read the document content
path = './0002.jpg'
with open(path, 'rb') as document_file:
    document_content = document_file.read()

# Configure the process request
request = documentai.ProcessRequest(
    name=name,
    raw_document=documentai.RawDocument(content=document_content, mime_type='image/jpeg')
)

# Process the document
result = client.process_document(request=request)

# Get the document from the result
document = result.document
print(document.entities)

doc_data = {}
items = {}
for entity in document.entities:
    if entity.type_ == 'line_item':
        for pair1, pair2 in itertools.pairwise(entity.properties):
            if pair1.mention_text != 'AMOUNT':
                items[pair1.mention_text] = pair2.mention_text
    else:
        doc_data[entity.type_] = entity.mention_text

# Print the dictionary
print(doc_data)
print(items)
