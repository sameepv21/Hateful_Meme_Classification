"""

Only one API, containing the path parameter with image name and path, and the return value is the predicted label.
Other API is a testing api to see whether the server is running or not.

"""
# Import libraries
from fastapi import FastAPI, File, UploadFile
import os

# Create the app object
app = FastAPI()

# Create a test endpoint
@app.get("/")
def root():
    return {"message": "Hello World"}

# Create a prediction endpoint
@app.post("/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    test_dir = '../data/facebook/test'
    with open(os.path.join(test_dir, file.filename), "wb") as f:
        f.write(contents)
        
    return {"filename": file.filename}