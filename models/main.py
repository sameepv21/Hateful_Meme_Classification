"""

Only one API, containing the path parameter with image name and path, and the return value is the predicted label.
Other API is a testing api to see whether the server is running or not.

"""
# Import libraries
from fastapi import FastAPI, File, UploadFile
import os
from inference import main as predict
from fastapi.middleware.cors import CORSMiddleware

# Create the app object
app = FastAPI()

# Enable cors for all origins
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

    print("Saved image and now generating prediction")

    # Generate predictions
    predicted = predict(os.path.join(test_dir, file.filename))

    # Return the predictions
    return {"prediction": predicted}