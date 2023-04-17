"""

Only one API, containing the path parameter with image name and path, and the return value is the predicted label.
Other API is a testing api to see whether the server is running or not.

"""

# Import fastapi
from fastapi import FastAPI

# Initialize the app
app = FastAPI()

# Declare root api
@app.get("/")
def root():
    return {"message": "Hello World"}

# Declare the api
@app.get("/predict/{image_path}")
def predict(image_path: str):
    pass