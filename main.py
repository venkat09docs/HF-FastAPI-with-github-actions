from transformers import pipeline 
import tensorflow as tf
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

generator = pipeline("text-generation", model="gpt2")

app = FastAPI(
    title= "Fast API for LLM Model",
    description= "A Text Generator API",
    version= "1.0"
)

class Body(BaseModel):
    text: str

@app.get('/')
def welcome():
    return HTMLResponse("<h1> Welcome to LLMOps with GPT2 model V1 </h1>")

@app.get('/generate')
def predict(body: Body):
    results = generator(body.text, max_length=200)
    return results[0]['generated_text']

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=80)






