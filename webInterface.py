from fastapi import FastAPI, WebSocket
import uvicorn

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.websocket("/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    while True:
        message = await websocket.receive_text()
        # Process the received message
        # Implement your chat logic here
        # You can send messages back to the client using `await websocket.send_text(message)`
        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)