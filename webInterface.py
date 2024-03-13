from flask import Flask, render_template
from flask_socketio import SocketIO, send
from llm import LLM

app = Flask(__name__)
app.config['SECRET'] = 'secret!123'
socketio = SocketIO(app, cors_allowed_origins='*')
llm = LLM()

@socketio.on('userMessage')
def handle_message(message):
    if message != "User connected":
        send(message, broadcast=True)

@socketio.on('messageReceived')
def generate_answer(message):
    answer = llm.generateResponse(message)
    send(answer, broadcast=True)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    socketio.run(app, host='localhost')