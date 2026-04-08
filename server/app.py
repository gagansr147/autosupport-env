from fastapi import FastAPI
from env import AutoSupportEnv
from models import Action
import uvicorn

app = FastAPI()
env = AutoSupportEnv()

@app.get("/")
def home():
    return {"message": "AutoSupportEnv is running"}

@app.get("/reset")
@app.post("/reset")
def reset():
    obs = env.reset()
    return {
        "ticket_id": obs.ticket_id,
        "customer_query": obs.customer_query,
        "sentiment": obs.sentiment,
        "category": obs.category,
        "history": obs.history
    }

@app.get("/step")
@app.post("/step")
def step(action: dict = None):
    if action is None:
        action = {
            "action_type": "reply",
            "message": "test"
        }

    act = Action(**action)
    obs, reward, done, info = env.step(act)

    return {
        "observation": {
            "ticket_id": obs.ticket_id,
            "customer_query": obs.customer_query,
            "sentiment": obs.sentiment,
            "category": obs.category,
            "history": obs.history
        },
        "reward": {
            "score": reward.score,
            "reason": reward.reason
        },
        "done": done,
        "info": info
    }

@app.get("/state")
def state():
    return env.state.dict() if env.state else {}


def main():
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)



if __name__ == "__main__":
    main()
