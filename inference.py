import os
from openai import OpenAI
from env import AutoSupportEnv
from models import Action

MAX_STEPS = 5

# ✅ Use platform-provided API key
client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_action_from_model(observation):

    prompt = f"""
    You are a professional customer support agent.

    Customer Query: {observation.customer_query}
    Sentiment: {observation.sentiment}

    Decide:
    - action_type (reply / escalate / request_info)
    - message

    Return strictly in format:
    action_type: <type>
    message: <message>
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful customer support assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    output = response.choices[0].message.content.lower()

    # default values
    action_type = "reply"
    message = output

    if "escalate" in output:
        action_type = "escalate"
    elif "request" in output:
        action_type = "request_info"

    return Action(action_type=action_type, message=message)


def run_task(task):
    print(f"[START] task={task}", flush=True)

    env = AutoSupportEnv(task=task)
    obs = env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):
        action = get_action_from_model(obs)

        obs, reward, done, _ = env.step(action)

        print(f"[STEP] step={step+1} reward={reward.score}", flush=True)

        total_reward = max(total_reward, reward.score)

        if done:
            break

    print(f"[END] task={task} score={total_reward} steps={step+1}", flush=True)


def main():
    tasks = ["easy_query", "angry_customer", "payment_issue"]

    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
