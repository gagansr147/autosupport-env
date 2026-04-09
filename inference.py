import os
from env import AutoSupportEnv
from models import Action

try:
    from openai import OpenAI
except:
    OpenAI = None

API_BASE_URL = os.getenv("API_BASE_URL")
MODEL_NAME = os.getenv("MODEL_NAME")
HF_TOKEN = os.getenv("HF_TOKEN")

MAX_STEPS = 5


def get_action_from_model(observation):

    
    if OpenAI and API_BASE_URL and MODEL_NAME and HF_TOKEN:
        try:
            client = OpenAI(
                base_url=API_BASE_URL,
                api_key=HF_TOKEN
            )

            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a customer support agent."},
                    {"role": "user", "content": observation.customer_query}
                ],
                timeout=5   
            )

            output = response.choices[0].message.content.lower()

            if "escalate" in output:
                return Action("escalate", output)
            elif "request" in output:
                return Action("request_info", output)
            else:
                return Action("reply", output)

        except Exception as e:
            
            print(f"[DEBUG] LLM failed: {e}", flush=True)

    
    query = observation.customer_query.lower()

    if "order" in query:
        return Action("reply", "Your order is on the way.")

    elif "money" in query or "refund" in query:
        return Action("reply", "Sorry for the inconvenience. Your refund will be processed.")

    elif "payment" in query:
        return Action("escalate", "Please share your transaction ID.")

    else:
        return Action("reply", "We will check your issue.")


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
