import os
from env import AutoSupportEnv
from models import Action

try:
    from openai import OpenAI
except:
    OpenAI = None


API_BASE_URL = os.getenv("API_BASE_URL")
API_KEY = os.getenv("API_KEY")


MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

MAX_STEPS = 5


def get_action_from_model(observation):

    
    if OpenAI and API_BASE_URL and API_KEY:
        try:
            client = OpenAI(
                base_url=API_BASE_URL,
                api_key=API_KEY
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
                return Action(action_type="escalate", message=output)
            elif "request" in output:
                return Action(action_type="request_info", message=output)
            else:
                return Action(action_type="reply", message=output)

        except Exception as e:
            print(f"[DEBUG] LLM failed: {e}", flush=True)

    
    query = observation.customer_query.lower()

    if "order" in query:
        return Action(action_type="reply", message="Your order is on the way.")

    elif "money" in query or "refund" in query:
        return Action(action_type="reply", message="Sorry for the inconvenience. Your refund will be processed.")

    elif "payment" in query:
        return Action(action_type="escalate", message="Please share your transaction ID.")

    else:
        return Action(action_type="reply", message="We will check your issue.")


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
