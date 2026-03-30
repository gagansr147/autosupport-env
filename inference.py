import os
from env import AutoSupportEnv
from models import Action

SYSTEM_PROMPT = """
You are a professional customer support agent.
- Be polite
- Handle angry users with apology
- Mention refund if needed
- Ask for details if required
"""

MAX_STEPS = 5


def get_action_from_model(observation):

    query = observation.customer_query.lower()

    # simple rule-based agent (offline)
    if "order" in query:
        return Action(
            action_type="reply",
            message="Your order is on the way."
        )

    elif "money" in query or "refund" in query:
        return Action(
            action_type="reply",
            message="Sorry for the inconvenience. Your refund will be processed."
        )

    elif "payment" in query:
      return Action(
        action_type="escalate",
        message="We are escalating your issue. Please share your transaction ID."
    )

    else:
        return Action(
            action_type="reply",
            message="We will check your issue."
        )

def run_task(task):
    print(f"\n Running Task: {task}")

    env = AutoSupportEnv(task=task)
    obs = env.reset()

    total_reward = 0

    for step in range(MAX_STEPS):
        action = get_action_from_model(obs)

        obs, reward, done, _ = env.step(action)

        print(f"Step {step+1}")
        print("Action:", action.action_type)
        print("Message:", action.message)
        print("Reward:", reward.score)
        print("Done:", done)

        total_reward = max(total_reward, reward.score)

        if done:
            break

    return total_reward


def main():
    tasks = ["easy_query", "angry_customer", "payment_issue"]

    results = {}

    for task in tasks:
        score = run_task(task)
        results[task] = round(score, 2)

    print("\n Final Results:")
    for k, v in results.items():
        print(f"{k}: {v}")

    avg = sum(results.values()) / len(results)
    print(f"\n Average Score: {avg:.2f}")


if __name__ == "__main__":
    main()