import os
from env import AutoSupportEnv
from models import Action

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

    return total_reward


def main():
    tasks = ["easy_query", "angry_customer", "payment_issue"]

    for task in tasks:
        run_task(task)


if __name__ == "__main__":
    main()
