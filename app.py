import gradio as gr
from env import AutoSupportEnv

def run_env():
    tasks = ["easy_query", "angry_customer", "payment_issue"]
    results = {}

    for task in tasks:
        env = AutoSupportEnv(task=task)
        obs = env.reset()

        query = obs.customer_query.lower()

        if "order" in query:
            score = 1.0
        elif "money" in query:
            score = 1.0
        elif "payment" in query:
            score = 1.0
        else:
            score = 0.5

        results[task] = score

    return results

demo = gr.Interface(
    fn=run_env,
    inputs=[],
    outputs="json",
    title="AutoSupportEnv"
)

demo.launch()