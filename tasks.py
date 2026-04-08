def get_task(task_name):
    if task_name == "easy_query":
        return {
            "query": "Where is my order?",
            "sentiment": "neutral",
            "category": "order"
        }

    elif task_name == "angry_customer":
        return {
            "query": "I WANT MY MONEY BACK!!!",
            "sentiment": "angry",
            "category": "refund"
        }

    elif task_name == "payment_issue":
        return {
            "query": "Payment deducted but failed",
            "sentiment": "angry",
            "category": "billing"
        }
