def grade_easy(response):
    return 1.0 if "order" in response.lower() else 0.0


def grade_medium(response):
    score = 0.0
    if "sorry" in response.lower():
        score += 0.5
    if "refund" in response.lower():
        score += 0.5
    return score


def grade_hard(response, action_type):
    score = 0.0
    if "transaction" in response.lower():
        score += 0.4
    if action_type == "escalate":
        score += 0.6
    return score