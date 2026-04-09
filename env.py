class AutoSupportEnv:

    def __init__(self, task="easy_query"):
        self.task = task
        self.state = None

    def _get_task_data(self):
        from models import Observation
        from tasks import get_task

        data = get_task(self.task)

        return Observation(
            ticket_id=1,
            customer_query=data["query"],
            sentiment=data["sentiment"],
            category=data["category"],
            history=[]
        )

    def reset(self):
        self.state = self._get_task_data()
        return self.state

    def step(self, action):
        reward = self._calculate_reward(action)

        
        done = reward.score > 0.9

        return self.state, reward, done, {}

    def _calculate_reward(self, action):
        from models import Reward

        score = 0.0
        reason = ""

        query = self.state.customer_query.lower()
        action_type = action.action_type.lower()
        message = action.message.lower()

        
        if "order" in query:
            if action_type == "reply" and "order" in message:
                score = 0.95   
                reason = "Correct order response"
            else:
                score = 0.3
                reason = "Incorrect order handling"

        
        elif "money" in query or "refund" in query:
            if action_type == "reply" and ("sorry" in message or "refund" in message):
                score = 0.92   
                reason = "Handled angry customer correctly"
            else:
                score = 0.5
                reason = "Partial handling"

        
        elif "payment" in query:
            if action_type in ["escalate", "request_info"]:
                score = 0.93   
                reason = "Correct payment handling"
            else:
                score = 0.4
                reason = "Incorrect payment handling"

        
        if score <= 0.0:
            score = 0.1

        return Reward(score=score, reason=reason)
