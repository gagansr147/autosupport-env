from models import Observation, Action, Reward
from tasks import get_task
from graders import grade_easy, grade_medium, grade_hard


class AutoSupportEnv:

    def __init__(self, task="easy_query"):
        self.task = task
        self.state = None

    def reset(self):
        self.state = self._get_task_data()
        return self.state

    def _get_task_data(self):   # ✅ INSIDE class
        data = get_task(self.task)

        return Observation(
            ticket_id=1,
            customer_query=data["query"],
            sentiment=data["sentiment"],
            category=data["category"],
            history=[]
        )

    def _calculate_reward(self, action: Action):
        if self.task == "easy_query":
            score = grade_easy(action.message)

        elif self.task == "angry_customer":
            score = grade_medium(action.message)

        else:
            score = grade_hard(action.message, action.action_type)

        return Reward(score=score, reason="calculated")

    def step(self, action: Action):
        reward = self._calculate_reward(action)
        done = reward.score > 0.7
        return self.state, reward, done, {}

    def state(self):
        return self.state