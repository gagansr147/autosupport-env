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
        done = reward.score > 0.7
        return self.state, reward, done, {}
