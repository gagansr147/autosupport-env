---
title: autosupport-env
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

AutoSupportEnv – Customer Support AI Environment -->Overview ...
AutoSupportEnv – Customer Support AI Environment
-->Overview
AutoSupportEnv is a real-world OpenEnv environment that simulates a customer support system.
It allows AI agents to learn how to handle customer queries such as order tracking, refunds, and payment issues.

-->Motivation
Customer support automation is widely used in industries like e-commerce and banking.
This environment helps evaluate how well AI agents:
respond politely
resolve issues
escalate when needed

-->Observation Space
Observation(
    ticket_id: int,
    customer_query: str,
    sentiment: str,
    category: str,
    history: list[str]
)
-->Action Space
Action(
    action_type: str  # reply, escalate, request_info
    message: str
)
-->Reward Model
Reward(
    score: float (0.0 to 1.0),
    reason: str
)
--Tasks--
->Easy Task – Order Query
Query: “Where is my order?”
Goal: Provide correct order-related response
Expected: polite reply mentioning order

--Medium Task – Angry Customer--
Query: “I WANT MY MONEY BACK!!!”
Goal: Handle emotion + refund
Expected: apology + refund mention

--Hard Task – Payment Issue--
Query: “Payment deducted but failed”
Goal: take correct decision
Expected:
escalate issue OR
request transaction details

--Reward Design--
Correct response → high reward
Partial correctness → partial reward
Wrong response → low reward
Score range: 0.0 to 1.0

--Baseline Agent--
A simple rule-based agent is used:
deterministic (same output every run)
no API dependency
ensures reproducibility

--Sample Output--
easy_query: 1.0
angry_customer: 1.0
payment_issue: 1.0

Average Score: 1.00

--Setup Instructions--
pip install -r requirements.txt
python inference.py

--Docker Usage--
docker build -t autosupport .
docker run autosupport

--OpenEnv Compliance--
step(), reset(), state() implemented
typed models using Pydantic
openenv.yaml included
3 tasks with graders
deterministic scoring
