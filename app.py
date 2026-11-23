"""Customer Support Bot (Day 7 Challenge: risk review).

A small Flask app. A customer sends a message, the LLM decides on a reply
and an action, and the app carries out the action.
"""
import json
import logging
import os
import time
import uuid

from flask import Flask, jsonify, request

from llm import call_llm

app = Flask(__name__)

# Risk #2 fix: read key from environment instead of hardcoding a secret
API_KEY = os.getenv("LLM_API_KEY", "")

logging.basicConfig(filename="logs/app.log", level=logging.INFO)

ORDERS = {
    "10231": {"status": "Shipped", "amount": 1299},
    "10232": {"status": "Delivered", "amount": 499},
    "10233": {"status": "Processing", "amount": 2499},
}
CUSTOMERS = {
    "ravi.kumar@example.com": {"name": "Ravi Kumar", "phone": "+91 98480 12345", "active": True},
    "priya.s@example.com": {"name": "Priya Sharma", "phone": "+91 99890 67890", "active": True},
}
REFUNDS = []


# ---------- Tools the LLM can use ----------

def get_order_status(order_id):
    order = ORDERS.get(order_id)
    return order["status"] if order else "Order not found"


def issue_refund(order_id, amount):
    REFUNDS.append({"order_id": order_id, "amount": amount})
    return f"Refunded Rs {amount} for order {order_id}"


def delete_customer_account(customer_email, confirmed=False):
    if not confirmed:
        return "Account deletion requires explicit confirmation."
    CUSTOMERS.pop(customer_email, None)
    return f"Deleted account {customer_email}"


TOOLS = {
    "get_order_status": get_order_status,
    "issue_refund": issue_refund,
    "delete_customer_account": delete_customer_account,
}


# ---------- Chat endpoint ----------

@app.post("/chat")
def chat():
    data = request.get_json(silent=True) or {}
    email = data.get("customer_email", "")
    message = data.get("message", "")
    customer = CUSTOMERS.get(email, {})

    prompt = (
        "You are a helpful support agent. Reply in JSON with reply, action and args.\n"
        f"Customer name: {customer.get('name')}\n"
        f"Customer email: {email}\n"
        f"Customer phone: {customer.get('phone')}\n"
        f"Customer message: {message}"
    )

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    text, usage = call_llm(prompt, API_KEY)
    latency_ms = (time.perf_counter() - start_time) * 1000

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = input_tokens + output_tokens
    estimated_cost = total_tokens * 0.0005

    logging.info(
        "request_id=%s model=%s latency_ms=%.2f input_tokens=%d output_tokens=%d cost_inr=%.4f",
        request_id,
        "fake-llm",
        latency_ms,
        input_tokens,
        output_tokens,
        estimated_cost,
    )

    result = json.loads(text)

    action = result.get("action")
    action_result = None
    if action:
        if action not in TOOLS:
            action_result = "Unsupported action"
        else:
            args = dict(result.get("args") or {})
            if action == "delete_customer_account":
                args["customer_email"] = email
                msg_lower = message.lower()
                args["confirmed"] = (
                    "confirm close my account" in msg_lower
                    or "confirm delete my account" in msg_lower
                )
            elif args.get("customer_email") == "from_request":
                args["customer_email"] = email
            action_result = TOOLS[action](**args)

    return jsonify({"reply": result["reply"], "action_result": action_result})


if __name__ == "__main__":
    app.run(debug=True)