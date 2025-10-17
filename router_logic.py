# router_logic.py
import json
from loguru import logger

ROUTING_TABLE = {
    "profile_change": "account_services",
    "complaint": "billing",
    "faq": "customer_support",
    "transaction_query": "transactions",
    "general": "customer_support",
    "loans": "loan_officer"
}

def rule_based_override(metadata):
    if metadata.get("is_vip"):
        return "priority_support"
    if metadata.get("customer_tier") == "gold":
        return "priority_support"
    return None

def map_intent_to_department(intent_label):
    return ROUTING_TABLE.get(intent_label, "customer_support")

def decide_route(predicted_intent, confidence, metadata):
    override = rule_based_override(metadata)
    if override:
        logger.info(f"Rule override to {override} based on metadata {metadata}")
        return override, "rule_override"
    dept = map_intent_to_department(predicted_intent)
    return dept, "intent_mapping"
