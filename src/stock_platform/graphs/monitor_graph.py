from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_platform.state import MonitoringState


def build_monitor_graph(agents: object):
    graph = StateGraph(MonitoringState)
    graph.add_node("refresh_signals", agents.refresh_signals)
    graph.add_node("load_context", agents.load_context)
    graph.add_node("monitor_industries", agents.monitor_industries)
    graph.add_node("monitor_stocks", agents.monitor_stocks)
    graph.add_node("rescore_quant", agents.rescore_quant)
    graph.add_node("review_thesis", agents.review_thesis)
    graph.add_node("drawdown_risk", agents.drawdown_risk)
    graph.add_node("decide_actions", agents.decide_actions)
    graph.add_node("behavioural_guard", agents.behavioural_guard)
    graph.add_node("replace_feedback", agents.replace_feedback)

    graph.add_edge(START, "refresh_signals")
    graph.add_edge("refresh_signals", "load_context")
    graph.add_edge("load_context", "monitor_industries")
    graph.add_edge("monitor_industries", "monitor_stocks")
    graph.add_edge("monitor_stocks", "rescore_quant")
    graph.add_edge("rescore_quant", "review_thesis")
    graph.add_edge("review_thesis", "drawdown_risk")
    graph.add_edge("drawdown_risk", "decide_actions")
    graph.add_edge("decide_actions", "behavioural_guard")
    graph.add_edge("behavioural_guard", "replace_feedback")
    graph.add_edge("replace_feedback", END)
    return graph.compile()

