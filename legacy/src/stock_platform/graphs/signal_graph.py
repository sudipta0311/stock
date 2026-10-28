from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_platform.state import SignalState


def build_signal_graph(agents: object):
    graph = StateGraph(SignalState)
    graph.add_node("collect_geopolitical_signals", agents.collect_geopolitical_signals)
    graph.add_node("collect_policy_signals", agents.collect_policy_signals)
    graph.add_node("collect_flow_sentiment", agents.collect_flow_sentiment)
    graph.add_node("detect_contrarian_signals", agents.detect_contrarian_signals)
    graph.add_node("aggregate_signals", agents.aggregate_signals)

    graph.add_edge(START, "collect_geopolitical_signals")
    graph.add_edge("collect_geopolitical_signals", "collect_policy_signals")
    graph.add_edge("collect_policy_signals", "collect_flow_sentiment")
    graph.add_edge("collect_flow_sentiment", "detect_contrarian_signals")
    graph.add_edge("detect_contrarian_signals", "aggregate_signals")
    graph.add_edge("aggregate_signals", END)
    return graph.compile()

