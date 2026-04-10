from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_platform.state import PortfolioState


def build_portfolio_graph(agents: object):
    graph = StateGraph(PortfolioState)
    graph.add_node("capture_user_portfolio", agents.capture_user_portfolio)
    graph.add_node("parse_mutual_funds", agents.parse_mutual_funds)
    graph.add_node("decompose_etfs", agents.decompose_etfs)
    graph.add_node("normalize_exposure", agents.normalize_exposure)
    graph.add_node("compute_overlap", agents.compute_overlap)
    graph.add_node("identify_gaps", agents.identify_gaps)

    graph.add_edge(START, "capture_user_portfolio")
    graph.add_edge("capture_user_portfolio", "parse_mutual_funds")
    graph.add_edge("parse_mutual_funds", "decompose_etfs")
    graph.add_edge("decompose_etfs", "normalize_exposure")
    graph.add_edge("normalize_exposure", "compute_overlap")
    graph.add_edge("compute_overlap", "identify_gaps")
    graph.add_edge("identify_gaps", END)
    return graph.compile()

