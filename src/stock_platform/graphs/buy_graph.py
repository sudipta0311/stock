from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from stock_platform.state import BuyState


def build_buy_graph(agents: object):
    graph = StateGraph(BuyState)
    graph.add_node("load_portfolio_gate", agents.load_portfolio_gate)
    graph.add_node("discover_universe", agents.discover_universe)
    graph.add_node("recommend_industries", agents.recommend_industries)
    graph.add_node("generate_candidates", agents.generate_candidates)
    graph.add_node("score_quality", agents.score_quality)
    graph.add_node("filter_risk", agents.filter_risk)
    graph.add_node("shortlist", agents.shortlist)
    graph.add_node("validate_qualitative", agents.validate_qualitative)
    graph.add_node("differentiate_portfolio", agents.differentiate_portfolio)
    graph.add_node("assess_timing", agents.assess_timing)
    graph.add_node("size_positions", agents.size_positions)
    graph.add_node("assess_tax_costs", agents.assess_tax_costs)
    graph.add_node("check_confidence", agents.check_confidence)
    graph.add_node("finalize_recommendation", agents.finalize_recommendation)

    graph.add_edge(START, "load_portfolio_gate")
    graph.add_edge("load_portfolio_gate", "discover_universe")
    graph.add_edge("discover_universe", "recommend_industries")
    graph.add_edge("recommend_industries", "generate_candidates")
    graph.add_edge("generate_candidates", "score_quality")
    graph.add_edge("score_quality", "filter_risk")
    graph.add_edge("filter_risk", "shortlist")
    graph.add_edge("shortlist", "validate_qualitative")
    graph.add_edge("validate_qualitative", "differentiate_portfolio")
    graph.add_edge("differentiate_portfolio", "assess_timing")
    graph.add_edge("assess_timing", "size_positions")
    graph.add_edge("size_positions", "assess_tax_costs")
    graph.add_edge("assess_tax_costs", "check_confidence")
    graph.add_edge("check_confidence", "finalize_recommendation")
    graph.add_edge("finalize_recommendation", END)
    return graph.compile()

