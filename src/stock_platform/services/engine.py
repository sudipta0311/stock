from __future__ import annotations

from pathlib import Path
from typing import Any

from stock_platform.agents import BuyAgents, MonitoringAgents, PortfolioAgents, SignalAgents
from stock_platform.config import AppConfig, ensure_data_dir, load_app_env
from stock_platform.data.repository import PlatformRepository
from stock_platform.providers import LiveMarketDataProvider
from stock_platform.services.llm import PlatformLLM
from stock_platform.services.mf_lookup import MutualFundHoldingsClient
from stock_platform.services.pdf_parser import NSDLCASParser


def _sample_portfolio_payload() -> dict[str, Any]:
    return {
        "macro_thesis": "Prefer defence, healthcare exporters, and industrial capex leaders over crowded consumer trades.",
        "investable_surplus": 500000,
        "direct_equity_corpus": 800000,
        "mutual_funds": [
            {"instrument_name": "Axis Bluechip Fund", "market_value": 650000},
            {"instrument_name": "Parag Parikh Flexi Cap", "market_value": 540000},
        ],
        "etfs": [
            {"instrument_name": "Nifty Bees", "market_value": 250000},
        ],
        "direct_equities": [
            {"instrument_name": "HDFC Bank", "symbol": "HDFCBANK", "quantity": 45, "market_value": 75600},
            {"instrument_name": "Titan Company", "symbol": "TITAN", "quantity": 20, "market_value": 73600},
        ],
    }


class PlatformEngine:
    def __init__(self, config: AppConfig | None = None) -> None:
        load_app_env()
        self.config = config or AppConfig()
        ensure_data_dir(self.config)
        self.repo = PlatformRepository(Path(self.config.db_path))
        self.repo.initialize()
        self.mf_holdings = MutualFundHoldingsClient(self.config)
        self.provider = LiveMarketDataProvider(holdings_client=self.mf_holdings)
        # Default LLM uses Anthropic; provider-keyed buy graphs are built on demand.
        self.llm = PlatformLLM(self.config, provider="anthropic")
        self.pdf_parser = NSDLCASParser()
        self._signal_graph = None
        self._portfolio_graph = None
        # Buy graphs are keyed by llm_provider so graphs for both providers
        # can coexist in the same session without being rebuilt each time.
        self._buy_graphs: dict[str, Any] = {}
        self._monitor_graphs: dict[str, Any] = {}

    # ── Graph builders ───────────────────────────────────────────────────────

    def _build_signal_graph(self):
        if self._signal_graph is None:
            from stock_platform.graphs.signal_graph import build_signal_graph
            self._signal_graph = build_signal_graph(SignalAgents(self.repo, self.provider))
        return self._signal_graph

    def _build_portfolio_graph(self):
        if self._portfolio_graph is None:
            from stock_platform.graphs.portfolio_graph import build_portfolio_graph
            self._portfolio_graph = build_portfolio_graph(PortfolioAgents(self.repo, self.provider))
        return self._portfolio_graph

    def _build_buy_graph(self, llm_provider: str = "anthropic"):
        """Return (and lazily build) the buy graph for a given LLM provider."""
        if llm_provider not in self._buy_graphs:
            from stock_platform.graphs.buy_graph import build_buy_graph
            llm = PlatformLLM(self.config, provider=llm_provider)
            self._buy_graphs[llm_provider] = build_buy_graph(
                BuyAgents(self.repo, self.provider, self.config, llm)
            )
        return self._buy_graphs[llm_provider]

    def _build_monitor_graph(self, llm_provider: str = "anthropic"):
        if llm_provider not in self._monitor_graphs:
            from stock_platform.graphs.monitor_graph import build_monitor_graph
            llm = PlatformLLM(self.config, provider=llm_provider)
            agents = MonitoringAgents(self.repo, self.provider, self.config, self.run_signal_refresh, llm)
            self._monitor_graphs[llm_provider] = build_monitor_graph(agents)
        return self._monitor_graphs[llm_provider]

    # ── Public workflows ─────────────────────────────────────────────────────

    def seed_demo_data(self) -> dict[str, Any]:
        payload = _sample_portfolio_payload()
        self.run_signal_refresh(trigger="seed")
        ingestion = self.ingest_portfolio(payload)
        return {"payload": payload, "ingestion": ingestion}

    def run_signal_refresh(self, trigger: str = "manual", macro_thesis: str | None = None) -> dict[str, Any]:
        graph = self._build_signal_graph()
        return graph.invoke({"trigger": trigger, "macro_thesis": macro_thesis or ""})

    def ingest_portfolio(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.run_signal_refresh(trigger="ingestion", macro_thesis=payload.get("macro_thesis", ""))
        graph = self._build_portfolio_graph()
        result = graph.invoke({"payload": payload})
        # Buy and monitoring outputs are portfolio-dependent. Clear them after
        # a successful ingest so the UI cannot show stale actions from an older statement.
        self.repo.clear_recommendations()
        self.repo.clear_monitoring_actions()
        self.repo.set_state("buy_comparison_result", {})
        return result

    def parse_portfolio_pdf(self, pdf_path: str | Path, password: str) -> dict[str, Any]:
        parsed = self.pdf_parser.parse_file(pdf_path, password)
        return parsed.to_payload()

    def run_buy_analysis(
        self, request: dict[str, Any], llm_provider: str = "anthropic"
    ) -> dict[str, Any]:
        """
        Run buy analysis with the specified LLM provider.

        llm_provider: "anthropic" (default) or "openai"
        Falls back to deterministic rationale if the provider key is not configured.
        """
        if not self.repo.list_signals("unified"):
            self.run_signal_refresh(trigger="buy-precheck")
        graph = self._build_buy_graph(llm_provider=llm_provider)
        return graph.invoke({"request": request})

    def run_buy_analysis_comparison(self, request: dict[str, Any]) -> dict[str, Any]:
        """
        Run buy analysis with BOTH providers and return results side-by-side.

        Returns:
          {
            "anthropic": {
                "enabled": bool,
                "recommendations": [...],
                "run_summary": {...},
                "model_info": {...},
                "error": str | None,
            },
            "openai": { ... same shape ... },
          }
        """
        if not self.repo.list_signals("unified"):
            self.run_signal_refresh(trigger="buy-precheck")

        results: dict[str, Any] = {}
        for llm_provider in ("anthropic", "openai"):
            llm = PlatformLLM(self.config, provider=llm_provider)
            model_info = llm.model_info()
            if not llm.enabled:
                results[llm_provider] = {
                    "enabled": False,
                    "recommendations": [],
                    "run_summary": {},
                    "model_info": model_info,
                    "error": f"No API key configured for {model_info['label']}. "
                             f"Set {'ANTHROPIC_API_KEY' if llm_provider == 'anthropic' else 'OPENAI_API_KEY'} "
                             f"in .env to enable.",
                }
                continue
            try:
                result = self.run_buy_analysis(request, llm_provider=llm_provider)
                results[llm_provider] = {
                    "enabled": True,
                    "recommendations": result.get("recommendations", []),
                    "skipped_stocks": result.get("skipped_stocks", []),
                    "run_summary": result.get("run_summary", {}),
                    "model_info": model_info,
                    "error": None,
                }
            except Exception as exc:
                results[llm_provider] = {
                    "enabled": True,
                    "recommendations": [],
                    "skipped_stocks": [],
                    "run_summary": {},
                    "model_info": model_info,
                    "error": str(exc),
                }

        # Merge skipped stocks from both providers (deduplicate by symbol).
        all_skipped: dict[str, dict] = {}
        for provider_key in ("anthropic", "openai"):
            for sk in results.get(provider_key, {}).get("skipped_stocks", []):
                all_skipped.setdefault(sk["symbol"], sk)
        results["skipped_stocks"] = list(all_skipped.values())

        # Build per-stock synthesis when both providers returned recommendations.
        a_recs = results.get("anthropic", {}).get("recommendations", [])
        o_recs = results.get("openai", {}).get("recommendations", [])
        synthesis_map: dict[str, str] = {}
        if a_recs and o_recs:
            a_by_symbol = {r["symbol"]: r for r in a_recs}
            o_by_symbol = {r["symbol"]: r for r in o_recs}
            synth_llm = PlatformLLM(self.config, provider="anthropic")
            for symbol, a_rec in a_by_symbol.items():
                o_rec = o_by_symbol.get(symbol)
                if not o_rec:
                    continue
                a_rationale = a_rec.get("rationale", "")
                o_rationale = o_rec.get("rationale", "")
                synthesis = synth_llm.synthesise_comparison(
                    stock_name=f"{a_rec.get('company_name', symbol)} ({symbol})",
                    anthropic_rationale=a_rationale,
                    openai_rationale=o_rationale,
                )
                if synthesis:
                    synthesis_map[symbol] = synthesis
        results["synthesis"] = synthesis_map
        return results

    def run_monitoring(
        self, request: dict[str, Any] | None = None, llm_provider: str = "anthropic"
    ) -> dict[str, Any]:
        prefs = self.repo.get_state("user_preferences", {})
        prefs["monitoring_runs_today"] = int(prefs.get("monitoring_runs_today", 0)) + 1
        self.repo.set_state("user_preferences", prefs)
        graph = self._build_monitor_graph(llm_provider=llm_provider)
        result = graph.invoke({"request": request or {}})
        last_monitor_run = self.repo.get_state("last_monitor_run", {})
        last_monitor_run["llm_provider"] = llm_provider
        last_monitor_run["llm_label"] = "Anthropic Claude" if llm_provider == "anthropic" else "OpenAI GPT"
        self.repo.set_state("last_monitor_run", last_monitor_run)
        return result

    # ── Watchlist ────────────────────────────────────────────────────────────

    def add_watchlist_stock(
        self, symbol: str, company_name: str, sector: str = "Unknown", note: str = ""
    ) -> None:
        self.repo.upsert_watchlist_stock(symbol, company_name, sector, note)

    def remove_watchlist_stock(self, symbol: str) -> None:
        self.repo.remove_watchlist_stock(symbol)

    def list_watchlist(self) -> list[dict[str, Any]]:
        return self.repo.list_watchlist()

    def get_dashboard_snapshot(self) -> dict[str, Any]:
        portfolio = self.repo.load_portfolio_context()
        return {
            "portfolio": portfolio,
            "recommendations": self.repo.list_recommendations(),
            "buy_comparison_result": self.repo.get_state("buy_comparison_result", {}),
            "monitoring_actions": self.repo.list_monitoring_actions(),
            "run_meta": {
                "buy": self.repo.get_state("last_buy_run", {}),
                "monitoring": self.repo.get_state("last_monitor_run", {}),
            },
            "signals": {
                "geo": self.repo.list_signals("geo"),
                "policy": self.repo.list_signals("policy"),
                "flow": self.repo.list_signals("flow"),
                "contrarian": self.repo.list_signals("contrarian"),
                "unified": self.repo.list_signals("unified"),
            },
            "llm": {
                # Per-provider availability
                "anthropic_enabled": self.config.anthropic_enabled,
                "openai_enabled": self.config.openai_enabled,
                # Anthropic model names
                "anthropic_fast_model": self.config.llm_fast_model,
                "anthropic_reasoning_model": self.config.llm_reasoning_model,
                # OpenAI model names
                "openai_fast_model": self.config.openai_fast_model,
                "openai_reasoning_model": self.config.openai_reasoning_model,
            },
        }
