from __future__ import annotations

import json
from typing import Any

from stock_platform.config import AppConfig

# Display names and model labels exposed for UI consumption.
PROVIDER_LABELS: dict[str, str] = {
    "anthropic": "Anthropic Claude",
    "openai": "OpenAI GPT",
}


class PlatformLLM:
    """
    Provider-agnostic tiered LLM service.

    Supported backends
    ──────────────────
    anthropic  Claude Haiku 4.5 (fast) + Claude Sonnet 4.6 (reasoning)
               System prompts are cached via cache_control=ephemeral on the
               Anthropic Messages API — ~90% input token saving on high-volume
               loops such as monitoring (50+ calls/session).

    openai     gpt-4.1-mini (fast) + gpt-4.1 (reasoning)
               Same tier split; prompt caching is handled server-side by
               OpenAI automatically (no explicit flag required).

    Both backends implement the same five public methods so callers are
    provider-agnostic.  All methods return None on failure; callers must
    supply a deterministic fallback string.
    """

    def __init__(self, config: AppConfig, provider: str = "anthropic") -> None:
        if provider not in ("anthropic", "openai"):
            raise ValueError(f"Unknown LLM provider: {provider!r}. Use 'anthropic' or 'openai'.")
        self.config = config
        self.provider = provider
        self._client: Any = None

    # ── Provider metadata ────────────────────────────────────────────────────

    @property
    def provider_label(self) -> str:
        return PROVIDER_LABELS.get(self.provider, self.provider)

    @property
    def enabled(self) -> bool:
        if self.provider == "anthropic":
            return self.config.anthropic_enabled
        return self.config.openai_enabled

    @property
    def _fast_model(self) -> str:
        if self.provider == "openai":
            return self.config.openai_fast_model
        return self.config.llm_fast_model

    @property
    def _smart_model(self) -> str:
        if self.provider == "openai":
            return self.config.openai_reasoning_model
        return self.config.llm_reasoning_model

    def model_info(self) -> dict[str, str]:
        """Return display info for the UI."""
        return {
            "provider": self.provider,
            "label": self.provider_label,
            "fast_model": self._fast_model,
            "reasoning_model": self._smart_model,
            "enabled": self.enabled,
        }

    # ── Internal client & dispatch ───────────────────────────────────────────

    def _client_instance(self) -> Any:
        if self._client is None:
            if self.provider == "anthropic":
                import anthropic  # lazy — only needed when key is present
                self._client = anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            else:
                from openai import OpenAI  # lazy
                self._client = OpenAI(api_key=self.config.openai_api_key)
        return self._client

    def _call(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 400,
        temperature: float = 0.2,
        cache_system: bool = False,
    ) -> str | None:
        if not self.enabled:
            return None
        try:
            if self.provider == "anthropic":
                return self._call_anthropic(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    cache_system=cache_system,
                )
            return self._call_openai(
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception:
            return None

    def _call_anthropic(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
        cache_system: bool,
    ) -> str | None:
        client = self._client_instance()
        system: list[dict[str, Any]] = [{"type": "text", "text": system_prompt}]
        if cache_system:
            # Marks the compiled prompt prefix for 5-minute server-side caching.
            system[0]["cache_control"] = {"type": "ephemeral"}
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return (response.content[0].text or "").strip()

    def _call_openai(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str | None:
        # OpenAI caches prompts server-side automatically — no explicit flag needed.
        client = self._client_instance()
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    # ── FAST TIER ────────────────────────────────────────────────────────────
    # Anthropic: Haiku 4.5 (cached)   OpenAI: gpt-4.1-mini

    def buy_rationale(self, item: dict[str, Any], portfolio_context: dict[str, Any]) -> str | None:
        """
        2-3 sentence portfolio-personalised buy rationale.
        Called once per recommendation (~3-4/session).
        System prompt cached on Anthropic to minimise repeated-call cost.
        """
        source_names = [e.get("instrument_name", "") for e in item.get("fund_attribution", [])][:4]
        system_prompt = (
            "You are an equity recommendation analyst for Indian markets. "
            "Write concise, portfolio-personalized rationales in exactly 2-3 sentences. "
            "Do not use generic disclaimers. Mention portfolio fit, overlap, and current signal context."
        )
        user_prompt = (
            f"Buy rationale for {item['company_name']} ({item['symbol']}), sector: {item['sector']}.\n"
            f"Entry signal: {item['entry_signal']} | Quality score: {item['quality_score']:.2f}\n"
            f"Overlap %: {item['overlap_pct']:.1f}% | Gap reason: {item['gap_reason']}\n"
            f"Already held via funds: {', '.join(source_names) or 'none'}\n"
            f"Portfolio holdings count: {len(portfolio_context.get('normalized_exposure', []))}\n"
            "Explain why this stock fits this specific portfolio in 2-3 sentences."
        )
        return self._call(
            model=self._fast_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=220,
            temperature=0.25,
            cache_system=True,
        )

    def monitoring_rationale(
        self, action_row: dict[str, Any], thesis: dict[str, Any], drawdown: dict[str, Any]
    ) -> str | None:
        """
        1-2 sentence action rationale for a monitored holding.
        Called once per holding (~50/session); system prompt cached on Anthropic.
        """
        system_prompt = (
            "You are a disciplined portfolio monitoring agent for Indian equity portfolios. "
            "Produce a single short rationale that is specific, decisive, and grounded in "
            "thesis status, drawdown, and risk context. Maximum 2 sentences."
        )
        user_prompt = (
            f"Action: {action_row['action']} on {action_row['symbol']}\n"
            f"Computed rationale: {action_row['rationale']}\n"
            f"Thesis status: {thesis['status']} | Sector signal: {thesis['geo_signal_change']}\n"
            f"Drawdown: {drawdown['drawdown_pct']:.1f}% | Alert severity: {drawdown['severity']}"
        )
        return self._call(
            model=self._fast_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=180,
            temperature=0.2,
            cache_system=True,
        )

    # ── REASONING TIER ───────────────────────────────────────────────────────
    # Anthropic: Sonnet 4.6   OpenAI: gpt-4.1

    def industry_reasoning(
        self,
        sectors: list[dict[str, Any]],
        macro_thesis: str,
        portfolio_summary: str,
    ) -> str | None:
        """
        3-4 sentence strategic sector prioritization narrative.
        Called once per buy session.
        """
        system_prompt = (
            "You are a senior portfolio strategist specializing in Indian equity markets. "
            "Given portfolio gaps, signal conviction, and the user's macro thesis, "
            "provide a concise 3-4 sentence explanation of which sectors to prioritize and why. "
            "Be specific about geopolitical, policy, or structural tailwinds."
        )
        sector_lines = "\n".join(
            f"- {s['sector']}: score={s['score']:.2f}, "
            f"market_signal={s.get('market_signal', 'N/A')}, reason={s.get('reason', '')}"
            for s in sectors[:6]
        )
        user_prompt = (
            f"Portfolio summary: {portfolio_summary}\n"
            f"User macro thesis: {macro_thesis or 'Not specified'}\n"
            f"Top ranked sectors:\n{sector_lines}\n"
            "Explain the sector prioritization in 3-4 sentences for the investor."
        )
        return self._call(
            model=self._smart_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=350,
            temperature=0.3,
        )

    def qualitative_analysis(
        self,
        candidate: dict[str, Any],
        news: dict[str, Any],
        signal_context: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        Structured qualitative validation.
        Returns {"approved": bool, "confidence": float, "reasoning": str} or None.
        Temperature 0.1 — reproducible, auditable JSON output.
        """
        system_prompt = (
            "You are an equity research analyst for Indian markets. "
            "Given a stock's news, sentiment, sector signals, and quality score, "
            "decide if it passes qualitative validation. "
            'Respond ONLY as valid JSON: {"approved": true|false, "confidence": 0.0-1.0, "reasoning": "..."}. '
            "No markdown, no prose outside the JSON object."
        )
        user_prompt = (
            f"Stock: {candidate['company_name']} ({candidate['symbol']}), "
            f"sector: {candidate['sector']}\n"
            f"Quality score: {candidate['quality_score']:.2f}\n"
            f"Latest headline: {news.get('headline', 'N/A')}\n"
            f"News sentiment score: {news.get('sentiment_score', 0.0):.2f}\n"
            f"Sector conviction: {signal_context.get('conviction', 'NEUTRAL')}\n"
            "Should this stock pass qualitative validation? Respond in the JSON format specified."
        )
        raw = self._call(
            model=self._smart_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.1,
        )
        if not raw:
            return None
        try:
            result = json.loads(raw)
            return {
                "approved": bool(result.get("approved", False)),
                "confidence": float(result.get("confidence", 0.5)),
                "reasoning": str(result.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    def thesis_review(
        self,
        holding: dict[str, Any],
        quant_score: float,
        sector_signal: dict[str, Any],
        stock_news: dict[str, Any],
    ) -> dict[str, Any] | None:
        """
        INTACT / WEAKENED / BREACHED thesis assessment.
        Returns {"status": str, "reasoning": str} or None.
        Hard rules (STRONG_AVOID, etc.) override the LLM result upstream.
        Temperature 0.1 — deterministic, traceable for compliance.
        """
        system_prompt = (
            "You are a portfolio risk manager for Indian equity portfolios. "
            "Review the investment thesis for a holding given quant score, sector signals, and news. "
            'Respond ONLY as valid JSON: {"status": "INTACT"|"WEAKENED"|"BREACHED", "reasoning": "..."}. '
            "BREACHED means sell immediately regardless of P&L. "
            "WEAKENED means monitor closely. INTACT means thesis still valid. "
            "No markdown, no prose outside the JSON object."
        )
        user_prompt = (
            f"Holding: {holding['symbol']} in sector {holding['sector']}\n"
            f"Current quant score: {quant_score:.2f} (concern threshold: 0.55)\n"
            f"Sector signal conviction: {sector_signal.get('conviction', 'NEUTRAL')}\n"
            f"Recent news headline: {stock_news.get('headline', 'N/A')}\n"
            f"News sentiment: {stock_news.get('sentiment_score', 0.0):.2f}\n"
            "Assess whether the investment thesis is INTACT, WEAKENED, or BREACHED."
        )
        raw = self._call(
            model=self._smart_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.1,
        )
        if not raw:
            return None
        try:
            result = json.loads(raw)
            status = str(result.get("status", "INTACT")).upper()
            if status not in {"INTACT", "WEAKENED", "BREACHED"}:
                status = "INTACT"
            return {
                "status": status,
                "reasoning": str(result.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            return None
