from __future__ import annotations

import json
from typing import Any

from stock_platform.config import AppConfig
from stock_platform.utils.risk_profiles import RISK_PROMPT_HINTS, get_risk_config

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

    openai     gpt-5.4-mini (fast) + gpt-5.4 (reasoning)
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
            max_completion_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def test_openai_connection(self) -> tuple[bool, str]:
        if self.provider != "openai":
            return False, "OpenAI provider not selected"
        if not self.config.openai_enabled:
            return False, "OPENAI_API_KEY missing"
        try:
            import openai
        except Exception as exc:
            return False, f"OpenAI SDK unavailable: {type(exc).__name__}"

        try:
            client = self._client_instance()
            client.chat.completions.create(
                model=self.config.openai_fast_model or "gpt-5.4-mini",
                messages=[{"role": "user", "content": "Say OK"}],
                max_completion_tokens=5,
            )
            return True, "Connected"
        except openai.AuthenticationError:
            return False, "Invalid API key - check OPENAI_API_KEY secret"
        except openai.RateLimitError:
            return False, "Rate limited - wait 60 seconds"
        except Exception as exc:
            return False, f"Connection failed: {type(exc).__name__}"

    # ── FAST TIER ────────────────────────────────────────────────────────────
    # Anthropic: Haiku 4.5 (cached)   OpenAI: gpt-5.4-mini

    def fetch_stock_news_context(self, symbol: str, company_name: str) -> str:
        """
        Fetch top 3 recent news headlines using Anthropic's web_search tool.

        Always uses the Anthropic client regardless of self.provider — web_search
        is Anthropic-only, so both analyst paths (Anthropic + OpenAI) share one
        cached news fetch per stock per session.

        Returns a formatted bullet-point string; "" on failure.
        Results are cached per-instance to avoid repeat API calls within a run.
        """
        if not hasattr(self, "_news_cache"):
            self._news_cache: dict[str, str] = {}
        cache_key = symbol.upper()
        if cache_key in self._news_cache:
            return self._news_cache[cache_key]
        if not self.config.anthropic_enabled:
            self._news_cache[cache_key] = ""
            return ""
        try:
            import anthropic as _anthropic
            # Use existing client when provider is anthropic; create a dedicated
            # one for OpenAI provider so no state leaks between the two.
            client = (
                self._client_instance()
                if self.provider == "anthropic"
                else _anthropic.Anthropic(api_key=self.config.anthropic_api_key)
            )
            news_prompt = (
                f"Search for the 3 most important news headlines about "
                f"{company_name} ({symbol} NSE India) from the last 30 days.\n\n"
                "Focus on:\n"
                "- Earnings results or guidance updates\n"
                "- Order wins or contract announcements\n"
                "- Management changes or analyst rating changes\n"
                "- Regulatory or legal developments\n"
                "- Capacity, expansion, or partnership announcements\n\n"
                "Return ONLY a brief 3-bullet summary:\n"
                "• [Date if known] Headline\n"
                "• [Date if known] Headline\n"
                "• [Date if known] Headline\n\n"
                "If no significant news found in the last 30 days, "
                "return exactly: No material news in last 30 days."
            )
            response = client.messages.create(
                model=self.config.llm_fast_model,  # Haiku — cheap and fast for news
                max_tokens=300,
                tools=[{"type": "web_search_20250305", "name": "web_search"}],
                messages=[{"role": "user", "content": news_prompt}],
            )
            # Extract text blocks only — skip internal tool_use / tool_result blocks
            text_parts = [
                b.text for b in response.content
                if hasattr(b, "text") and b.text
            ]
            result = "\n".join(text_parts).strip() or "No material news in last 30 days."
            print(f"News context [{symbol}]: {result[:100]}...")
            self._news_cache[cache_key] = result
            return result
        except Exception as exc:
            print(f"fetch_stock_news_context failed for {symbol}: {exc}")
            self._news_cache[cache_key] = ""
            return ""

    def buy_rationale(self, item: dict[str, Any], portfolio_context: dict[str, Any]) -> str | None:
        """
        Structured analyst rationale using verified factual snapshot.
        Each provider uses a distinct analytical lens so Compare Both surfaces real disagreement.
          Anthropic — skeptical institutional risk analyst: stress-tests the thesis, surfaces bear case.
          OpenAI    — momentum / catalyst analyst: surfaces near-term catalysts and entry timing.
        Called once per recommendation (~3-4/session).
        System prompt cached on Anthropic to minimise repeated-call cost.
        """
        # ── Build tech-signals dict for snapshot ────────────────────────────
        live_facts = item.get("live_financials", {}) or {}
        price_ctx  = item.get("price_context", {}) or {}
        current_price_val = (
            price_ctx.get("price")
            or item.get("current_price")
            or live_facts.get("currentPrice")
            or live_facts.get("current_price")
            or 0.0
        )
        tech_signals_dict: dict[str, Any] = {}
        try:
            w52_low = float(live_facts.get("week52_low") or live_facts.get("fiftyTwoWeekLow") or 0)
            if w52_low > 0 and current_price_val:
                tech_signals_dict["pct_from_52w_low"] = round(
                    (float(current_price_val) - w52_low) / w52_low * 100, 1
                )
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        try:
            dma_200 = float(live_facts.get("dma_200") or 0)
            if dma_200 > 0 and current_price_val:
                tech_signals_dict["above_200dma"] = float(current_price_val) > dma_200
        except (TypeError, ValueError):
            pass

        # ── Sector gap for snapshot ──────────────────────────────────────────
        sector = item.get("sector", "")
        gaps = {g["sector"]: g for g in portfolio_context.get("identified_gaps", [])}
        sector_gap = gaps.get(sector, {})

        # ── Valuation reliability (pre-computed upstream or absent) ──────────
        val_reliability = item.get("val_reliability") or {}

        # ── Factual snapshot ─────────────────────────────────────────────────
        # Prefer pre-formatted text already placed on item by finalize_recommendation().
        # Fall back to building it here if called from a different path.
        snapshot_text = item.get("factual_snapshot_text") or ""
        if not snapshot_text:
            from stock_platform.utils.stock_context import (
                build_factual_snapshot,
                format_snapshot_for_prompt,
            )
            pe_ctx = item.get("pe_context") or {}
            snap = build_factual_snapshot(
                symbol=item.get("symbol", ""),
                fin_data=live_facts,
                current_price=float(current_price_val),
                pe_context=pe_ctx,
                tech_signals=tech_signals_dict,
                portfolio_overlap=float(item.get("overlap_pct", 0)),
                sector_gap=sector_gap,
            )
            if val_reliability:
                snap["val_reliability"]       = val_reliability.get("label", "")
                snap["val_reliability_note"]  = val_reliability.get("note", "")
                snap["val_reliability_flags"] = val_reliability.get("flags", [])
            snapshot_text = format_snapshot_for_prompt(snap)

        # ── Analyst target for prompt ────────────────────────────────────────
        analyst_target = (
            price_ctx.get("analyst_target")
            or live_facts.get("targetMeanPrice")
            or item.get("analyst_target")
            or 0.0
        )
        try:
            analyst_target = float(analyst_target)
        except (TypeError, ValueError):
            analyst_target = 0.0

        quality_score = item.get("quality_score", 0.0)
        entry_signal  = item.get("entry_signal", "")
        symbol        = item.get("symbol", "")
        target_line   = f"₹{analyst_target:,.0f}" if analyst_target > 0 else "N/A"

        # ── Low-reliability warning (added to both prompts when triggered) ───
        low_val_warning = ""
        if val_reliability.get("label") == "LOW":
            low_val_warning = (
                "\nIMPORTANT: Valuation reliability is LOW for this stock. "
                "Do NOT build your thesis primarily on PE comparison. "
                "Use order book, margin trends, earnings durability, "
                "and cash conversion instead.\n"
            )

        # ── Macro flow block (FII/DII — injected if available) ──────────────
        from stock_platform.utils.fii_dii_fetcher import format_macro_flow_for_prompt
        _macro_flow = item.get("macro_flow") or {}
        macro_flow_block = format_macro_flow_for_prompt(_macro_flow)

        # ── Recent news context (web_search — fetched once per stock per run) ─
        _news_ctx = item.get("news_context", "")
        news_block = (
            f"\n\nRECENT NEWS (last 30 days):\n{_news_ctx}\n"
            if _news_ctx and _news_ctx != "No material news in last 30 days."
            else (
                "\n\nRECENT NEWS: No material news in last 30 days.\n"
                if _news_ctx
                else ""
            )
        )

        # ── Risk profile instruction block ───────────────────────────────────
        risk_profile      = item.get("risk_profile", "Balanced")
        _risk_cfg         = get_risk_config(risk_profile)
        risk_profile_hint = RISK_PROMPT_HINTS.get(risk_profile, RISK_PROMPT_HINTS["Balanced"])
        risk_profile_block = (
            f"\n\nINVESTOR RISK PROFILE INSTRUCTION:\n{risk_profile_hint}\n"
            f"R/R minimum for this profile: {_risk_cfg['min_rr_ratio']}x | "
            f"Staleness cap: {_risk_cfg['staleness_cap_days']} days"
        )

        # ════════════════════════════════════════════════════════════════════
        # ANTHROPIC — Bear-biased risk analyst (quality candidate pool)
        # ════════════════════════════════════════════════════════════════════
        if self.provider == "anthropic":
            _conservative_gate = (
                "\nIMPORTANT: Your job is to SURFACE stocks with a risk assessment, not to PRE-REJECT them. "
                "Even if you believe a stock fails conservative criteria, provide your full bear analysis "
                "and state WHY it fails — do not simply omit it.\n\n"
                "The synthesis layer applies the hard Conservative filters. "
                "You are the risk lens, not the gatekeeper. "
                "Surface every stock in your candidate pool with:\n"
                "1. Your bear case\n"
                "2. The single most likely failure mode\n"
                "3. Whether it passes or fails each conservative criterion explicitly\n"
                "4. Your verdict: ACCUMULATE / WATCHLIST / AVOID\n\n"
                "Do not leave stocks unanalysed — an absent analysis is worse than a negative one.\n"
                if risk_profile == "Conservative" else ""
            )
            system_prompt = (
                "You are a BEAR-BIASED risk analyst reviewing Indian equity recommendations. "
                "You received this stock because it scored highly on QUALITY (ROCE, low debt, "
                "earnings durability) — your job is to find why that quality is NOT enough to buy. "
                "Assume the bull case is already priced in. "
                "You are NOT a general assistant.\n\n"
                "YOUR ANALYSIS MUST ANSWER ALL FOUR:\n"
                "1. What is the single most likely way this thesis fails?\n"
                "2. What does the stock look like if earnings disappoint by 15%?\n"
                "3. Is there a governance, regulatory, or concentration risk the market is ignoring?\n"
                "4. What is the downside price if PE reverts to sector median?\n\n"
                "Only recommend BUY if you cannot construct a credible bear case. "
                "Default to WATCHLIST or AVOID unless the margin of safety is explicit and quantified.\n\n"
                "RULES YOU MUST FOLLOW:\n"
                "1. Start from the factual snapshot only. Do not invent or estimate numbers.\n"
                "2. Clearly label: FACT / DERIVED / MY INFERENCE\n"
                "3. Do not praise quality metrics without stress-testing them.\n"
                "4. For PE comparison: ONLY use it if the PE signal is CHEAP_VS_HISTORY or "
                "EXPENSIVE_VS_HISTORY. If signal is NO_HISTORY or FAIR_VS_HISTORY, do NOT build "
                "a thesis on PE alone.\n"
                "5. Never say 'this is a bargain' without quantifying what happens if the thesis is wrong.\n"
                "6. End with a specific INVALIDATION CONDITION — the single most likely event that "
                "would make you wrong.\n\n"
                f"{_conservative_gate}"
                "OUTPUT STRUCTURE:\n"
                "## RISK VERDICT: [AVOID / WAIT / ACCUMULATE WITH CONDITIONS / BUY]\n\n"
                "### What the facts show\n"
                "[2-3 sentences citing only measured data from snapshot]\n\n"
                "### Bear case\n"
                "[Answer all four questions above: thesis failure, -15% earnings scenario, "
                "governance/regulatory risk, PE reversion downside price]\n\n"
                "### Valuation assessment\n"
                "[Only if PE signal is clear — otherwise say "
                "'valuation unreliable, insufficient PE history']\n\n"
                "### What would make me wrong\n"
                "[Single most important invalidation condition]\n\n"
                "### Risk conclusion\n"
                "[One sentence stance]"
            )
            user_prompt = (
                f"STOCK: {symbol} ({item.get('company_name', '')}) | SECTOR: {sector}\n"
                f"QUALITY SCORE: {quality_score:.2f} | ENTRY SIGNAL: {entry_signal}\n"
                f"ANALYST TARGET: {target_line}\n"
                f"{low_val_warning}"
                f"{risk_profile_block}"
                f"{macro_flow_block}"
                f"{news_block}"
                f"\n{snapshot_text}\n"
                "Stress-test this recommendation and produce the risk-focused verdict."
            )
            return self._call(
                model=self._fast_model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=700,
                temperature=0.35,
                cache_system=True,
            )

        # ════════════════════════════════════════════════════════════════════
        # OPENAI — Bull-biased catalyst analyst (momentum candidate pool)
        # ════════════════════════════════════════════════════════════════════
        system_prompt = (
            "You are a BULL-BIASED catalyst analyst reviewing Indian equity recommendations. "
            "You received this stock because it scored highly on MOMENTUM (revenue acceleration, "
            "technical strength, entry timing) — your job is to find the trigger that re-rates "
            "it upward. "
            "You are NOT a general assistant.\n\n"
            "YOUR ANALYSIS MUST ANSWER ALL FOUR:\n"
            "1. What is the specific near-term catalyst (earnings, order win, policy event) "
            "that closes the valuation gap?\n"
            "2. Why is the market underpricing this catalyst today?\n"
            "3. What does the stock look like in 18 months if the catalyst fires?\n"
            "4. At what price does the risk/reward become compelling?\n\n"
            "Only recommend WATCHLIST or AVOID if you cannot identify a credible catalyst "
            "within the investment horizon.\n\n"
            "RULES YOU MUST FOLLOW:\n"
            "1. Start from the factual snapshot only. Do not invent or estimate numbers.\n"
            "2. Clearly label: FACT / DERIVED / MY INFERENCE\n"
            "3. Name a SPECIFIC catalyst with a time window. "
            "Do NOT say 'continued growth momentum' — that is not a catalyst.\n"
            "4. Identify what the market is specifically underestimating, with evidence.\n"
            "5. State your exit signal clearly — what observable event would make you exit.\n"
            "6. Do NOT build a thesis on PE rerating alone unless there is confirmed earnings acceleration.\n\n"
            "OUTPUT STRUCTURE:\n"
            "## CATALYST VERDICT: [AVOID / WATCHLIST / ACCUMULATE / BUY NOW]\n\n"
            "### What the facts show\n"
            "[2-3 sentences citing only measured data]\n\n"
            "### Bull case\n"
            "[Answer all four questions above: specific catalyst with timing, why market is "
            "underpricing it, 18-month scenario if catalyst fires, compelling entry price]\n\n"
            "### Operating leverage / earnings path\n"
            "[How catalyst converts to earnings, with numbers]\n\n"
            "### Exit signal\n"
            "[Observable event that breaks the thesis]\n\n"
            "### Catalyst conclusion\n"
            "[One sentence stance]"
        )
        user_prompt = (
            f"STOCK: {symbol} ({item.get('company_name', '')}) | SECTOR: {sector}\n"
            f"QUALITY SCORE: {quality_score:.2f} | ENTRY SIGNAL: {entry_signal}\n"
            f"ANALYST TARGET: {target_line}\n"
            f"{low_val_warning}"
            f"{risk_profile_block}"
            f"{macro_flow_block}"
            f"{news_block}"
            f"\n{snapshot_text}\n"
            "Identify the catalyst path and produce the timing verdict."
        )
        try:
            import openai

            client = self._client_instance()
            response = client.chat.completions.create(
                model=self._fast_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_completion_tokens=2000,
                timeout=45,
            )
            finish_reason = response.choices[0].finish_reason
            rationale = (response.choices[0].message.content or "").strip()
            print(
                f"OpenAI {symbol}: finish_reason={finish_reason}, "
                f"content_length={len(rationale)}"
            )
            if finish_reason == "content_filter":
                return "[OpenAI content filter triggered]"
            if finish_reason == "length":
                return "[OpenAI response truncated — synthesis based on partial analysis]"
            if len(rationale.strip()) < 100:
                print(
                    f"WARNING: OpenAI short response for {symbol}: "
                    f"'{rationale[:50]}...' finish={finish_reason}"
                )
                rationale = f"[OpenAI returned short response: {rationale}]"
            if not rationale:
                print(f"OpenAI returned empty string for {symbol}")
                print(f"Full response: {response}")
                return "[OpenAI returned empty — check logs]"
            return rationale
        except openai.RateLimitError as exc:
            print(f"OpenAI rate limit hit for {symbol}: {exc}")
            return "[OpenAI rate limited - retry in 60s]"
        except openai.AuthenticationError as exc:
            print(f"OpenAI API key invalid or expired: {exc}")
            return "[OpenAI authentication failed - check API key]"
        except openai.APITimeoutError as exc:
            print(f"OpenAI timeout for {symbol}: {exc}")
            return "[OpenAI timeout - server slow, retry]"
        except Exception as exc:
            print(
                f"OpenAI unexpected error for {symbol}: "
                f"{type(exc).__name__}: {exc}"
            )
            return f"[OpenAI error: {type(exc).__name__}]"

    def synthesise_comparison(
        self,
        stock_name: str,
        anthropic_rationale: str,
        openai_rationale: str,
        agreement_type: str = "both",
        factual_snapshot: str = "",
        entry_data: dict[str, Any] | None = None,
        macro_flow: dict[str, Any] | None = None,
        risk_profile: str = "Balanced",
    ) -> str | None:
        """
        Synthesis of Anthropic (risk) and OpenAI (catalyst) views.
        Uses Anthropic Sonnet — called once per stock in Compare Both mode.
        agreement_type: "both" | "anthropic_only" | "openai_only"
        Returns None if Anthropic is not enabled (caller must skip synthesis section).
        """
        # Synthesis always uses Anthropic Sonnet regardless of self.provider.
        if not self.config.anthropic_enabled:
            return None
        try:
            import anthropic as _anthropic
            client = _anthropic.Anthropic(api_key=self.config.anthropic_api_key)

            # ── Shared system prompt ─────────────────────────────────────────
            _synth_risk_cfg  = get_risk_config(risk_profile)
            _synth_min_rr    = _synth_risk_cfg["min_rr_ratio"]
            _synth_stale_cap = _synth_risk_cfg["staleness_cap_days"]
            _synth_hint      = RISK_PROMPT_HINTS.get(risk_profile, RISK_PROMPT_HINTS["Balanced"])
            _critical_instruction = {
                "Conservative": (
                    "CRITICAL INSTRUCTION: This is a CONSERVATIVE investor. "
                    "Default to WATCHLIST unless the entry plan has R/R >= 2.5x AND results are fresh (<45 days). "
                    "Never recommend ACCUMULATE GRADUALLY or STRONG BUY on stale data. "
                    "Prefer false negatives over false positives."
                ),
                "Balanced": (
                    "CRITICAL INSTRUCTION: This is a BALANCED investor. "
                    "Recommend ACCUMULATE GRADUALLY when R/R >= 2.0x and business quality is confirmed, "
                    "even if results are up to 90 days old. "
                    "Reserve WATCHLIST for when the risk/reward is genuinely unclear, not merely as a default."
                ),
                "Aggressive": (
                    "CRITICAL INSTRUCTION: This is an AGGRESSIVE investor. "
                    "Recommend ACCUMULATE GRADUALLY or ACCUMULATE ON DIPS when R/R >= 1.5x and "
                    "business quality is confirmed, even if results are up to 120 days old. "
                    "Do NOT default to WATCHLIST on staleness grounds — the investor explicitly accepts this risk. "
                    "Only use WATCHLIST when the R/R is below threshold or business quality is in genuine doubt."
                ),
                "Moderate": (
                    "CRITICAL INSTRUCTION: This is a BALANCED investor. "
                    "Recommend ACCUMULATE GRADUALLY when R/R >= 2.0x and business quality is confirmed, "
                    "even if results are up to 90 days old. "
                    "Reserve WATCHLIST for when the risk/reward is genuinely unclear, not merely as a default."
                ),
            }.get(risk_profile, (
                "CRITICAL INSTRUCTION: Recommend ACCUMULATE GRADUALLY when R/R threshold is met "
                "and business quality is confirmed. Do not default to WATCHLIST on staleness alone."
            ))

            system_prompt = (
                "You are a senior portfolio manager synthesising two analyst views on an Indian equity. "
                "Your job is to produce a clean, honest verdict — not a merger of both texts.\n\n"
                f"INVESTOR RISK PROFILE: {risk_profile}\n"
                f"{_synth_hint}\n"
                f"The minimum acceptable R/R for this profile is {_synth_min_rr}x. "
                f"Data up to {_synth_stale_cap} days stale is tolerated for this profile. "
                "If either analyst's entry plan meets the R/R threshold AND business quality is "
                "confirmed, the synthesis MUST consider ACCUMULATE GRADUALLY as a valid verdict "
                "-- do not default to WATCHLIST purely on data staleness grounds when the risk "
                "profile explicitly tolerates it.\n\n"
                f"{_critical_instruction}\n\n"
                "SYNTHESIS RULES:\n"
                "1. First identify whether disagreement is about:\n"
                "   - FACTS (one analyst has wrong data)\n"
                "   - TIME HORIZON (both right, different windows)\n"
                "   - VALUATION vs EXECUTION (classic divergence)\n"
                "   - POLICY RISK vs GROWTH OPTIMISM\n"
                "   - STALE ASSUMPTIONS (one using outdated framing)\n"
                "2. Do not average the views. Resolve the disagreement or acknowledge it "
                "cannot be resolved with current data.\n"
                "3. Do not repeat what both analysts said. Only state what the synthesis ADDS.\n"
                "4. End with ONE of these exact stances:\n"
                "   AVOID | WATCHLIST | BUY AFTER CONFIRMATION |\n"
                "   ACCUMULATE ON DIPS | ACCUMULATE GRADUALLY | STRONG BUY\n"
                "5. State confidence: HIGH / MEDIUM / LOW and the specific reason for that level.\n"
                "6. State the single most important condition that would change your stance.\n\n"
                "OUTPUT STRUCTURE:\n"
                "## SYNTHESIS VERDICT: [STANCE] | Confidence: [LEVEL]\n\n"
                "### Where analysts agree\n"
                "[Facts both accept as true]\n\n"
                "### Nature of disagreement\n"
                "[LABEL from rule 1 + one sentence explanation]\n\n"
                "### Resolution\n"
                "[Which view is better supported by current evidence and why — "
                "or why it cannot be resolved]\n\n"
                "### Portfolio fit\n"
                "[Why this stock specifically for this portfolio — gap it fills, risk it adds]\n\n"
                "### Confidence explanation\n"
                "[Why HIGH/MEDIUM/LOW — specific to evidence quality]\n\n"
                "### Stance-changing condition\n"
                "[The single event that would move stance up or down]"
            )

            # ── Build entry guidance block (provided verbatim for accuracy) ─
            entry_guidance = ""
            if entry_data:
                cmp    = float(entry_data.get("cmp", 0) or 0)
                ep     = float(entry_data.get("entry", 0) or 0)
                stop   = float(entry_data.get("stop", 0) or 0)
                target = float(entry_data.get("target", 0) or 0)
                rr     = entry_data.get("rr", 0)
                t_src  = str(entry_data.get("target_source_label", "model estimate") or "")
                is_model_target = "model" in t_src.lower()
                target_caveat = (
                    "\nTarget is MODEL-DERIVED — treat as indicative, not precise."
                    if is_model_target else ""
                )
                if cmp > 0:
                    entry_guidance = (
                        f"\n\n### Entry guidance\n"
                        f"CMP: ₹{cmp:,.0f} | Enter: ₹{ep:,.0f} | Stop: ₹{stop:,.0f}\n"
                        f"Target: ₹{target:,.0f} | R/R: {rr}x"
                        f"{target_caveat}"
                    )

            # ── Build user prompt — vary only the analyst section ────────────
            snapshot_section = f"\nFACTUAL SNAPSHOT:\n{factual_snapshot}\n" if factual_snapshot else ""
            from stock_platform.utils.fii_dii_fetcher import format_macro_flow_for_prompt
            _macro_block_synth = format_macro_flow_for_prompt(macro_flow or {})
            macro_section = f"\n{_macro_block_synth}" if _macro_block_synth else ""

            if agreement_type == "anthropic_only":
                user_prompt = (
                    f"The Risk Analyst selected {stock_name} but the Catalyst Analyst "
                    f"did NOT include it in recommendations.\n\n"
                    f"RISK ANALYST view:\n{anthropic_rationale}\n\n"
                    f"CATALYST ANALYST view: [did not select this stock]\n"
                    f"{snapshot_section}"
                    f"{macro_section}"
                    "Apply the synthesis rules. The 'Nature of disagreement' should explain "
                    "why the catalyst lens likely excluded this stock.\n"
                    "The 'Resolution' should state whether the risk analyst's case stands "
                    "without catalyst confirmation."
                    + entry_guidance
                )
            elif agreement_type == "openai_only":
                user_prompt = (
                    f"The Catalyst Analyst selected {stock_name} but the Risk Analyst "
                    f"did NOT include it in recommendations.\n\n"
                    f"RISK ANALYST view: [did not select this stock]\n\n"
                    f"CATALYST ANALYST view:\n{openai_rationale}\n"
                    f"{snapshot_section}"
                    f"{macro_section}"
                    "Apply the synthesis rules. The 'Nature of disagreement' should explain "
                    "what valuation or governance concern the risk lens likely flagged.\n"
                    "The 'Resolution' should state whether the catalyst case stands "
                    "without risk validation."
                    + entry_guidance
                )
            else:
                user_prompt = (
                    f"Two analysts reviewed {stock_name}:\n\n"
                    f"RISK ANALYST (Anthropic):\n{anthropic_rationale}\n\n"
                    f"CATALYST ANALYST (OpenAI):\n{openai_rationale}\n"
                    f"{snapshot_section}"
                    f"{macro_section}"
                    "Apply the synthesis rules above and produce the structured verdict."
                    + entry_guidance
                )

            response = client.messages.create(
                model=self.config.llm_reasoning_model,
                max_tokens=1500,
                temperature=0.2,
                system=[{"type": "text", "text": system_prompt}],
                messages=[{"role": "user", "content": user_prompt}],
            )
            synthesis_text = (response.content[0].text or "").strip()
            if response.stop_reason == "max_tokens":
                synthesis_text += "\n\n[Analysis truncated — re-run for full synthesis]"
                print(f"WARNING: synthesis truncated for {stock_name}")
            if not synthesis_text:
                return None
            if "SYNTHESIS VERDICT" not in synthesis_text:
                synthesis_text += (
                    "\n\n## SYNTHESIS VERDICT: WATCHLIST | Confidence: LOW\n"
                    "Synthesis incomplete — re-run Compare Both to regenerate."
                )
            return synthesis_text
        except Exception:
            return None

    def monitoring_rationale(
        self, action_row: dict[str, Any], thesis: dict[str, Any], drawdown: dict[str, Any]
    ) -> dict[str, Any] | None:
        """
        Structured action assessment for a monitored holding.
        Returns {"action": str, "severity": str, "rationale": str} or None.
        Called once per holding (~50/session); system prompt cached on Anthropic.
        JSON output prevents rationale text that starts with a directive word
        (e.g. "REJECT" or "BUY MORE") from being misread as the action field.
        """
        system_prompt = (
            "You are a disciplined portfolio monitoring agent for Indian equity portfolios. "
            "Return ONLY valid JSON in this exact format:\n"
            '{"action": "HOLD", "severity": "LOW", "rationale": "explanation here"}\n'
            "Valid actions: BUY MORE, HOLD, TRIM, SELL, REPLACE\n"
            "Valid severities: LOW, MEDIUM, HIGH, CRITICAL\n"
            "The rationale must be 1-2 sentences, specific and grounded in thesis status, "
            "drawdown, and risk context. Never include action words in the rationale field. "
            "No markdown, no prose outside the JSON object."
        )
        user_prompt = (
            f"Computed action: {action_row['action']} on {action_row['symbol']}\n"
            f"Computed severity: {action_row['severity']}\n"
            f"Computed rationale: {action_row['rationale']}\n"
            f"Thesis status: {thesis['status']} | Sector signal: {thesis['geo_signal_change']}\n"
            f"Drawdown: {drawdown['drawdown_pct']:.1f}% | Alert severity: {drawdown['severity']}"
        )
        raw = self._call(
            model=self._fast_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=200,
            temperature=0.2,
            cache_system=True,
        )
        if not raw:
            return None
        try:
            result = json.loads(raw)
            action = str(result.get("action", action_row["action"])).upper()
            if action not in {"BUY MORE", "HOLD", "TRIM", "SELL", "REPLACE"}:
                action = action_row["action"]
            severity = str(result.get("severity", action_row["severity"])).upper()
            if severity not in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}:
                severity = action_row["severity"]
            return {
                "action": action,
                "severity": severity,
                "rationale": str(result.get("rationale", "")),
            }
        except (json.JSONDecodeError, ValueError, KeyError):
            return None

    # ── REASONING TIER ───────────────────────────────────────────────────────
    # Anthropic: Sonnet 4.6   OpenAI: gpt-5.4

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
