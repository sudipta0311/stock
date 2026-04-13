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

    def buy_rationale(self, item: dict[str, Any], portfolio_context: dict[str, Any]) -> str | None:
        """
        2-3 sentence portfolio-personalised buy rationale.
        Each provider uses a distinct analytical lens so Compare Both surfaces real disagreement.
          Anthropic — contrarian / risk analyst: challenges the score, surfaces the key bear case.
          OpenAI    — momentum / catalyst analyst: surfaces near-term catalysts and entry timing.
        Called once per recommendation (~3-4/session).
        System prompt cached on Anthropic to minimise repeated-call cost.
        """
        source_names = [e.get("instrument_name", "") for e in item.get("fund_attribution", [])][:4]
        live_facts = item.get("live_financials", {}) or {}
        price_ctx = item.get("price_context", {}) or {}
        metric_bits: list[str] = []
        roce = live_facts.get("roce_ttm") or live_facts.get("returnOnCapitalEmployed")
        if isinstance(roce, (int, float)):
            metric_bits.append(f"ROCE {roce * 100:.1f}%")
        revenue_growth = live_facts.get("revenueGrowth") or live_facts.get("revenue_growth")
        if isinstance(revenue_growth, (int, float)):
            metric_bits.append(f"Revenue growth {revenue_growth * 100:.1f}%")
        debt_to_equity = live_facts.get("debtToEquity") or live_facts.get("debt_to_equity")
        if isinstance(debt_to_equity, (int, float)):
            metric_bits.append(f"D/E {debt_to_equity:.2f}")
        eps = live_facts.get("trailingEps")
        if isinstance(eps, (int, float)):
            metric_bits.append(f"EPS {eps:.2f}")
        analyst_target = price_ctx.get("analyst_target") or live_facts.get("targetMeanPrice")
        if isinstance(analyst_target, (int, float)):
            metric_bits.append(f"Target {analyst_target:.2f}")
        tech_signals = item.get("technical_signals", [])
        if tech_signals:
            tech_parts = [
                f"{s['type']}: {s['value']} ({s['signal']}) — {s['note']}"
                for s in tech_signals
            ]
            tech_ctx = "Technical context: " + " | ".join(tech_parts)
        else:
            tech_ctx = ""

        pe_ctx = item.get("pe_context") or {}
        pe_block = ""
        if pe_ctx.get("pe_current") is not None:
            pe_block = (
                f"\nVERIFIED PE DATA (use these exact numbers, do not estimate PE from other fields):\n"
                f"Current PE:    {pe_ctx['pe_current']:.1f}x\n"
                f"5-Year Median: {pe_ctx['pe_5yr_median']:.1f}x\n" if pe_ctx.get("pe_5yr_median") else
                f"\nVERIFIED PE DATA (use these exact numbers, do not estimate PE from other fields):\n"
                f"Current PE:    {pe_ctx['pe_current']:.1f}x\n"
                f"5-Year Median: N/A\n"
            ) + f"PE Assessment: {pe_ctx['pe_assessment']}\nPE Signal:     {pe_ctx['pe_signal']}"

        stock_line = (
            f"Stock: {item['company_name']} ({item['symbol']}), sector: {item['sector']}.\n"
            f"Entry signal: {item['entry_signal']} | Quality score: {item['quality_score']:.2f}\n"
            f"Overlap: {item['overlap_pct']:.1f}% | Gap reason: {item['gap_reason']}\n"
            f"Held via funds: {', '.join(source_names) or 'none'}\n"
            f"Portfolio size: {len(portfolio_context.get('normalized_exposure', []))} positions\n"
            f"Metrics: {' | '.join(metric_bits) if metric_bits else 'Live metrics unavailable'}"
            + (f"\n{tech_ctx}" if tech_ctx else "")
            + (pe_block if pe_block else "")
        )

        pe_instruction = (
            "IMPORTANT: Use ONLY the verified PE data provided above. "
            "Do NOT estimate or calculate PE yourself. "
            "If PE Assessment says CHEAP_VS_HISTORY, your analysis must acknowledge this — "
            "a cheap-vs-history PE changes the risk framing significantly. "
            "If PE Assessment says EXPENSIVE_VS_HISTORY, that is a key risk to surface."
        ) if pe_block else ""

        if self.provider == "anthropic":
            system_prompt = (
                "You are a contrarian portfolio analyst. Your job is to: "
                "1. Challenge the quantitative recommendation and find weaknesses the score misses. "
                "2. Identify the single most important RISK that could invalidate this thesis. "
                "3. State clearly what would make you WRONG about this recommendation. "
                "4. Give a verdict: does the risk/reward justify entry NOW or should the investor wait for a better entry point? "
                "Write exactly 4 short bullets labelled RISK, WE ARE WRONG IF, VERDICT, and SUPPORTING METRIC. "
                "Be direct and specific. Reference actual financial metrics, not generalities."
            )
            user_prompt = (
                f"{stock_line}\n"
                + (f"\n{pe_instruction}\n" if pe_instruction else "")
                + "Challenge the thesis and produce the risk-focused verdict."
            )
        else:
            system_prompt = (
                "You are a financial analyst specialising in Indian equities. "
                "For the stock provided write 3-4 sentences covering:\n"
                "1. The single most important near-term catalyst that will drive price appreciation in 3-6 months\n"
                "2. What the market is currently underestimating\n"
                "3. The specific price level or event that signals exit\n"
                "4. Whether to enter now or wait for a better level\n"
                "Be specific. Reference actual financial metrics provided. "
                "Avoid generic phrases. Name specific events and numbers."
            )
            user_prompt = (
                f"{stock_line}\n"
                + (f"\n{pe_instruction}\n" if pe_instruction else "")
                + "Identify the catalyst path and produce the timing verdict."
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
                    max_completion_tokens=800,
                    timeout=30,
                )
                finish_reason = response.choices[0].finish_reason
                rationale = (response.choices[0].message.content or "").strip()
                print(
                    f"OpenAI {item['symbol']}: finish_reason={finish_reason}, "
                    f"content_length={len(rationale)}"
                )
                if finish_reason == "content_filter":
                    return "[OpenAI content filter triggered]"
                if finish_reason == "length":
                    return "[OpenAI response truncated — increase max_tokens]"
                if len(rationale.strip()) < 100:
                    print(f"WARNING: OpenAI short response for {item['symbol']}: "
                          f"'{rationale[:50]}...' finish={finish_reason}")
                    rationale = f"[OpenAI returned short response: {rationale}]"
                if not rationale:
                    print(f"OpenAI returned empty string for {item['symbol']}")
                    print(f"Full response: {response}")
                    return "[OpenAI returned empty — check logs]"
                return rationale
            except openai.RateLimitError as exc:
                print(f"OpenAI rate limit hit for {item['symbol']}: {exc}")
                return "[OpenAI rate limited - retry in 60s]"
            except openai.AuthenticationError as exc:
                print(f"OpenAI API key invalid or expired: {exc}")
                return "[OpenAI authentication failed - check API key]"
            except openai.APITimeoutError as exc:
                print(f"OpenAI timeout for {item['symbol']}: {exc}")
                return "[OpenAI timeout - server slow, retry]"
            except Exception as exc:
                print(
                    f"OpenAI unexpected error for {item['symbol']}: "
                    f"{type(exc).__name__}: {exc}"
                )
                return f"[OpenAI error: {type(exc).__name__}]"

        return self._call(
            model=self._fast_model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_tokens=300,
            temperature=0.35,
            cache_system=True,
        )

    def synthesise_comparison(
        self,
        stock_name: str,
        anthropic_rationale: str,
        openai_rationale: str,
        agreement_type: str = "both",
    ) -> str | None:
        """
        Synthesis of Anthropic (contrarian/risk) and OpenAI (momentum/catalyst) views.
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

            if agreement_type == "anthropic_only":
                system_prompt = (
                    "You are a senior portfolio analyst. "
                    "The Risk Analyst selected a stock but the Catalyst Analyst did NOT. "
                    "Produce exactly 4 bullet points as instructed."
                )
                user_prompt = (
                    f"The Risk Analyst selected {stock_name} but the Catalyst Analyst "
                    f"did NOT include it in recommendations.\n\n"
                    f"RISK ANALYST view:\n{anthropic_rationale}\n\n"
                    "In 4 bullet points:\n"
                    "- Why did the Catalyst Analyst likely exclude this stock — "
                    "what near-term catalyst is missing?\n"
                    "- Is the Risk Analyst's selection justified WITHOUT a catalyst confirmation?\n"
                    "- What specific event would cause the Catalyst Analyst to also recommend this stock?\n"
                    "- COMBINED VERDICT: Is this a genuine opportunity the catalyst lens missed, "
                    "or a value trap the momentum lens correctly avoided?\n"
                    "Be specific. Reference the actual risk and metrics identified."
                )
            elif agreement_type == "openai_only":
                system_prompt = (
                    "You are a senior portfolio analyst. "
                    "The Catalyst Analyst selected a stock but the Risk Analyst did NOT. "
                    "Produce exactly 4 bullet points as instructed."
                )
                user_prompt = (
                    f"The Catalyst Analyst selected {stock_name} but the Risk Analyst "
                    f"did NOT include it in recommendations.\n\n"
                    f"CATALYST ANALYST view:\n{openai_rationale}\n\n"
                    "In 4 bullet points:\n"
                    "- Why did the Risk Analyst likely exclude this stock — "
                    "what valuation or quality concern caused it to be filtered?\n"
                    "- Is the Catalyst Analyst's selection justified WITHOUT risk validation?\n"
                    "- What specific risk would the Risk Analyst flag if it had analysed this stock?\n"
                    "- COMBINED VERDICT: Is this a genuine near-term opportunity, "
                    "or is the missing risk analysis itself a warning signal?\n"
                    "Be specific. Reference the actual catalyst and metrics identified."
                )
            else:
                system_prompt = (
                    "Two analysts reviewed the same stock. "
                    "One is a contrarian risk analyst and the other is a momentum catalyst analyst. "
                    "Produce exactly 3 bullet points:\n"
                    "- Where do both analysts AGREE?\n"
                    "- Where do they DISAGREE and why does that disagreement matter?\n"
                    "- COMBINED VERDICT: ENTER NOW / ACCUMULATE GRADUALLY / WAIT FOR BETTER ENTRY.\n"
                    "Be specific. Reference the actual risk and catalyst identified."
                )
                user_prompt = (
                    f"Two analysts reviewed {stock_name}:\n\n"
                    f"RISK ANALYST (Anthropic):\n{anthropic_rationale}\n\n"
                    f"CATALYST ANALYST (OpenAI):\n{openai_rationale}\n\n"
                    "In 3 bullet points:\n"
                    "- Where do both analysts AGREE?\n"
                    "- Where do they DISAGREE and why does the disagreement matter?\n"
                    "- COMBINED VERDICT: ENTER NOW / ACCUMULATE GRADUALLY / WAIT FOR BETTER ENTRY?\n"
                    "Be specific. Reference the actual risk and catalyst identified."
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
            if "COMBINED VERDICT" not in synthesis_text:
                synthesis_text += (
                    "\n\n• COMBINED VERDICT: Synthesis incomplete — "
                    "re-run Compare Both to regenerate."
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
