"""The DKK/INR reference rate the PWA uses to show amounts in kroner.

The browser can't call a rate service itself (the CSP allows only this API), and the
ECB rates the daily market-data job stores are the same ones the server values with.
"""

import datetime as dt
from collections.abc import Callable
from datetime import date
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from wealth.marketdata.store import latest_fx_rate

# Same freshness rule as savings and loans in other currencies.
_MAX_AGE_DAYS = 10


class FxRateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # FastAPI resolves route annotations at runtime, and a field named "date" would
    # shadow the type, so the field type is spelled through the module.
    date: dt.date
    rate: str


def build_router(authenticated_user: Callable[..., Any]) -> APIRouter:
    from wealth.api.money_map import get_today

    router = APIRouter()
    AuthUser = Annotated[Any, Depends(authenticated_user)]
    Today = Annotated[date, Depends(get_today)]

    @router.get(
        "/v1/fx/dkk-inr",
        response_model=FxRateResponse,
        responses={404: {"description": "No recent DKK/INR rate is stored."}},
    )
    def get_dkk_inr_rate(user: AuthUser, today: Today) -> FxRateResponse:
        row = latest_fx_rate(user.session, "DKK/INR", today)
        if row is None or (today - row[1]).days > _MAX_AGE_DAYS:
            raise HTTPException(status_code=404, detail="No recent DKK/INR rate is available.")
        rate, as_of, _source = row
        return FxRateResponse(date=as_of, rate=format(rate.normalize(), "f"))

    return router
