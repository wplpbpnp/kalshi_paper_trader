import asyncio
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .kalshi_http import KalshiHttpClient


@dataclass(frozen=True)
class KalshiWsConfig:
    ws_url: str = "wss://api.elections.kalshi.com/trade-api/ws/v2"
    ping_interval_s: float = 20.0
    ping_timeout_s: float = 20.0
    close_timeout_s: float = 5.0


class KalshiWsClient:
    def __init__(
        self,
        *,
        http_client: Optional[KalshiHttpClient],
        cfg: KalshiWsConfig,
        auth: bool,
    ):
        self._http_client = http_client
        self._cfg = cfg
        self._auth = auth

    def connect(self):
        try:
            import websockets  # type: ignore
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "Missing dependency 'websockets'. Install it with:\n"
                "  pip install websockets\n"
                "or install repo deps (if you maintain them) with:\n"
                "  pip install -r kalshi_paper_trader/requirements.txt"
            ) from e

        extra_headers: Optional[Dict[str, str]] = None
        if self._auth:
            if self._http_client is None:
                raise RuntimeError("auth=True requires a KalshiHttpClient for header signing.")
            # Per Kalshi WS docs, sign for GET + /trade-api/ws/v2 (no host).
            extra_headers = self._http_client.auth_headers("GET", "/trade-api/ws/v2")

        kwargs = dict(
            ping_interval=self._cfg.ping_interval_s,
            ping_timeout=self._cfg.ping_timeout_s,
            close_timeout=self._cfg.close_timeout_s,
        )

        # websockets changed header kwarg name across versions.
        try:
            return websockets.connect(self._cfg.ws_url, additional_headers=extra_headers, **kwargs)
        except TypeError:
            return websockets.connect(self._cfg.ws_url, extra_headers=extra_headers, **kwargs)

    @staticmethod
    async def send_json(ws, payload: Dict[str, Any]) -> None:
        await ws.send(json.dumps(payload, separators=(",", ":")))

    @staticmethod
    async def recv_json(ws, *, timeout_s: Optional[float] = None) -> Dict[str, Any]:
        if timeout_s is None:
            raw = await ws.recv()
        else:
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_s)
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        return json.loads(raw)
