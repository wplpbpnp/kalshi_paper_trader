import base64
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding


def _load_api_key(env_file: Optional[str]) -> str:
    api_key = os.environ.get("KALSHI_API_KEY") or os.environ.get("API_KEY")
    if api_key:
        return api_key.strip()

    if env_file:
        with open(env_file, "r") as f:
            for line in f:
                if line.startswith("KALSHI_API_KEY=") or line.startswith("API_KEY="):
                    return line.split("=", 1)[1].strip()

    raise RuntimeError(
        "Missing Kalshi API key. Set env var KALSHI_API_KEY (or API_KEY) "
        "or pass --env-file pointing at a .env containing KALSHI_API_KEY=..."
    )


def _load_private_key(pem_file: str):
    with open(pem_file, "r") as f:
        secret_key_str = f.read().strip()

    if not secret_key_str.startswith("-----"):
        secret_key_str = (
            "-----BEGIN RSA PRIVATE KEY-----\n"
            + secret_key_str
            + "\n-----END RSA PRIVATE KEY-----"
        )

    return serialization.load_pem_private_key(secret_key_str.encode(), password=None)


@dataclass(frozen=True)
class KalshiHttpClient:
    base_url: str
    api_key: str
    private_key: Any
    timeout_s: float = 10.0

    @classmethod
    def from_files(
        cls,
        *,
        base_url: str,
        env_file: Optional[str],
        pem_file: str,
        timeout_s: float = 10.0,
    ) -> "KalshiHttpClient":
        return cls(
            base_url=base_url.rstrip("/"),
            api_key=_load_api_key(env_file),
            private_key=_load_private_key(pem_file),
            timeout_s=timeout_s,
        )

    def _sign_pss_text(self, text: str) -> str:
        signature = self.private_key.sign(
            text.encode("utf-8"),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _headers(self, method: str, path: str) -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        path_for_signing = path.split("?", 1)[0]
        msg = timestamp + method + path_for_signing
        signature = self._sign_pss_text(msg)
        return {
            "Content-Type": "application/json",
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": timestamp,
        }

    def auth_headers(self, method: str, path: str) -> Dict[str, str]:
        return self._headers(method, path)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        headers = self._headers(method.upper(), path)
        resp = requests.request(
            method=method.upper(),
            url=self.base_url + path,
            headers=headers,
            params=params or {},
            json=json_body,
            timeout=self.timeout_s,
        )
        resp.raise_for_status()
        if not resp.text:
            return {}
        return resp.json()

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("GET", path, params=params)

    def post(self, path: str, json_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return self._request("POST", path, json_body=json_body)

    def delete(self, path: str) -> Dict[str, Any]:
        return self._request("DELETE", path)
