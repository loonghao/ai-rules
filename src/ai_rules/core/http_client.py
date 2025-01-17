"""HTTP client utilities."""

# Import built-in modules
import logging
import ssl
from functools import lru_cache
from typing import Any, Dict, Optional

# Import third-party modules
import aiohttp
import certifi
from aiohttp import ClientTimeout
from yarl import URL

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


@lru_cache()
def create_ssl_context(verify_ssl: bool = True) -> ssl.SSLContext:
    """Create an SSL context with proper configuration.

    Args:
        verify_ssl: Whether to verify SSL certificates

    Returns:
        ssl.SSLContext: Configured SSL context
    """
    if verify_ssl:
        ssl_context = ssl.create_default_context(cafile=certifi.where())
    else:
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.set_ciphers("DEFAULT@SECLEVEL=1")  # Allow older ciphers

    # Support all SSL/TLS versions
    ssl_context.minimum_version = ssl.TLSVersion.MINIMUM_SUPPORTED
    ssl_context.maximum_version = ssl.TLSVersion.MAXIMUM_SUPPORTED

    # Enable legacy renegotiation and older protocols
    ssl_context.options &= ~ssl.OP_NO_SSLv3
    ssl_context.options &= ~ssl.OP_NO_TLSv1
    ssl_context.options &= ~ssl.OP_NO_TLSv1_1
    ssl_context.options &= ~ssl.OP_NO_RENEGOTIATION

    return ssl_context


class HTTPClient:
    """HTTP client with proper SSL handling."""

    def __init__(self, timeout: int = 30, verify_ssl: bool = True, headers: Optional[Dict[str, str]] = None):
        """Initialize HTTP client.

        Args:
            timeout: Request timeout in seconds
            verify_ssl: Whether to verify SSL certificates
            headers: Default headers to use for all requests
        """
        self.timeout = ClientTimeout(total=timeout)
        self.verify_ssl = verify_ssl
        self.headers = headers or {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "*/*",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
        }
        self._session = None

    async def __aenter__(self) -> "HTTPClient":
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context."""
        await self.close()

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists."""
        if self._session is None or self._session.closed:
            if self.verify_ssl:
                ssl_context = create_ssl_context(True)
                connector = aiohttp.TCPConnector(ssl=ssl_context)
            else:
                # For problematic SSL sites, try with no SSL at all
                connector = aiohttp.TCPConnector(ssl=False)

            self._session = aiohttp.ClientSession(connector=connector, timeout=self.timeout, headers=self.headers)

    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()

    def _process_url(self, url: str) -> str:
        """Process URL to ensure it's valid.

        Args:
            url: URL to process

        Returns:
            str: Processed URL
        """
        # Convert URL to yarl.URL for better handling
        parsed = URL(url)

        # Ensure scheme is present
        if not parsed.scheme:
            parsed = parsed.with_scheme("https")

        return str(parsed)

    async def get(
        self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> aiohttp.ClientResponse:
        """Send GET request.

        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.get

        Returns:
            aiohttp.ClientResponse: Response object
        """
        await self._ensure_session()
        url = self._process_url(url)

        try:
            response = await self._session.get(
                url, params=params, headers={**self.headers, **(headers or {})}, **kwargs
            )
            return response
        except Exception as e:
            if "SSL" in str(e):
                logger.warning(f"SSL error for {url}, retrying with SSL verification disabled")
                # Retry with SSL verification disabled
                old_verify = self.verify_ssl
                self.verify_ssl = False
                await self.close()  # Close old session to create new one with updated SSL context
                try:
                    await self._ensure_session()
                    response = await self._session.get(
                        url, params=params, headers={**self.headers, **(headers or {})}, **kwargs
                    )
                    return response
                finally:
                    self.verify_ssl = old_verify
                    await self.close()  # Reset session for future requests
            raise

    async def get_bytes(
        self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> bytes:
        """Send GET request and return response content as bytes.

        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.get

        Returns:
            bytes: Response content
        """
        async with await self.get(url, params, headers, **kwargs) as response:
            return await response.read()

    async def get_text(
        self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> str:
        """Send GET request and return response content as text.

        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.get

        Returns:
            str: Response content
        """
        async with await self.get(url, params, headers, **kwargs) as response:
            return await response.text()

    async def get_json(
        self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None, **kwargs
    ) -> Any:
        """Send GET request and return response content as JSON.

        Args:
            url: URL to request
            params: Query parameters
            headers: Request headers
            **kwargs: Additional arguments to pass to aiohttp.ClientSession.get

        Returns:
            Any: Response content as JSON
        """
        async with await self.get(url, params, headers, **kwargs) as response:
            return await response.json()
