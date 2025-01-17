"""Image search and download plugin.

This module provides functionality for searching and downloading images from Bing.
It implements a robust download strategy with multiple fallback options.
"""

# Import built-in modules
import asyncio
import base64
import json
import logging
import os
import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Protocol, Union
from pathlib import Path
import re
import urllib.parse
from enum import Enum, auto
from dataclasses import dataclass

# Import third-party modules
import aiohttp
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright, Page
import click
from pydantic import BaseModel, Field

# Import local modules
from ai_rules.core.http_client import HTTPClient
from ai_rules.core.plugin import Plugin, BasePluginResponse
from ai_rules.core.config import get_images_dir

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)

class DownloadStrategy(Enum):
    """Available download strategies."""
    HTTP = auto()
    HTTP_NO_SSL = auto()
    PLAYWRIGHT = auto()

class ImageSource(Enum):
    """Image source types."""
    DIRECT = auto()
    BASE64 = auto()
    SCREENSHOT = auto()

@dataclass
class DownloadResult:
    """Download result data."""
    success: bool
    file_path: Optional[Path]
    strategy: DownloadStrategy
    source: Optional[ImageSource] = None
    error: Optional[str] = None

class ImageResponse(BasePluginResponse):
    """Image search and download response model."""
    
    class ImageData(BaseModel):
        """Structure for image data."""
        url: str = Field(..., description="Image URL")
        file_path: Optional[str] = Field(None, description="Local file path if downloaded")
        strategy: Optional[str] = Field(None, description="Download strategy used")
        source: Optional[str] = Field(None, description="Image source type")

    data: List[ImageData] = Field(default_factory=list, description="List of image results")

class DownloadError(Exception):
    """Base exception for download errors."""
    pass

class SSLError(DownloadError):
    """SSL verification error."""
    pass

class HTTPError(DownloadError):
    """HTTP request error."""
    pass

class PlaywrightError(DownloadError):
    """Playwright-specific error."""
    pass

# Configuration constants
PROBLEMATIC_DOMAINS = {
    'docs.unrealengine.com': 'Known 403 issues',
    'jonas-erkert.de': 'SSL version issues'
}

BROWSER_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Connection': 'keep-alive'
}

FIREFOX_PREFS = {
    "security.tls.version.min": 1,
    "security.ssl.enable_ocsp_stapling": False,
    "security.ssl.enable_ocsp_must_staple": False,
    "security.ssl.require_safe_negotiation": False,
    "security.ssl.treat_unsafe_negotiation_as_broken": False,
    "security.ssl3.rsa_des_ede3_sha": True,
    "security.ssl3.rsa_rc4_128_sha": True,
    "security.ssl3.rsa_rc4_40_md5": True,
    "security.tls.insecure_fallback_hosts": "",
    "network.stricttransportsecurity.preloadlist": False,
    "network.http.spdy.enabled.http2": False
}

CHROMIUM_ARGS = [
    '--disable-web-security',
    '--allow-running-insecure-content',
    '--ignore-certificate-errors',
    '--ignore-ssl-errors',
    '--disable-gpu',
    '--no-sandbox',
    '--disable-setuid-sandbox'
]

class ImageResult(BaseModel):
    """Image result model."""

    title: str
    image_url: str
    thumbnail_url: str
    source_url: str
    source: str
    height: int
    width: int
    local_path: Optional[str] = None


class DownloadStrategyProtocol(Protocol):
    """Protocol for download strategies."""
    
    async def download(self, url: str, save_dir: Path) -> DownloadResult:
        """Download image using this strategy."""
        ...

def clean_filename(url: str) -> str:
    """Clean URL to create a valid filename.
    
    Args:
        url: URL to clean
        
    Returns:
        Clean filename
    """
    # Remove query parameters and fragments
    url = url.split('?')[0].split('#')[0]
    
    # Get the last part of the URL as filename
    filename = url.split('/')[-1]
    
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    
    # Get extension
    ext = os.path.splitext(filename)[1]
    if not ext:
        ext = '.jpg'
    
    # Create hash
    url_hash = hashlib.md5(url.encode()).hexdigest()
    
    return f"{url_hash}{ext}"

class HTTPDownloader:
    """HTTP download strategy."""

    def __init__(self, verify_ssl: bool = True, timeout: int = 10, max_retries: int = 3, retry_delay: float = 1.0) -> None:
        """Initialize HTTP downloader.
        
        Args:
            verify_ssl: Whether to verify SSL certificates
            timeout: Timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.strategy = DownloadStrategy.HTTP if verify_ssl else DownloadStrategy.HTTP_NO_SSL
        self._session = None

    async def _get_session(self) -> HTTPClient:
        """Get or create HTTP client session.
        
        Returns:
            HTTPClient session
        """
        if self._session is None:
            self._session = HTTPClient(timeout=self.timeout, verify_ssl=self.verify_ssl)
        return self._session

    async def close(self) -> None:
        """Close HTTP client session."""
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _handle_response(self, response: aiohttp.ClientResponse, save_path: Path) -> DownloadResult:
        """Handle HTTP response and save image if successful.
        
        Args:
            response: HTTP response
            save_path: Path to save the image
        
        Returns:
            DownloadResult with download status and details
        
        Raises:
            HTTPError: If response status is not 200 or content is not an image
        """
        if response.status == 200:
            content_type = response.headers.get('content-type', '')
            if 'image' in content_type:
                data = await response.read()
                save_path.write_bytes(data)
                logger.info(f"Saved image to {save_path}")
                return DownloadResult(
                    success=True,
                    file_path=save_path,
                    strategy=self.strategy,
                    source=ImageSource.DIRECT
                )
            else:
                error_msg = f"Content type is not an image: {content_type}"
                logger.warning(error_msg)
                raise HTTPError(error_msg)
        elif response.status in (403, 404, 502):
            error_msg = f"Failed to download image: status {response.status}"
            logger.warning(error_msg)
            raise HTTPError(error_msg)
        else:
            error_msg = f"Failed to download image: status {response.status}"
            logger.warning(error_msg)
            if response.status >= 500:  # Server errors might be temporary
                raise HTTPError(error_msg, temporary=True)
            raise HTTPError(error_msg)

    async def download(self, url: str, save_dir: Path) -> DownloadResult:
        """Download image using HTTP client.
        
        Args:
            url: Image URL
            save_dir: Directory to save the image
        
        Returns:
            DownloadResult with download status and details
        """
        retries = 0
        last_error = None
        filename = clean_filename(url)
        save_path = save_dir / filename

        # Skip if already downloaded
        if save_path.exists():
            logger.info(f"Image already exists: {save_path}")
            return DownloadResult(
                success=True,
                file_path=save_path,
                strategy=self.strategy,
                source=ImageSource.DIRECT
            )

        while retries < self.max_retries:
            try:
                client = await self._get_session()
                try:
                    response = await client.get(url)
                    return await self._handle_response(response, save_path)
                except aiohttp.ClientSSLError as e:
                    error_msg = f"SSL error: {e}"
                    logger.warning(error_msg)
                    raise SSLError(error_msg)
                except aiohttp.ClientError as e:
                    error_msg = f"HTTP error: {e}"
                    logger.warning(error_msg)
                    raise HTTPError(error_msg, temporary=True)
                except Exception as e:
                    error_msg = f"Unexpected error: {e}"
                    logger.error(error_msg)
                    raise DownloadError(error_msg)

            except (HTTPError, SSLError, DownloadError) as e:
                last_error = e
                if hasattr(e, 'temporary') and e.temporary:
                    retries += 1
                    if retries < self.max_retries:
                        delay = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                        logger.info(f"Retrying download ({retries}/{self.max_retries}) in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    error_msg = f"Maximum retries ({self.max_retries}) exceeded"
                    logger.error(error_msg)
                    raise DownloadError(error_msg)
                else:
                    raise

        if last_error:
            raise last_error
        raise DownloadError("Maximum retries exceeded")

class PlaywrightDownloader:
    """Playwright download strategy."""

    def __init__(self, timeout: int = 30, max_retries: int = 2, retry_delay: float = 2.0) -> None:
        """Initialize Playwright downloader.
        
        Args:
            timeout: Page load timeout in seconds
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.strategy = DownloadStrategy.PLAYWRIGHT

    async def _take_screenshot(self, page: Page, save_path: Path) -> DownloadResult:
        """Take screenshot of the page.
        
        Args:
            page: Playwright page
            save_path: Path to save the screenshot
        
        Returns:
            DownloadResult with download status and details
        """
        try:
            img_element = await page.query_selector("img")
            if img_element:
                src = await img_element.get_attribute("src")
                if src:
                    content = await img_element.screenshot()
                    save_path.write_bytes(content)
                    logger.info(f"Saved image from src to {save_path}")
                    return DownloadResult(
                        success=True,
                        file_path=save_path,
                        strategy=self.strategy,
                        source=ImageSource.PLAYWRIGHT
                    )
            
            # If no img element found or no src attribute, take full page screenshot
            await page.screenshot(path=str(save_path))
            logger.info(f"Saved page screenshot to {save_path}")
            return DownloadResult(
                success=True,
                file_path=save_path,
                strategy=self.strategy,
                source=ImageSource.SCREENSHOT
            )

        except Exception as e:
            error_msg = f"Failed to take screenshot: {e}"
            logger.warning(error_msg)
            raise PlaywrightError(error_msg)

    async def download(self, url: str, save_dir: Path) -> DownloadResult:
        """Download image using Playwright.
        
        Args:
            url: Image URL
            save_dir: Directory to save the image
        
        Returns:
            DownloadResult with download status and details
        """
        retries = 0
        last_error = None
        filename = clean_filename(url)
        save_path = save_dir / filename

        if save_path.exists():
            logger.info(f"Image already exists: {save_path}")
            return DownloadResult(
                success=True,
                file_path=save_path,
                strategy=self.strategy,
                source=ImageSource.DIRECT
            )

        while retries < self.max_retries:
            try:
                async with async_playwright() as p:
                    browser = await p.chromium.launch(
                        ignore_default_args=["--mute-audio"],
                        args=["--no-sandbox"]
                    )
                    context = await browser.new_context(
                        ignore_https_errors=True,
                        bypass_csp=True
                    )
                    page = await context.new_page()

                    try:
                        await page.goto(url, wait_until="networkidle", timeout=self.timeout * 1000)
                        return await self._take_screenshot(page, save_path)

                    except TimeoutError:
                        error_msg = "Page load timeout"
                        logger.warning(error_msg)
                        raise PlaywrightError(error_msg, temporary=True)

                    except Exception as e:
                        error_msg = f"Playwright error: {e}"
                        logger.warning(error_msg)
                        raise PlaywrightError(error_msg, temporary=True)

                    finally:
                        await context.close()
                        await browser.close()

            except PlaywrightError as e:
                last_error = e
                if hasattr(e, 'temporary') and e.temporary:
                    retries += 1
                    if retries < self.max_retries:
                        delay = self.retry_delay * (2 ** (retries - 1))  # Exponential backoff
                        logger.info(f"Retrying download ({retries}/{self.max_retries}) in {delay:.1f}s...")
                        await asyncio.sleep(delay)
                        continue
                    error_msg = f"Maximum retries ({self.max_retries}) exceeded"
                    logger.error(error_msg)
                    raise DownloadError(error_msg)
                else:
                    raise

            except Exception as e:
                error_msg = f"Unexpected error: {e}"
                logger.error(error_msg)
                raise DownloadError(error_msg)

        if last_error:
            raise last_error
        raise DownloadError("Maximum retries exceeded")

class ImageDownloader:
    """Image downloader with multiple strategies."""

    def __init__(self) -> None:
        """Initialize image downloader."""
        self.http_downloader = HTTPDownloader(verify_ssl=True, timeout=10)
        self.http_no_ssl_downloader = HTTPDownloader(verify_ssl=False, timeout=10)
        self.playwright_downloader = PlaywrightDownloader(timeout=30)

    async def close(self) -> None:
        """Close all downloaders."""
        await self.http_downloader.close()
        await self.http_no_ssl_downloader.close()

    def _choose_initial_strategy(self, url: str) -> DownloadStrategy:
        """Choose initial download strategy based on URL.
        
        Args:
            url: Image URL
        
        Returns:
            Most appropriate download strategy for the URL
        """
        domain = urllib.parse.urlparse(url).netloc
        if domain in PROBLEMATIC_DOMAINS:
            return DownloadStrategy.PLAYWRIGHT
        return DownloadStrategy.HTTP

    async def download(self, url: str, save_dir: Path) -> DownloadResult:
        """Download image using appropriate strategy.
        
        Args:
            url: Image URL
            save_dir: Directory to save the image
        
        Returns:
            DownloadResult with download status and details
        """
        strategy = self._choose_initial_strategy(url)
        last_error = None
        
        try:
            if strategy == DownloadStrategy.HTTP:
                try:
                    return await self.http_downloader.download(url, save_dir)
                except SSLError:
                    logger.info("Retrying with SSL verification disabled")
                    try:
                        return await self.http_no_ssl_downloader.download(url, save_dir)
                    except (SSLError, HTTPError) as e:
                        last_error = e
                        logger.info("Falling back to Playwright")
                        try:
                            result = await self.playwright_downloader.download(url, save_dir)
                            if result and result.success:
                                return result
                            last_error = DownloadError(result.error if result else "Unknown error")
                        except Exception as e:
                            last_error = e
                except HTTPError as e:
                    last_error = e
                    logger.info("Falling back to Playwright")
                    try:
                        result = await self.playwright_downloader.download(url, save_dir)
                        if result and result.success:
                            return result
                        last_error = DownloadError(result.error if result else "Unknown error")
                    except Exception as e:
                        last_error = e
            else:
                try:
                    result = await self.playwright_downloader.download(url, save_dir)
                    if result and result.success:
                        return result
                    last_error = DownloadError(result.error if result else "Unknown error")
                except Exception as e:
                    last_error = e

            error_msg = f"All download strategies failed: {last_error}"
            logger.error(error_msg)
            return DownloadResult(
                success=False,
                file_path=None,
                strategy=strategy,
                error=error_msg
            )

        except Exception as e:
            error_msg = f"All download strategies failed: {e}"
            logger.error(error_msg)
            return DownloadResult(
                success=False,
                file_path=None,
                strategy=strategy,
                error=error_msg
            )

class ImagePlugin(Plugin):
    """Bing image search plugin."""

    def __init__(self):
        """Initialize plugin."""
        super().__init__()
        self.version = "0.1.0"

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "image"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search and download images from Bing"

    @property
    def arguments(self) -> List[click.Argument]:
        """Get plugin arguments.

        Returns:
            List[click.Argument]: List of Click arguments
        """
        return [
            click.Argument(["query"], type=str),
        ]

    @property
    def options(self) -> List[click.Option]:
        """Get plugin options.

        Returns:
            List[click.Option]: List of Click options
        """
        return [
            click.Option(
                ["--region"],
                type=str,
                default="wt-wt",
                help="Region for search results",
            ),
            click.Option(
                ["--safesearch"],
                type=click.Choice(["on", "moderate", "off"]),
                default="moderate",
                help="Safe search level",
            ),
        ]

    async def download_image(self, url: str, save_dir: Path) -> Optional[Path]:
        """Download an image from a URL.

        Args:
            url: Image URL
            save_dir: Directory to save the image

        Returns:
            Optional[Path]: Path to saved image if successful, None otherwise
        """
        image_downloader = ImageDownloader()
        download_result = await image_downloader.download(url, save_dir)
        if download_result.success:
            return download_result.file_path
        else:
            logger.error(f"Failed to download image: {download_result.error}")
            return None

    async def execute(self, **kwargs) -> str:
        """Execute the image search plugin.

        Args:
            **kwargs: Keyword arguments from Click

        Returns:
            str: JSON response string
        """
        downloader = None
        search_client = None
        try:
            query = kwargs["query"]
            region = kwargs.get("region", "wt-wt")
            safesearch = kwargs.get("safesearch", "moderate")

            # Create save directory
            save_dir = Path(get_images_dir()) / datetime.now().strftime("%Y-%m-%d")
            save_dir.mkdir(parents=True, exist_ok=True)

            # Initialize downloader
            downloader = ImageDownloader()

            # Search for images
            search_url = (
                f"https://www.bing.com/images/search?q={query}"
                f"&setlang=en&mkt={region}&safesearch={safesearch}"
            )

            search_client = HTTPClient()
            response = await search_client.get(search_url)
            if response.status != 200:
                raise HTTPError(f"Failed to search images: status {response.status}")

            html = await response.text()
            soup = BeautifulSoup(html, "html.parser")
            image_elements = soup.find_all("a", class_="iusc")

            results = []
            for element in image_elements:
                try:
                    data = json.loads(element.get("m", "{}"))
                    image_url = data.get("murl")
                    if not image_url:
                        continue

                    download_result = await downloader.download(image_url, save_dir)
                    if download_result:  
                        results.append({
                            "url": image_url,
                            "file_path": str(download_result.file_path) if download_result.success else None,
                            "strategy": download_result.strategy.name if download_result.success else None,
                            "source": download_result.source.name if download_result.success and download_result.source else None,
                            "error": download_result.error if not download_result.success else None
                        })

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse image data: {element.get('m')}")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to process image: {e}")
                    continue

            response = ImageResponse(
                status="success",
                data=results,
                message=f"Found and downloaded {len([r for r in results if r.get('file_path')])} images for '{query}' to {save_dir}",
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "query": query,
                    "region": region,
                    "safesearch": safesearch,
                    "save_dir": str(save_dir),
                    "timestamp": datetime.now().isoformat()
                }
            )

            return response.json()

        except Exception as e:
            logger.error(f"Error executing image search: {e}")
            error_response = ImageResponse(
                status="error",
                data=[],
                error=str(e),
                message=f"Failed to search and download images for '{query}'",
                metadata={
                    "plugin_name": self.name,
                    "plugin_version": self.version,
                    "timestamp": datetime.now().isoformat()
                }
            )
            return error_response.json()

        finally:
            if downloader:
                await downloader.close()
            if search_client:
                await search_client.close()

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "version": self.version,
        }

    @property
    def click_command(self) -> click.Command:
        """Get click command for the plugin."""

        @click.command(name="image", help=self.description)
        @click.argument("query", type=str)
        @click.option(
            "--region",
            type=str,
            default="wt-wt",
            help="Region for search results",
        )
        @click.option(
            "--safesearch",
            type=click.Choice(["on", "moderate", "off"]),
            default="moderate",
            help="Safe search level",
        )
        def image_command(**kwargs) -> str:
            """Execute the image search command."""
            import asyncio
            return asyncio.run(self.execute(**kwargs))

        return image_command
