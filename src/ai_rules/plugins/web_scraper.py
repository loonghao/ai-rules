"""Web scraper plugin."""

# Import built-in modules
import asyncio
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Import third-party modules
import click
import html2text
from playwright.async_api import async_playwright

# Import local modules
from ai_rules.core.plugin import Plugin

logger: logging.Logger = logging.getLogger(__name__)


class WebPage:
    """Model for web page data."""

    def __init__(self, url: str, title: str = "", links: List[str] = None):
        self.url = url
        self.title = title
        self.links = links or []


class WebScraperPlugin(Plugin):
    """Plugin for scraping web content."""

    def __init__(self):
        """Initialize the plugin."""
        super().__init__()
        self._browser = None
        self._context = None

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "web_scraper"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Scrape web content"

    async def __aenter__(self):
        """Enter the context manager."""
        if not self._browser:
            playwright = await async_playwright().start()
            self._browser = await playwright.chromium.launch()
            self._context = await self._browser.new_context()
            logger.info("Successfully installed and launched Playwright browser")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        if self._context:
            await self._context.close()
        if self._browser:
            await self._browser.close()
        self._context = None
        self._browser = None

    async def scrape_with_playwright(
        self, url: str, selector: str = "body", timeout: int = 30000
    ) -> Optional[Tuple[str, str, str]]:
        """Scrape content from a URL using Playwright.

        Args:
            url: URL to scrape
            selector: CSS selector for content
            timeout: Timeout in milliseconds

        Returns:
            Tuple of (title, html_content, text_content) if successful, None otherwise
        """
        try:
            logger.info(f"Creating new page for {url}")
            page = await self._context.new_page()

            logger.info(f"Navigating to {url}")
            await page.goto(url, timeout=timeout)

            logger.info("Waiting for network idle")
            await page.wait_for_load_state("networkidle", timeout=timeout)

            logger.info("Waiting for 5 seconds")
            await asyncio.sleep(5)

            # Get page title
            logger.info("Getting page title")
            title = await page.title()
            logger.info(f"Page title: {title}")

            # Try different selectors in order of preference
            selectors = [
                "article.document",
                "main.document",
                "div.document",
                "#content .document",
                ".content .document",
                "article",
                "main",
                "div.content",
                "#content",
                "body",
            ]

            content = None
            for selector in selectors:
                logger.info(f"Trying selector: {selector}")
                content = await page.query_selector(selector)
                if content:
                    logger.info(f"Found content with selector: {selector}")
                    break

            if not content:
                logger.error("Could not find any content")
                return None

            # Remove unwanted elements
            logger.info("Removing unwanted elements")
            unwanted_selectors = [
                "nav",
                "header",
                "footer",
                ".privacy-policy",
                ".cookie-notice",
                ".advertisement",
                ".ads",
                "#cookie-notice",
                "#privacy-policy",
                "script",
                "style",
            ]

            for selector in unwanted_selectors:
                elements = await page.query_selector_all(selector)
                for element in elements:
                    try:
                        await element.evaluate("element => element.remove()")
                    except Exception as e:
                        logger.warning(f"Failed to remove element {selector}: {str(e)}")

            logger.info("Getting content HTML and text")
            html_content = await content.inner_html()
            text_content = await content.text_content()

            logger.info(f"Content length - HTML: {len(html_content)}, Text: {len(text_content)}")

            await page.close()
            return title, html_content, text_content

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None

    async def fetch_page(self, url: str, context) -> Optional[str]:
        """Fetch a page using Playwright.

        Args:
            url: URL to fetch
            context: Browser context

        Returns:
            Page content if successful, None otherwise
        """
        try:
            page = await context.new_page()
            await page.goto(url)
            await page.wait_for_load_state("networkidle")
            content = await page.content()
            await page.close()
            return content
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def parse_html(self, html: str, output_format: str = "markdown") -> str:
        """Parse HTML content.

        Args:
            html: HTML content to parse
            output_format: Output format (markdown, text, html)

        Returns:
            Parsed content in the specified format
        """
        try:
            if output_format == "html":
                return html

            h = html2text.HTML2Text()
            h.ignore_links = False
            h.ignore_images = False
            h.ignore_tables = False
            h.body_width = 0

            if output_format == "text":
                h.ignore_links = True
                h.ignore_images = True

            return h.handle(html)
        except Exception as e:
            logger.error(f"Error parsing HTML: {str(e)}")
            return ""

    async def get_page_links(self, url: str) -> Optional[WebPage]:
        """Get all links from a page.

        Args:
            url: URL to get links from

        Returns:
            WebPage object containing page data and links
        """
        try:
            page = await self._context.new_page()
            await page.goto(url)
            await page.wait_for_load_state("networkidle")

            # Get page title
            title = await page.title()

            # Get all links
            links = await page.eval_on_selector_all(
                "a[href]",
                """elements => elements.map(el => {
                    const href = el.getAttribute('href');
                    if (href && !href.startsWith('#') && !href.startsWith('javascript:')) {
                        return href;
                    }
                    return null;
                }).filter(href => href !== null)""",
            )

            await page.close()
            return WebPage(url=url, title=title, links=links)

        except Exception as e:
            logger.error(f"Error getting links from {url}: {str(e)}")
            return None

    async def process_urls(
        self, urls: List[str], max_concurrent: int = 5, output_format: str = "markdown"
    ) -> List[dict]:
        """Process multiple URLs concurrently.

        Args:
            urls: List of URLs to process
            max_concurrent: Maximum number of concurrent requests
            output_format: Output format (markdown, text, html)

        Returns:
            List of processed results
        """
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)

        async def process_url(url: str):
            async with semaphore:
                try:
                    async with async_playwright() as p:
                        browser = await p.chromium.launch()
                        context = await browser.new_context()
                        content = await self.fetch_page(url, context)
                        if content:
                            parsed_content = self.parse_html(content, output_format)
                            return {"url": url, "content": parsed_content, "error": None}
                        return {"url": url, "content": None, "error": "Failed to fetch content"}
                except Exception as e:
                    return {"url": url, "content": None, "error": str(e)}

        tasks = [process_url(url) for url in urls]
        results = await asyncio.gather(*tasks)
        return results

    def validate(self, **kwargs) -> bool:
        """Validate input parameters.

        Args:
            kwargs: Keyword arguments
                urls: List of URLs to scrape
                max_concurrent: Maximum number of concurrent requests
                format: Output format (markdown, text, html)

        Returns:
            True if parameters are valid, False otherwise
        """
        urls = kwargs.get("urls", [])
        max_concurrent = kwargs.get("max_concurrent", 5)
        output_format = kwargs.get("format", "markdown")

        if not urls:
            return False

        if not all(url.startswith(("http://", "https://")) for url in urls):
            return False

        if not isinstance(max_concurrent, int) or max_concurrent < 1:
            return False

        if output_format not in ["markdown", "text", "html"]:
            return False

        return True

    def get_metadata(self):
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0.0",
            "author": "AI Rules Team",
            "supported_formats": ["markdown", "text", "html"],
        }

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin.

        Returns:
            Click command
        """

        @click.command(name=self.name, help=self.description)
        @click.argument("url")
        @click.argument("output_dir")
        @click.option("--selector", help="CSS selector to extract specific content")
        @click.option(
            "--recursive/--no-recursive", default=False, help="Recursively scrape linked pages (default: False)"
        )
        @click.option("--max-depth", default=1, help="Maximum depth for recursive scraping (default: 1)")
        def command(url, output_dir, selector, recursive, max_depth):
            """Scrape web content from URLs.

            Args:
                url: URL to scrape
                output_dir: Directory to save scraped content
                selector: CSS selector to extract specific content
                recursive: Recursively scrape linked pages
                max_depth: Maximum depth for recursive scraping
            """
            return asyncio.run(
                self.execute(
                    url=url, output_dir=output_dir, selector=selector, recursive=recursive, max_depth=max_depth
                )
            )

        return command

    def execute(self, **kwargs) -> str:
        """Execute the plugin.

        Args:
            kwargs: Keyword arguments from Click

        Returns:
            Execution result as JSON string
        """
        url = kwargs.get("url")
        selector = kwargs.get("selector", "body")

        try:
            content = asyncio.run(self.scrape_with_playwright(url, selector))
            if not content:
                raise ValueError(f"No content found at {url}")

            title, html_content, text_content = content

            # Convert HTML to Markdown
            markdown_content = self.parse_html(html_content, output_format="markdown")

            # Save content
            output_dir = Path(kwargs.get("output_dir"))
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename from URL
            filename = re.sub(r"[^\w\-_.]", "_", url.split("/")[-1])
            if not filename.endswith(".md"):
                filename += ".md"

            # Add timestamp to avoid conflicts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{filename}_{timestamp}.md"

            # Save as Markdown with frontmatter
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("---\n")
                f.write(f"title: {title}\n")
                f.write(f"url: {url}\n")
                f.write(f"date: {datetime.now().isoformat()}\n")
                f.write("type: web-content\n")
                f.write("---\n\n")
                f.write(markdown_content)

            response_data = {
                "data": {"url": url, "title": title, "output_file": str(filepath)},
                "message": f"Successfully scraped content from {url}",
            }

            return json.dumps(response_data, indent=2, ensure_ascii=False)

        except Exception as e:
            error_data = {"error": str(e), "url": url}
            return json.dumps(error_data, indent=2, ensure_ascii=False)
