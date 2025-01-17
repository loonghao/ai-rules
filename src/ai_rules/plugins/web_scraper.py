"""Web scraper plugin."""

# Import built-in modules
import asyncio
import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import urljoin

# Import third-party modules
import click
from bs4 import BeautifulSoup
from html2text import HTML2Text
from playwright.async_api import async_playwright

# Import local modules
from ai_rules.core.config import get_downloads_dir
from ai_rules.core.plugin import BasePluginResponse, Plugin

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
        self._playwright = None
        self._browser = None
        self._context = None
        self._description = "Scrape content from web pages"

    @property
    def name(self) -> str:
        """Get plugin name.

        Returns:
            str: Plugin name
        """
        return "web-scraper"

    @property
    def description(self) -> str:
        """Get plugin description.

        Returns:
            str: Plugin description
        """
        return self._description

    async def __aenter__(self):
        """Enter the context manager."""
        try:
            logger.info("Starting Playwright browser")
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch()
            self._context = await self._browser.new_context()
            logger.info("Successfully installed and launched Playwright browser")
            return self
        except Exception as e:
            logger.error(f"Failed to start browser: {e}")
            await self.__aexit__(None, None, None)
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        try:
            if self._context:
                await self._context.close()
                self._context = None
            if self._browser:
                await self._browser.close()
                self._browser = None
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
        except Exception as e:
            logger.error(f"Error cleaning up browser: {e}")

    async def try_goto_page(self, page, url: str, max_retries: int = 3, timeout: int = 30000) -> bool:
        """Try to navigate to a page with retries.

        Args:
            page: Playwright page
            url: URL to navigate to
            max_retries: Maximum number of retries
            timeout: Timeout in milliseconds

        Returns:
            bool: True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                # First try with networkidle
                try:
                    await page.goto(url, wait_until="networkidle", timeout=timeout)
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load with networkidle, trying with load: {e}")
                    
                # If networkidle fails, try with load
                try:
                    await page.goto(url, wait_until="load", timeout=timeout)
                    return True
                except Exception as e:
                    logger.warning(f"Failed to load with load, trying with domcontentloaded: {e}")
                    
                # If load fails, try with domcontentloaded
                await page.goto(url, wait_until="domcontentloaded", timeout=timeout)
                return True
                
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"All attempts failed for URL {url}: {e}")
                    return False
        return False

    async def scrape_with_playwright(
        self, url: str, selector: str = "body", timeout: int = 30000
    ) -> Optional[Tuple[str, str, str]]:
        """Scrape content from a URL using Playwright.

        Args:
            url: URL to scrape
            selector: CSS selector for content
            timeout: Timeout in milliseconds

        Returns:
            Optional[Tuple[str, str, str]]: Title, HTML content, and text content
        """
        try:
            logger.info(f"Creating new page for {url}")
            page = await self._context.new_page()
            
            # Set viewport size for better rendering
            await page.set_viewport_size({"width": 1920, "height": 1080})
            
            # Add custom headers
            await page.set_extra_http_headers({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1"
            })
            
            logger.info(f"Navigating to {url}")
            if not await self.try_goto_page(page, url, timeout=timeout):
                return None
            
            # Wait for dynamic content with reduced timeout
            logger.info("Waiting for 2 seconds")
            await asyncio.sleep(2)
            
            # Get page title
            logger.info("Getting page title")
            title = await page.title()
            
            # Try different selectors to find the main content
            selectors = [
                "article.document",
                "main.document",
                "div.document",
                "#content .document",
                ".content .document",
                "article",
                "main",
                ".content",
                "#content",
                ".main",
                ".container",
                "body"
            ]
            
            content_element = None
            used_selector = None
            
            for sel in selectors:
                logger.info(f"Trying selector: {sel}")
                try:
                    element = await page.query_selector(sel)
                    if element:
                        # Check if element has meaningful content
                        text = await element.text_content()
                        if len(text.strip()) > 50:  # Reduced minimum content length
                            content_element = element
                            used_selector = sel
                            logger.info(f"Found content with selector: {sel}")
                            break
                except Exception as e:
                    logger.debug(f"Selector {sel} failed: {e}")
            
            if not content_element:
                logger.warning("No suitable content found with any selector")
                return None

            # Get content
            logger.info("Getting content HTML and text")
            html = await content_element.inner_html()
            text = await content_element.text_content()
            
            # Clean up HTML while preserving structure
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements but preserve structure
            for element in soup.find_all(['script', 'style', 'iframe', 'noscript']):
                element.decompose()
            
            # Fix relative URLs for images and links
            base_tag = soup.find('base')
            base_url = base_tag['href'] if base_tag else url
            
            for img in soup.find_all('img'):
                src = img.get('src', '')
                if src:
                    if not src.startswith(('http://', 'https://', 'data:')):
                        img['src'] = urljoin(base_url, src)
                    # Preserve original dimensions
                    if not img.get('width') and not img.get('height'):
                        img['style'] = 'max-width: 100%; height: auto;'
                        
            for a in soup.find_all('a'):
                href = a.get('href', '')
                if href and not href.startswith(('http://', 'https://', '#', 'mailto:', 'tel:', 'javascript:')):
                    a['href'] = urljoin(base_url, href)
            
            # Preserve layout elements
            for div in soup.find_all('div'):
                if 'class' in div.attrs:
                    div['class'] = ' '.join(div['class'])  # Preserve classes
                if 'style' in div.attrs:
                    div['style'] = div['style']  # Preserve styles
            
            html = str(soup)
            
            logger.info(f"Content length - HTML: {len(html)}, Text: {len(text)}")
            
            await page.close()
            return title, html, text
            
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}")
            return None

    async def download_image(self, page, img_url: str, output_dir: Path) -> Optional[str]:
        """Download image and return its local path.

        Args:
            page: Playwright page
            img_url: Image URL
            output_dir: Output directory

        Returns:
            Optional[str]: Local image path if successful
        """
        try:
            # Create images directory
            images_dir = output_dir / "images"
            os.makedirs(images_dir, exist_ok=True)

            # Generate safe filename from URL
            img_filename = hashlib.md5(img_url.encode()).hexdigest()[:10]
            img_filename += os.path.splitext(img_url)[-1] or '.jpg'
            local_path = images_dir / img_filename

            # Skip if image already exists
            if local_path.exists():
                logger.info(f"Image already exists: {local_path}")
                return f"images/{img_filename}"

            # Download image using page.goto() and screenshot
            async with self._context.new_page() as img_page:
                try:
                    # Set viewport size for better image capture
                    await img_page.set_viewport_size({"width": 1920, "height": 1080})
                    
                    # Navigate to image URL
                    response = await img_page.goto(img_url, wait_until="networkidle", timeout=10000)
                    if not response:
                        logger.error(f"Failed to load image {img_url}: No response")
                        return None
                        
                    # Check content type
                    content_type = response.headers.get("content-type", "")
                    if not content_type.startswith("image/"):
                        logger.error(f"Invalid content type for {img_url}: {content_type}")
                        return None
                        
                    # Get image data
                    img_data = await response.body()
                    if not img_data:
                        logger.error(f"No image data received for {img_url}")
                        return None
                        
                    # Save image
                    with open(local_path, "wb") as f:
                        f.write(img_data)
                        
                    logger.info(f"Downloaded image: {img_url} -> {local_path}")
                    return f"images/{img_filename}"
                    
                except Exception as e:
                    logger.error(f"Failed to download image {img_url}: {e}")
                    return None

        except Exception as e:
            logger.error(f"Error downloading image {img_url}: {e}")
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

    async def _fetch_url(self, url: str) -> Optional[str]:
        """Fetch content from a URL.

        Args:
            url: URL to fetch content from

        Returns:
            HTML content if successful, None otherwise
        """
        try:
            async with self as scraper:
                result = await scraper.scrape_with_playwright(url)
                if result:
                    _, html_content, _ = result
                    return html_content
                return None
        except Exception as e:
            logger.error(f"Error fetching URL {url}: {str(e)}")
            return None

    def format_markdown(self, text: str) -> str:
        """Format markdown text to improve readability.

        Args:
            text: Input markdown text

        Returns:
            str: Formatted markdown text
        """
        try:
            lines = text.split('\n')
            processed_lines = []
            current_section = None
            
            for line in lines:
                line = line.rstrip()
                
                # Handle headings
                if line.startswith('#'):
                    if current_section != 'heading':
                        processed_lines.append('')
                    processed_lines.append(line)
                    processed_lines.append('')
                    current_section = 'heading'
                    continue

                # Handle lists
                if line.lstrip().startswith(('- ', '* ', '1. ')):
                    if current_section != 'list':
                        processed_lines.append('')
                    processed_lines.append(line)
                    current_section = 'list'
                    continue

                # Handle paragraphs
                if line.strip():
                    if current_section != 'paragraph':
                        processed_lines.append('')
                    processed_lines.append(line)
                    current_section = 'paragraph'
                else:
                    if current_section:
                        processed_lines.append('')
                    current_section = None
            
            # Remove consecutive empty lines
            result = []
            prev_empty = False
            for line in processed_lines:
                if not line.strip():
                    if not prev_empty:
                        result.append(line)
                    prev_empty = True
                else:
                    result.append(line)
                    prev_empty = False
            
            return '\n'.join(result).strip()
            
        except Exception as e:
            logger.error(f"Error formatting markdown: {e}")
            return text

    def parse_html(self, html_content: str, output_format: str = "markdown") -> str:
        """Parse HTML content and convert to the specified format.

        Args:
            html_content: HTML content to parse
            output_format: Output format (markdown or text)

        Returns:
            str: Parsed content in the specified format
        """
        try:
            if output_format == "markdown":
                h = HTML2Text()
                # Basic settings
                h.body_width = 0  # No wrapping
                h.unicode_snob = True
                h.escape_snob = True
                
                # Link settings
                h.ignore_links = False
                h.wrap_links = False
                h.inline_links = True
                h.protect_links = True
                h.use_automatic_links = True
                h.skip_internal_links = False
                
                # Image settings
                h.ignore_images = False
                h.handle_image = lambda src, alt='': f'\n![]({src})\n' if src else ''
                
                # List settings
                h.ul_item_mark = "-"
                h.wrap_list_items = False
                h.list_indent = "  "
                
                # Table settings
                h.ignore_tables = False
                h.pad_tables = True
                
                # Formatting settings
                h.emphasis_mark = '*'
                h.strong_mark = '**'
                h.single_line_break = True
                h.mark_code = True
                
                # Convert HTML to Markdown
                markdown = h.handle(html_content)
                
                # Post-process markdown
                lines = markdown.split('\n')
                processed_lines = []
                current_section = None
                
                for line in lines:
                    line = line.rstrip()
                    
                    # Handle headings
                    if line.startswith('#'):
                        if current_section != 'heading':
                            processed_lines.append('')
                        processed_lines.append(line)
                        processed_lines.append('')
                        current_section = 'heading'
                        continue
                    
                    # Handle images
                    if line.startswith('!'):
                        processed_lines.append('')
                        processed_lines.append(line)
                        processed_lines.append('')
                        current_section = 'image'
                        continue
                    
                    # Handle lists
                    if line.lstrip().startswith(('- ', '* ', '1. ')):
                        if current_section != 'list':
                            processed_lines.append('')
                        processed_lines.append(line)
                        current_section = 'list'
                        continue
                    
                    # Handle code blocks
                    if line.startswith('```'):
                        processed_lines.append('')
                        processed_lines.append(line)
                        if not line.strip() == '```':
                            current_section = 'code'
                        else:
                            current_section = None
                        continue
                    
                    # Handle paragraphs
                    if line.strip():
                        if current_section not in ('paragraph', 'code'):
                            processed_lines.append('')
                        processed_lines.append(line)
                        current_section = 'paragraph'
                    else:
                        if current_section:
                            processed_lines.append('')
                        current_section = None
                
                return '\n'.join(processed_lines).strip()
            else:
                # For text format, preserve some structure
                soup = BeautifulSoup(html_content, 'html.parser')
                # Remove unwanted elements
                for element in soup.find_all(['script', 'style', 'iframe']):
                    element.decompose()
                # Get text with better spacing
                lines = []
                for element in soup.find_all(['p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                    text = element.get_text(strip=True)
                    if text:
                        lines.append(text)
                return '\n\n'.join(lines)
                
        except Exception as e:
            logger.error(f"Error parsing HTML: {e}")
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

    async def execute(self, **kwargs) -> str:
        """Execute the plugin.

        Args:
            kwargs: Keyword arguments from Click

        Returns:
            str: Execution result as formatted string
        """
        url = kwargs.get("url")
        selector = kwargs.get("selector", "body")
        output_dir = Path(kwargs.get("output_dir", get_downloads_dir()))

        try:
            content = await self.scrape_with_playwright(url, selector)
            if not content:
                return BasePluginResponse(
                    status="error",
                    error=BasePluginResponse.ErrorDetails(
                        code="no_content",
                        message=f"No content found at {url}",
                        details={"url": url}
                    )
                ).format_for_llm()

            title, html_content, text_content = content

            # Convert HTML to Markdown
            markdown_content = self.parse_html(html_content, output_format="markdown")

            # Save content
            os.makedirs(output_dir, exist_ok=True)

            # Generate filename from URL and title
            safe_title = "".join(c if c.isalnum() or c in "-_." else "_" for c in title[:50])
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = output_dir / f"{safe_title}_{timestamp}.md"

            # Save as Markdown with frontmatter and better formatting
            with open(filepath, "w", encoding="utf-8") as f:
                # Write frontmatter
                f.write("---\n")
                f.write(f"title: {title}\n")
                f.write(f"url: {url}\n")
                f.write(f"date: {datetime.now().isoformat()}\n")
                f.write("type: web-content\n")
                f.write("---\n\n")

                # Process and write content
                lines = markdown_content.split('\n')
                current_section = None
                
                for line in lines:
                    # Handle headings
                    if line.startswith('#'):
                        if current_section != 'heading':
                            f.write('\n')
                        f.write(line + '\n\n')
                        current_section = 'heading'
                        continue

                    # Handle lists
                    if line.lstrip().startswith(('- ', '* ', '1. ')):
                        if current_section != 'list':
                            f.write('\n')
                        f.write(line + '\n')
                        current_section = 'list'
                        continue

                    # Handle paragraphs
                    if line.strip():
                        if current_section != 'paragraph':
                            f.write('\n')
                        f.write(line + '\n')
                        current_section = 'paragraph'
                    else:
                        if current_section:
                            f.write('\n')
                        current_section = None

            response = BasePluginResponse(
                status="success",
                message=f"Successfully scraped content from {url}",
                data={
                    "url": url,
                    "title": title,
                    "output_file": str(filepath)
                },
                metadata=BasePluginResponse.ResponseMetadata(
                    source=self.name,
                    version=self.version
                )
            )
            return response.format_for_llm()

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            response = BasePluginResponse(
                status="error",
                error=BasePluginResponse.ErrorDetails(
                    code="scraping_error",
                    message=str(e),
                    details={"url": url}
                ),
                metadata=BasePluginResponse.ResponseMetadata(
                    source=self.name,
                    version=self.version
                )
            )
            return response.format_for_llm()

    def format_response(self, response):
        return response.format_for_llm()

    def format_error(self, response):
        return response.format_for_llm()

    @property
    def click_command(self):
        """Get the click command for this plugin.

        Returns:
            Click command
        """
        @click.command(help=self.description)
        @click.argument("url", type=str)
        @click.option(
            "--output-dir", "-o",
            default=str(get_downloads_dir()),
            help="Directory to save scraped content (default: user downloads directory)",
            type=click.Path(file_okay=False, dir_okay=True, path_type=str)
        )
        @click.option(
            "--selector", "-s",
            default="article, main, .content, .main, #content, #main, body",
            help="CSS selector to extract specific content (default: common content elements)",
            type=str
        )
        @click.option(
            "--recursive/--no-recursive", "-r/-R",
            default=False,
            help="Recursively scrape linked pages (default: False)"
        )
        @click.option(
            "--max-depth", "-d",
            default=1,
            help="Maximum depth for recursive scraping (default: 1)",
            type=int
        )
        def command(url: str, output_dir: str, selector: str, recursive: bool, max_depth: int):
            """Scrape web content from URLs.

            Args:
                url: URL to scrape
                output_dir: Directory to save scraped content
                selector: CSS selector to extract specific content
                recursive: Recursively scrape linked pages
                max_depth: Maximum depth for recursive scraping
            """
            async def _run():
                try:
                    async with self:
                        result = await self.execute(
                            url=url,
                            output_dir=output_dir,
                            selector=selector,
                            recursive=recursive,
                            max_depth=max_depth
                        )
                        click.echo(result)
                        return result
                except Exception as e:
                    error_msg = str(e)
                    logger.error(f"Failed to scrape {url}: {error_msg}")
                    result = BasePluginResponse(
                        status="error",
                        error=BasePluginResponse.ErrorDetails(
                            code="command_error",
                            message=error_msg,
                            details={"url": url}
                        ),
                        metadata=BasePluginResponse.ResponseMetadata(
                            source=self.name,
                            version=self.version
                        )
                    ).format_for_llm()
                    click.echo(result, err=True)
                    return result

            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    return loop.run_until_complete(_run())
                finally:
                    loop.close()
            except KeyboardInterrupt:
                logger.info("Operation cancelled by user")
                result = BasePluginResponse(
                    status="error",
                    error=BasePluginResponse.ErrorDetails(
                        code="operation_cancelled",
                        message="Operation cancelled by user",
                        details={"url": url}
                    ),
                    metadata=BasePluginResponse.ResponseMetadata(
                        source=self.name,
                        version=self.version
                    )
                ).format_for_llm()
                click.echo(result, err=True)
                return result
            except Exception as e:
                error_msg = str(e)
                logger.error(f"Unexpected error: {error_msg}")
                result = BasePluginResponse(
                    status="error",
                    error=BasePluginResponse.ErrorDetails(
                        code="unexpected_error",
                        message=f"Unexpected error: {error_msg}",
                        details={"url": url}
                    ),
                    metadata=BasePluginResponse.ResponseMetadata(
                        source=self.name,
                        version=self.version
                    )
                ).format_for_llm()
                click.echo(result, err=True)
                return result

        return command
