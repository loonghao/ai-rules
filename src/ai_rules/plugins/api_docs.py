"""API documentation search and save plugin."""

# Import built-in modules
import asyncio
import json
import logging
import math
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urljoin, urlparse

import aiofiles
import aiohttp

# Import third-party modules
import click
import html2text
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from pydantic import BaseModel

# Import local modules
from ai_rules.core.plugin import Plugin
from ai_rules.plugins.translate import TranslatePlugin
from ai_rules.plugins.web_scraper import WebScraperPlugin

logger: logging.Logger = logging.getLogger(__name__)


class APIDocResult(BaseModel):
    """Model for API documentation search results."""

    title: str
    link: str
    snippet: str
    source: str  # e.g. 'readthedocs', 'github', 'official'


class APIDocResponse(BaseModel):
    """Response model for API documentation search."""

    results: List[APIDocResult]
    total: int
    query: str


class APIDocs(Plugin):
    """API documentation search and save plugin."""

    def __init__(self):
        """Initialize the plugin."""
        super().__init__()
        self.web_scraper = WebScraperPlugin()
        self._visited_urls = set()
        self.ddg = DDGS()
        self.translate_plugin = TranslatePlugin()

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "api-docs"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search and save API documentation with content filtering"

    def _translate_to_english(self, query: str) -> str:
        """Translate Chinese query to English.

        Args:
            query: Chinese query string

        Returns:
            English query string
        """
        try:
            # 使用翻译插件进行翻译
            translated = asyncio.run(self.translate_plugin.execute(text=query, source_lang="zh", target_lang="en"))

            # 解析JSON响应
            result = json.loads(translated)
            if "data" in result and "translation" in result["data"]:
                return result["data"]["translation"]

            logger.error(f"Unexpected translation response: {result}")
            return query

        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return query  # 如果翻译失败，返回原始查询

    async def _detect_search_endpoint(self, base_url: str) -> Optional[Dict[str, Any]]:
        """Detect search endpoint of the API documentation site.

        Args:
            base_url: Base URL of the documentation

        Returns:
            Dictionary containing search endpoint information if found, None otherwise
        """
        # 常见的搜索端点模式
        search_patterns = [
            {
                "type": "sphinx",
                "search_url": "{base_url}/search.html?q={query}",
                "selector": ".search li.search-result-item a, .search div.context a",
                "api_endpoint": "{base_url}/searchindex.js",
            },
            {
                "type": "readthedocs",
                "search_url": "{base_url}/search.html?q={query}",
                "selector": "article.search-result a",
                "api_endpoint": "{base_url}/_/api/v2/search/?q={query}",
            },
            {
                "type": "algolia",
                "search_url": None,
                "selector": None,
                "api_endpoint": "https://{application_id}-dsn.algolia.net/1/indexes/{index_name}/query",
            },
        ]

        try:
            async with self.web_scraper as scraper:
                logger.info(f"Detecting search endpoint for {base_url}")
                # 检查页面源代码中的搜索相关信息
                content = await scraper.scrape_with_playwright(url=base_url, selector="html", timeout=60000)

                if not content:
                    logger.warning(f"No content found at {base_url}")
                    return None

                _, html_content, _ = content

                # 检测搜索类型
                if "READTHEDOCS_DATA" in html_content:
                    logger.info("Detected ReadTheDocs search")
                    return next(p for p in search_patterns if p["type"] == "readthedocs")
                elif "DOCUMENTATION_OPTIONS" in html_content:
                    logger.info("Detected Sphinx search")
                    return next(p for p in search_patterns if p["type"] == "sphinx")
                elif "docsearch" in html_content or "algolia" in html_content.lower():
                    logger.info("Detected Algolia search")
                    # 提取Algolia配置
                    algolia_config = re.search(
                        r'docsearch\s*\(\s*{\s*appId:\s*[\'"]([^\'"]+)[\'"].*indexName:\s*[\'"]([^\'"]+)[\'"]',
                        html_content,
                        re.DOTALL,
                    )
                    if algolia_config:
                        pattern = next(p for p in search_patterns if p["type"] == "algolia")
                        pattern["application_id"] = algolia_config.group(1)
                        pattern["index_name"] = algolia_config.group(2)
                        return pattern

                # 尝试检测自定义搜索实现
                if "search.html" in html_content:
                    logger.info("Detected custom search page")
                    return {"type": "custom", "search_url": urljoin(base_url, "search.html?q={query}"), "selector": "a"}

                logger.warning("No search functionality detected")

        except Exception as e:
            logger.error(f"Error detecting search endpoint: {str(e)}")

        return None

    async def _search_documentation(self, url: str, query: str) -> List[str]:
        """Search documentation using site's search functionality.

        Args:
            url: Base URL of the documentation
            query: Search query

        Returns:
            List of relevant page URLs
        """
        try:
            # 检测搜索端点
            search_config = await self._detect_search_endpoint(url)
            if not search_config:
                logger.info("No search endpoint detected, falling back to base pages")
                base_pages = [
                    f"{url}/reference.html",
                    f"{url}/cookbook.html",
                    f"{url}/advanced.html",
                    f"{url}/index.html",
                ]
                logger.info(f"Using base pages: {base_pages}")
                return base_pages

            logger.info(f"Using search config: {search_config}")

            # 根据不同的搜索类型执行搜索
            if search_config["type"] in ["sphinx", "readthedocs", "custom"]:
                # 使用网页搜索
                search_url = search_config["search_url"].format(base_url=url, query=query)
                logger.info(f"Searching at URL: {search_url}")

                async with self.web_scraper as scraper:
                    content = await scraper.scrape_with_playwright(
                        url=search_url, selector=search_config["selector"], timeout=60000
                    )

                    if not content:
                        logger.warning(f"No search results found at {search_url}")
                        return []

                    _, html_content, _ = content

                    # 提取搜索结果链接
                    soup = BeautifulSoup(html_content, "html.parser")
                    links = soup.select(search_config["selector"])

                    result_urls = [urljoin(url, link["href"]) for link in links]
                    logger.info(f"Found {len(result_urls)} search results: {result_urls}")
                    return result_urls

            elif search_config["type"] == "algolia":
                # 使用Algolia API搜索
                logger.info("Using Algolia search API")
                headers = {
                    "X-Algolia-API-Key": "your_api_key",  # 需要从页面配置中提取
                    "X-Algolia-Application-Id": search_config["application_id"],
                }

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        search_config["api_endpoint"], json={"query": query, "hitsPerPage": 5}, headers=headers
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            result_urls = [hit["url"] for hit in data.get("hits", [])]
                            logger.info(f"Found {len(result_urls)} Algolia results: {result_urls}")
                            return result_urls

            logger.warning("Search returned no results")
            return []

        except Exception as e:
            logger.error(f"Error searching documentation: {str(e)}")
            return []

    async def _search_relevant_pages(self, url: str, query: str) -> List[str]:
        """Search for pages relevant to the query.

        Args:
            url: Base URL of the documentation
            query: Search query

        Returns:
            List of relevant page URLs
        """
        try:
            # 首先尝试直接访问一些可能相关的页面
            base_pages = [
                f"{url}/reference.html",  # API 参考
                f"{url}/cookbook.html",  # API 指南
                f"{url}/advanced.html",  # 高级主题
                f"{url}/index.html",  # 主页
            ]

            # 然后使用DuckDuckGo搜索其他相关页面
            search_query = f"site:{url} {query}"
            logger.info(f"Searching with query: {search_query}")
            results = list(self.ddg.text(search_query, max_results=5))

            # 提取搜索结果的URL
            urls = set(base_pages)  # 使用集合避免重复
            for result in results:
                result_url = result.get("link", "")
                if result_url.startswith(url):
                    urls.add(result_url)

            logger.info(f"Found URLs: {urls}")
            return list(urls)

        except Exception as e:
            logger.error(f"Error searching pages: {str(e)}")
            return base_pages  # 如果搜索失败，至少返回基本页面

    async def _extract_relevant_sections(self, content: str, query: str) -> str:
        """Extract sections of content relevant to the query.

        Args:
            content: Content to extract from
            query: Query to match against

        Returns:
            Relevant sections of content
        """
        try:
            # 将内容分割成段落
            paragraphs = re.split(r"\n\s*\n", content)

            # 计算每个段落的相关性分数
            scored_paragraphs = []
            query_terms = set(query.lower().split())

            for paragraph in paragraphs:
                if not paragraph.strip():
                    continue

                # 计算段落中包含的查询词数量
                paragraph_terms = set(paragraph.lower().split())
                matches = len(query_terms.intersection(paragraph_terms))

                # 计算相关性分数 (匹配数量 / 段落长度的对数)
                score = matches / (1 + math.log(len(paragraph_terms) + 1))

                if score > 0:
                    scored_paragraphs.append((score, paragraph))

            # 按分数排序并选择最相关的段落
            scored_paragraphs.sort(reverse=True)
            relevant_paragraphs = [p for _, p in scored_paragraphs[:5]]

            # 如果没有找到相关段落，返回前几个段落
            if not relevant_paragraphs and paragraphs:
                relevant_paragraphs = paragraphs[:3]

            return "\n\n".join(relevant_paragraphs)

        except Exception as e:
            logger.error(f"Error extracting relevant sections: {str(e)}")
            # 如果出错，返回原始内容的前1000个字符
            return content[:1000] if content else ""

    async def save_docs(self, docs: List[Dict[str, Any]], output_dir: Path) -> None:
        """Save documentation to files.

        Args:
            docs: List of documents to save
            output_dir: Directory to save documents to
        """
        try:
            # 确保输出目录存在
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存每个文档
            for i, doc in enumerate(docs):
                try:
                    # 创建一个有意义的文件名
                    title = re.sub(r"[^\w\s-]", "", doc["title"])
                    title = re.sub(r"[-\s]+", "-", title).strip("-")
                    filename = f"{title[:50]}-{i+1}.md"

                    # 构建完整的文档内容
                    content = [f"# {doc['title']}", f"\nSource: {doc['url']}\n", doc["content"]]

                    # 写入文件
                    file_path = output_dir / filename
                    async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                        await f.write("\n".join(content))

                    logger.info(f"Saved document to {file_path}")

                except Exception as e:
                    logger.error(f"Error saving document {i}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error saving documents: {str(e)}")
            raise

    async def scrape_url_recursively(self, url: str, base_url: str) -> List[Dict[str, str]]:
        """Scrape a URL and its linked pages recursively.

        Args:
            url: URL to scrape
            base_url: Base URL to restrict scraping to

        Returns:
            List of scraped pages with their content
        """
        if url in self._visited_urls or not url.startswith(base_url):
            return []

        self._visited_urls.add(url)
        results = []

        try:
            async with self.web_scraper as scraper:
                # Scrape main content
                content = await scraper.scrape_with_playwright(
                    url=url,
                    selector="#content, .content, main, article, .documentation, .doc-content, .markdown-body",
                    timeout=60000,  # 60 seconds
                )

                if not content:
                    logger.warning(f"No content found at {url}")
                    return []

                title, html_content, text_content = content

                # Convert HTML to Markdown
                h = html2text.HTML2Text()
                h.ignore_links = False
                h.ignore_images = False
                h.body_width = 0
                markdown_content = h.handle(html_content)

                # Fix relative links
                markdown_content = self._fix_relative_links(markdown_content, base_url)

                results.append({"url": url, "title": title, "content": markdown_content})

                # Extract links from the page
                page = await scraper.get_page_links(url)
                if page and page.links:
                    # Filter and normalize links
                    doc_links = []
                    for link in page.links:
                        # Convert relative links to absolute
                        if link.startswith("/"):
                            link = base_url.rstrip("/") + link
                        elif not link.startswith(("http://", "https://")):
                            link = base_url.rstrip("/") + "/" + link.lstrip("/")

                        # Only process documentation links
                        if (
                            link.startswith(base_url)
                            and not link.endswith((".png", ".jpg", ".jpeg", ".gif", ".css", ".js"))
                            and link not in self._visited_urls
                        ):
                            doc_links.append(link)

                    # Sort links for consistent ordering
                    doc_links.sort()

                    # Process each documentation link
                    for link in doc_links:
                        logger.info(f"Processing link: {link}")
                        sub_results = await self.scrape_url_recursively(link, base_url)
                        results.extend(sub_results)

        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")

        return results

    def _fix_relative_links(self, content: str, base_url: str) -> str:
        """Fix relative links in markdown content.

        Args:
            content: Markdown content
            base_url: Base URL for converting relative links

        Returns:
            Content with fixed links
        """

        def fix_link(match):
            link = match.group(2)
            if link.startswith("/"):
                return f'[{match.group(1)}]({base_url.rstrip("/")}{link})'
            elif not link.startswith(("http://", "https://")):
                return f'[{match.group(1)}]({base_url.rstrip("/")}/{link.lstrip("/")})'
            return match.group(0)

        # Fix markdown links: [text](link)
        content = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", fix_link, content)

        return content

    def _clean_content(self, content: str) -> str:
        """Clean the markdown content.

        Args:
            content: Raw markdown content

        Returns:
            Cleaned markdown content
        """
        # 移除隐私政策和广告相关内容
        privacy_patterns = [
            r"(?s)We use .* to collect data.*?Privacy Policy.*?\)",  # 匹配隐私政策段落
            r"(?s)Privacy settings.*?less customized experience\?",  # 匹配隐私设置段落
            r"(?s)cookie.*?preferences",  # 匹配cookie相关内容
            r"Do not sell or share my personal information",  # 移除CCPA相关内容
            r"Privacy/Cookies",  # 移除隐私/Cookie链接
        ]

        cleaned = content
        for pattern in privacy_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

        # 移除空行和多余的空格
        lines = [line.strip() for line in cleaned.split("\n")]
        lines = [line for line in lines if line]

        # 移除重复的标题
        seen_headers = set()
        filtered_lines = []
        for line in lines:
            if line.startswith("#"):
                header = line.lower()
                if header in seen_headers:
                    continue
                seen_headers.add(header)
            filtered_lines.append(line)

        return "\n\n".join(filtered_lines)

    def _format_with_frontmatter(self, frontmatter: Dict[str, Any], content: str) -> str:
        """Format document with YAML frontmatter.

        Args:
            frontmatter: Frontmatter metadata
            content: Document content

        Returns:
            Formatted document
        """
        # Convert frontmatter to YAML
        frontmatter_yaml = "---\n"
        for key, value in frontmatter.items():
            frontmatter_yaml += f"{key}: {value}\n"
        frontmatter_yaml += "---\n\n"

        return frontmatter_yaml + content

    def _generate_filename(self, url: str) -> str:
        """Generate a filename from URL.

        Args:
            url: URL of the page

        Returns:
            Generated filename
        """
        # Extract the last part of the URL path
        path = urlparse(url).path
        name = path.rstrip("/").split("/")[-1] or "index"

        # Remove file extension if present
        name = os.path.splitext(name)[0]

        # Add timestamp to ensure uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        return f"{name}.md_{timestamp}"

    async def execute(self, **kwargs) -> str:
        """Execute the plugin.

        Args:
            kwargs: Keyword arguments from Click

        Returns:
            Execution result as JSON string
        """
        url = kwargs.get("url")
        output_dir = Path(kwargs.get("output_dir"))
        content_query = kwargs.get("content")

        try:
            # Reset visited URLs
            self._visited_urls = set()

            if content_query:
                # 将中文查询转换为英文
                english_query = self._translate_to_english(content_query)
                logger.info(f"Translated query: {english_query}")

                # 使用网站的搜索功能
                relevant_urls = await self._search_documentation(url, english_query)
                logger.info(f"Found {len(relevant_urls)} relevant pages")

                # 抓取相关页面并提取相关内容
                docs = []
                for page_url in relevant_urls:
                    async with self.web_scraper as scraper:
                        content = await scraper.scrape_with_playwright(
                            url=page_url,
                            selector="#content, .content, main, article, .documentation, .doc-content, .markdown-body",
                            timeout=60000,
                        )

                        if content:
                            title, html_content, _ = content

                            # 转换为Markdown
                            h = html2text.HTML2Text()
                            h.ignore_links = False
                            h.ignore_images = False
                            h.body_width = 0
                            markdown_content = h.handle(html_content)

                            # 提取相关章节
                            relevant_content = await self._extract_relevant_sections(markdown_content, english_query)

                            if relevant_content:
                                docs.append({"url": page_url, "title": title, "content": relevant_content})
            else:
                # 如果没有指定内容查询，则抓取整个文档
                docs = await self.scrape_url_recursively(url, url)

            # 保存文档
            if docs:
                await self.save_docs(docs, output_dir)

                response_data = {
                    "data": {
                        "url": url,
                        "pages_scraped": len(docs),
                        "output_dir": str(output_dir),
                        "query": content_query,
                    },
                    "message": f"Successfully scraped {len(docs)} pages from {url}",
                }
            else:
                response_data = {"data": {"url": url, "query": content_query}, "message": "No relevant content found"}

            return json.dumps(response_data, indent=2, ensure_ascii=False)

        except Exception as e:
            error_data = {"error": str(e), "url": url, "query": content_query}
            logger.error(f"Error: {str(e)}")
            return json.dumps(error_data, indent=2, ensure_ascii=False)

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin.

        Returns:
            Click command
        """

        @click.command(name=self.name, help=self.description)
        @click.argument("url")
        @click.argument("output_dir")
        @click.option("--content", help="Search query to find specific content in the documentation")
        def command(url, output_dir, content):
            """Scrape and search API documentation.

            Args:
                url: URL of the API documentation to scrape
                output_dir: Directory to save the scraped documentation
                content: Search query to find specific content
            """
            return asyncio.run(self.execute(url=url, output_dir=output_dir, content=content))

        return command
