"""DuckDuckGo image search and download plugin."""

# Import built-in modules
import asyncio
import json
import logging
import os
from typing import Any, Dict, List

import aiohttp

# Import third-party modules
import click
from duckduckgo_search import DDGS
from pydantic import BaseModel

from ai_rules.core.config import get_images_dir

# Import local modules
from ai_rules.core.plugin import Plugin

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


class ImageResult(BaseModel):
    """Model for image results."""

    title: str
    image_url: str
    thumbnail_url: str
    source_url: str
    source: str
    height: int
    width: int
    local_path: str = ""


class ImageResponse(BaseModel):
    """Response model for image search."""

    results: List[ImageResult]
    total: int
    download_dir: str


class ImagePlugin(Plugin):
    """DuckDuckGo image search and download plugin."""

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "image_search"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search and download images using DuckDuckGo"

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin.

        Returns:
            Click command
        """

        @click.command(name="image", help="Search and download images from various sources.")
        @click.argument("query")
        @click.argument("output_dir")
        @click.option("--max-results", default=10, help="Maximum number of images to download (default: 10)")
        @click.option(
            "--size",
            type=click.Choice(["small", "medium", "large"]),
            default="medium",
            help="Size of images to search for (default: medium)",
        )
        def command(query, output_dir, max_results, size):
            """Search and download images.

            Args:
                query: Search query for finding images
                output_dir: Directory to save downloaded images
                max_results: Maximum number of images to download
                size: Size of images to search for
            """
            return asyncio.run(self.execute(query=query, output_dir=output_dir, max_results=max_results, size=size))

        return command

    async def download_image(self, session: aiohttp.ClientSession, image: ImageResult, download_dir: str) -> None:
        """Download an image.

        Args:
            session: aiohttp client session
            image: Image result to download
            download_dir: Directory to save the image
        """
        try:
            # Create filename from title
            filename = "".join(x for x in image.title if x.isalnum() or x in "._- ")
            filename = f"{filename[:50]}.jpg"  # Truncate long filenames
            filepath = os.path.join(download_dir, filename)

            async with session.get(image.image_url) as response:
                if response.status == 200:
                    with open(filepath, "wb") as f:
                        f.write(await response.read())
                    image.local_path = filepath
                    logger.debug("Downloaded image to %s", filepath)
                else:
                    logger.error("Failed to download image: %s", response.status)

        except Exception as e:
            logger.error("Error downloading image: %s", str(e))

    async def download_images(self, images: List[ImageResult], download_dir: str) -> None:
        """Download multiple images concurrently.

        Args:
            images: List of image results to download
            download_dir: Directory to save the images
        """
        os.makedirs(download_dir, exist_ok=True)

        async with aiohttp.ClientSession() as session:
            tasks = []
            for image in images:
                task = asyncio.create_task(self.download_image(session, image, download_dir))
                tasks.append(task)
            await asyncio.gather(*tasks)

    def execute(self, **kwargs) -> str:
        """Execute image search and download.

        Args:
            **kwargs: Keyword arguments
                query: Search query
                max_results: Maximum number of results to return
                download_dir: Directory to save downloaded images
                type: Type of images to search for
                size: Size of images to search for
                layout: Layout of images to search for

        Returns:
            Formatted string containing image results
        """
        try:
            query = kwargs.get("query")
            max_results = kwargs.get("max_results", 10)
            download_dir = kwargs.get("download_dir")

            # Use configured images directory if no download directory specified
            if not download_dir:
                download_dir = str(get_images_dir())

            logger.debug("Searching images with query: %s", query)
            # Perform image search
            results = []
            with DDGS() as ddgs:
                logger.debug("Searching images with query: %s", query)
                for r in ddgs.images(
                    query,
                    max_results=max_results,
                ):
                    logger.debug("Raw image result: %s", r)
                    result = ImageResult(
                        title=r.get("title", ""),
                        image_url=r.get("image", ""),
                        thumbnail_url=r.get("thumbnail", ""),
                        source_url=r.get("url", ""),
                        source=r.get("source", ""),
                        height=r.get("height", 0),
                        width=r.get("width", 0),
                    )
                    logger.debug("Parsed image result: %s", result)
                    results.append(result)

            # Download images
            if results:
                asyncio.run(self.download_images(results, download_dir))

            # Create response
            response = ImageResponse(results=results, total=len(results), download_dir=download_dir)

            # Format response
            formatted_response = self.format_response(
                data=response.model_dump(),
                message=f"Found and downloaded {len(results)} images for '{query}' to {download_dir}",
            )

            # Print response
            print(formatted_response)
            return formatted_response

        except Exception as e:
            logger.error("Error executing image search: %s", str(e))
            error_response = self.format_error(str(e))
            print(error_response)
            return error_response

    def format_response(self, data: Dict[str, Any], message: str = "") -> str:
        """Format response data as JSON string.

        Args:
            data: Dictionary containing response data.
            message: Optional message to include in the response.

        Returns:
            JSON string containing formatted response data.
        """
        response_data = {"data": data}
        if message:
            response_data["message"] = message
        return json.dumps(response_data, indent=2, ensure_ascii=False)

    def format_error(self, error: str) -> str:
        """Format error message as JSON string.

        Args:
            error: Error message to include in the response.

        Returns:
            JSON string containing formatted error message.
        """
        return json.dumps({"error": error}, indent=2, ensure_ascii=False)

    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata.

        Returns:
            Dictionary containing plugin metadata.
        """
        return {
            "name": self.name,
            "description": self.description,
            "version": "1.0.0",
            "author": "AI Rules Team",
        }
