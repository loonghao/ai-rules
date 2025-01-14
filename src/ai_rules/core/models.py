"""Core models for ai-rules."""

# Import third-party modules
from pydantic import BaseModel


class WebContent(BaseModel):
    """Model for web content."""

    url: str
    title: str
    content: str
    timestamp: str


class WebScraperResponse(BaseModel):
    """Response model for web scraper."""

    content: WebContent
    output_file: str
