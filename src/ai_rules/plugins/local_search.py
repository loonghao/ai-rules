"""Local search plugin for searching downloaded content."""

# Import built-in modules
import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import third-party modules
import click
import chromadb
from chromadb.config import Settings
from pydantic import BaseModel, Field

# Import local modules
from ai_rules.core.plugin import Plugin
from ai_rules.core.config import get_app_dir

# Configure logger
logger: logging.Logger = logging.getLogger(__name__)


class ContentResult(BaseModel):
    """Model for content search results."""

    file_path: str
    file_type: str
    last_modified: str
    size: int
    score: float = Field(default=0.0, description="Similarity score")
    content_preview: str = Field(default="", description="Preview of the matched content")


class ContentResponse(BaseModel):
    """Response model for content search."""

    results: List[ContentResult]
    total: int


class LocalSearchPlugin(Plugin):
    """Local search plugin for searching downloaded content."""

    def __init__(self) -> None:
        """Initialize plugin."""
        super().__init__()
        self.db_path = get_app_dir() / "local-search" / "chroma"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.db_path))
        try:
            self.client.delete_collection("local_search")
        except:
            pass
        self.collection = self.client.create_collection("local_search")

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "local"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search downloaded content in local directories using vector similarity"

    def _index_file(self, file: Path) -> None:
        """Index a single file."""
        try:
            content = ""
            if file.suffix == ".md":
                with open(file, "r", encoding="utf-8") as f:
                    content = f.read()
            elif file.suffix == ".json":
                with open(file, "r", encoding="utf-8") as f:
                    content = json.dumps(json.load(f), ensure_ascii=False)
            else:
                content = file.stem  # For image files, just use filename

            stat = file.stat()
            metadata = {
                "file_path": str(file),
                "file_type": file.suffix[1:] if file.suffix else "unknown",
                "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "size": stat.st_size
            }

            logger.debug("Indexing file: %s", file)
            self.collection.add(
                documents=[content],
                metadatas=[metadata],
                ids=[str(file)]
            )
            logger.debug("Successfully indexed file: %s", file)

        except Exception as e:
            logger.error("Error indexing file %s: %s", file, e)

    def _index_directory(self, directory: Path, recursive: bool = True) -> None:
        """Index all files in a directory.

        Args:
            directory: Directory to index
            recursive: Whether to index recursively
        """
        if not directory.exists():
            return

        pattern = "**/*" if recursive else "*"
        files = []
        for ext in [".md", ".json", ".jpg", ".png"]:
            files.extend(directory.glob(f"{pattern}{ext}"))

        documents = []
        ids = []
        metadatas = []
        for file in files:
            doc = self._index_file(file)
            if doc:
                documents.append(doc["document"])
                ids.append(doc["id"])
                metadatas.append(doc["metadata"])

        if documents:
            self.collection.add(
                documents=documents,
                ids=ids,
                metadatas=metadatas
            )
            logger.info("Indexed %d files in %s", len(documents), directory)

    def _get_content_preview(self, content: str, query: str, context_chars: int = 100) -> str:
        """Get a preview of the content around the query.

        Args:
            content: Full content text
            query: Search query
            context_chars: Number of characters to show before and after match

        Returns:
            Content preview
        """
        try:
            lower_content = content.lower()
            lower_query = query.lower()
            
            # Find the position of the query in the content
            pos = lower_content.find(lower_query)
            if pos == -1:
                return content[:200] + "..."  # If exact match not found, return start of content
                
            # Calculate preview bounds
            start = max(0, pos - context_chars)
            end = min(len(content), pos + len(query) + context_chars)
            
            # Add ellipsis if needed
            prefix = "..." if start > 0 else ""
            suffix = "..." if end < len(content) else ""
            
            return prefix + content[start:end] + suffix
            
        except Exception as e:
            logger.error("Error getting content preview: %s", str(e))
            return ""

    @property
    def click_command(self) -> click.Command:
        """Get the click command for this plugin."""
        @click.command(name="local", help=self.description)
        @click.argument("path", type=click.Path(exists=True))
        @click.argument("query", nargs=-1)  # Allow multiple words in query
        @click.option(
            "--recursive/--no-recursive",
            default=True,
            help="Index recursively in subdirectories (default: True)",
        )
        @click.option(
            "--reindex/--no-reindex",
            default=False,
            help="Force reindexing of files (default: False)",
        )
        @click.option(
            "--max-results",
            default=10,
            help="Maximum number of results to return (default: 10)",
        )
        @click.option(
            "--threshold",
            default=0.3, # Changed from 0.5 to 0.3
            help="Similarity threshold (0-1, default: 0.3)",
        )
        def local_search(
            path: str,
            query: tuple,  # Changed from str to tuple
            recursive: bool,
            reindex: bool,
            max_results: int,
            threshold: float,
        ) -> str:
            """Execute local search."""
            # Join multiple words into a single query string
            query_str = " ".join(query)
            return asyncio.run(
                self.execute(
                    path=path,
                    query=query_str,
                    recursive=recursive,
                    reindex=reindex,
                    max_results=max_results,
                    threshold=threshold,
                )
            )

        return local_search

    async def execute(self, **kwargs) -> str:
        """Execute content search."""
        try:
            path = kwargs.get("path")
            query = kwargs.get("query")
            recursive = kwargs.get("recursive", True)
            reindex = kwargs.get("reindex", False)
            max_results = kwargs.get("max_results", 10)
            threshold = kwargs.get("threshold", 0.3) # Changed from 0.5 to 0.3

            directory = Path(path)
            if not directory.exists():
                raise click.ClickException(f"Directory {path} does not exist")

            # Index directory if needed
            if reindex:
                self._index_directory(directory, recursive)

            # Search for similar documents
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Filter and format results
            content_results = []
            if results["metadatas"] and results["metadatas"][0]:  # Check if we have any results
                for i, metadata in enumerate(results["metadatas"][0]):
                    distance = results["distances"][0][i]
                    score = 1.0 - distance  # Convert distance to similarity score
                    if score >= threshold:
                        content_preview = self._get_content_preview(results["documents"][0][i], query)
                        
                        content_results.append(
                            ContentResult(
                                file_path=metadata["file_path"],
                                file_type=metadata["file_type"],
                                last_modified=metadata["last_modified"],
                                size=metadata["size"],
                                score=score,
                                content_preview=content_preview
                            )
                        )

            # Create response
            response = ContentResponse(
                results=content_results,
                total=len(content_results)
            )

            # Format response
            formatted_response = self.format_response(
                data=response.model_dump(),
                message=f"Found {len(content_results)} results for '{query}' in '{path}'"
            )

            print(formatted_response)
            return formatted_response

        except Exception as e:
            error_msg = f"Error executing local search: {str(e)}"
            logger.error(error_msg)
            raise click.ClickException(error_msg)

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
