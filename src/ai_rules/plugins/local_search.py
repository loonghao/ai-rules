"""Local search plugin for searching files in a directory."""

import fnmatch
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import click
from whoosh.analysis import StemmingAnalyzer
from whoosh.fields import ID, STORED, TEXT, Schema
from whoosh.index import create_in, exists_in, open_dir
from whoosh.qparser import MultifieldParser, QueryParser
from whoosh.writing import AsyncWriter

from ai_rules.core.plugin import Plugin

logger = logging.getLogger(__name__)


class LocalSearchPlugin(Plugin):
    """Plugin for searching local files using Whoosh.

    This plugin provides fast and efficient local file searching capabilities using
    the Whoosh search engine. It supports:
    - Full-text search within files
    - File path based search
    - Fielded search (search by path or content)
    - Fuzzy matching and stemming
    """

    def __init__(self) -> None:
        """Initialize the plugin."""
        super().__init__()
        self.plugin_dir = None
        self.index_dir = None
        self._index = None
        # Use StemmingAnalyzer for better search results
        self._analyzer = StemmingAnalyzer()
        self._schema = Schema(path=ID(stored=True, unique=True), filename=STORED, content=TEXT(analyzer=self._analyzer))

    @property
    def name(self) -> str:
        """Get plugin name."""
        return "local_search"

    @property
    def description(self) -> str:
        """Get plugin description."""
        return "Search files in a directory using Whoosh"

    @property
    def click_command(self) -> click.Command:
        """Get click command for this plugin."""

        @click.command()
        @click.argument("directory", type=click.Path(exists=True))
        @click.argument("query")
        @click.option("--pattern", help="File pattern to filter results")
        @click.option(
            "--field", type=click.Choice(["content", "path", "both"]), default="both", help="Field to search in"
        )
        async def search(directory: str, query: str, pattern: Optional[str] = None, field: str = "both") -> None:
            """Search files in a directory.

            Args:
                directory: Directory to search in
                query: Search query
                pattern: Optional file pattern to filter results
                field: Field to search in (content, path, or both)
            """
            results = await self.execute(directory=directory, query=query, pattern=pattern, field=field)
            for result in results:
                click.echo(f"{result['file']} (score: {result['score']})")

        return search

    def _init_index(self) -> None:
        """Initialize the Whoosh index."""
        if self._index is not None:
            return

        # Create index directory if it doesn't exist
        os.makedirs(self.index_dir, exist_ok=True)

        # Create or open index
        if exists_in(self.index_dir):
            self._index = open_dir(self.index_dir)
        else:
            self._index = create_in(self.index_dir, self._schema)

    async def execute(
        self, directory: str, query: str, pattern: Optional[str] = None, field: str = "both"
    ) -> List[Dict[str, Any]]:
        """Execute search query.

        Args:
            directory: Directory to search in
            query: Search query
            pattern: Optional file pattern to filter results
            field: Field to search in (content, path, or both)

        Returns:
            List of search results with file paths and scores

        Raises:
            ValueError: If directory does not exist or query is empty
        """
        # Convert to Path object for cross-platform compatibility
        directory = str(Path(directory))

        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        if not query:
            raise ValueError("Query cannot be empty")

        self.plugin_dir = directory
        self.index_dir = os.path.join(directory, ".whoosh_index")

        try:
            # Initialize index and index directory
            self._init_index()
            await self.index_directory(directory, pattern)
        except ValueError as e:
            # Re-raise directory not found error
            raise ValueError(f"Directory does not exist: {directory}") from e

        # Create query parser based on search field
        if field == "content":
            parser = QueryParser("content", schema=self._schema)
        elif field == "path":
            parser = QueryParser("path", schema=self._schema)
        else:  # both
            parser = MultifieldParser(["content", "path"], schema=self._schema)

        # Parse query
        q = parser.parse(query)

        # Search
        with self._index.searcher() as searcher:
            results = searcher.search(q)
            search_results = []
            for result in results:
                # Apply pattern filter if specified
                if pattern:
                    if not fnmatch.fnmatch(result["filename"], pattern):
                        continue
                search_results.append({"file": result["path"], "score": result.score, "filename": result["filename"]})
            return search_results

    async def index_directory(self, directory: str, pattern: Optional[str] = None) -> None:
        """Index all files in a directory.

        Args:
            directory: Directory to index
            pattern: Optional file pattern to filter files

        Raises:
            ValueError: If directory does not exist
        """
        # Convert to Path object for cross-platform compatibility
        directory = str(Path(directory))

        if not os.path.isdir(directory):
            raise ValueError(f"Directory does not exist: {directory}")

        self.plugin_dir = directory
        self.index_dir = os.path.join(directory, ".whoosh_index")

        # Initialize index
        self._init_index()

        # Index files
        writer = AsyncWriter(self._index)
        try:
            for root, _, files in os.walk(directory):
                for file in files:
                    # Skip index directory files
                    if ".whoosh_index" in root:
                        continue

                    if pattern and not fnmatch.fnmatch(file, pattern):
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read()

                        writer.add_document(path=file_path, filename=file, content=content)
                        logger.debug(f"Indexed file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to index file {file_path}: {e}")
        finally:
            writer.commit()  # Not async anymore

    def close(self) -> None:
        """Close the index."""
        if self._index is not None:
            self._index = None
