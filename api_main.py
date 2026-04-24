"""
Entry point for Gaprio API Server.

Run with: gaprio-api or python -m gaprio.api_main
"""

import logging
import sys

import uvicorn

from gaprio.config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger(__name__)


def main():
    """Run the Gaprio API server."""
    logger.info("=" * 50)
    logger.info("Starting Gaprio API Server")
    logger.info("=" * 50)
    
    # Default config
    host = "0.0.0.0"
    port = 8000
    
    logger.info(f"Server will be available at: http://localhost:{port}")
    logger.info(f"API docs at: http://localhost:{port}/docs")
    logger.info(f"Chat UI at: http://localhost:{port}/")
    
    uvicorn.run(
        "gaprio.api:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
