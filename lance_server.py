#!/usr/bin/env python3
"""
LanceDB Server Entry Point

This script serves as the main entry point for the LanceDB server with Flask HTTP API.
It uses the modular structure with separate database and server components.

Usage:
    python lance_server.py [options]
    
Examples:
    # Start with default settings
    python lance_server.py
    
    # Start with custom database path and port
    python lance_server.py --db-path ./custom_db --port 8080
    
    # Start in debug mode
    python lance_server.py --debug
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add modules directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import modular components
from modules.lance_db_manager import LanceDBManager
from modules.flask_server import LanceDBFlaskServer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='LanceDB Server with Flask API for PSL RAG Project',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Start with defaults
  %(prog)s --port 8080              # Custom port
  %(prog)s --db-path ./my_db        # Custom database path
  %(prog)s --debug                  # Debug mode
  %(prog)s --embedding-dim 768      # Custom embedding dimension
        """
    )
    
    parser.add_argument('--host', 
                       default='0.0.0.0', 
                       help='Host to bind to (default: 0.0.0.0)')
    
    parser.add_argument('--port', 
                       type=int, 
                       default=5000, 
                       help='Port to bind to (default: 5000)')
    
    parser.add_argument('--debug', 
                       action='store_true', 
                       help='Run in debug mode')
    
    parser.add_argument('--db-path', 
                       help='Path to LanceDB database (default: ./lance_db)')
    
    parser.add_argument('--embedding-dim', 
                       type=int, 
                       default=384, 
                       help='Embedding vector dimension (default: 384)')
    
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level (default: INFO)')
    
    return parser.parse_args()


def setup_logging(log_level: str):
    """Setup logging configuration."""
    logging.getLogger().setLevel(getattr(logging, log_level))
    
    # Set specific loggers
    logging.getLogger('werkzeug').setLevel(logging.WARNING)
    if log_level == 'DEBUG':
        logging.getLogger('modules').setLevel(logging.DEBUG)


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    if args.port < 1 or args.port > 65535:
        errors.append("Port must be between 1 and 65535")
    
    if args.embedding_dim < 1 or args.embedding_dim > 2048:
        errors.append("Embedding dimension must be between 1 and 2048")
    
    if args.db_path:
        db_path = Path(args.db_path)
        if db_path.exists() and db_path.is_file():
            errors.append(f"Database path {args.db_path} exists but is a file, not a directory")
    
    if errors:
        print("Argument validation errors:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def main():
    """Main entry point."""
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Validate arguments
    validate_arguments(args)
    
    # Log startup information
    logger.info("Starting LanceDB Server for PSL RAG Project")
    logger.info(f"Configuration:")
    logger.info(f"  Host: {args.host}")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Debug: {args.debug}")
    logger.info(f"  Database path: {args.db_path or './lance_db'}")
    logger.info(f"  Embedding dimension: {args.embedding_dim}")
    logger.info(f"  Log level: {args.log_level}")
    
    try:
        # Initialize LanceDB manager
        logger.info("Initializing LanceDB manager...")
        db_manager = LanceDBManager(
            db_path=args.db_path,
            embedding_dim=args.embedding_dim
        )
        
        # Get database statistics
        stats = db_manager.get_database_stats()
        logger.info(f"Database initialized:")
        logger.info(f"  Tables: {stats.total_tables}")
        logger.info(f"  Documents: {stats.total_documents}")
        logger.info(f"  Path: {stats.database_path}")
        
        # Initialize Flask server
        logger.info("Initializing Flask server...")
        server = LanceDBFlaskServer(db_manager)
        
        # Start server
        logger.info(f"Server ready! Access at http://{args.host}:{args.port}")
        logger.info("Available endpoints:")
        logger.info("  GET  /health          - Health check")
        logger.info("  GET  /stats           - Database statistics")
        logger.info("  GET  /tables          - List tables")
        logger.info("  POST /ingest          - Ingest parquet data")
        logger.info("  POST /search          - Search documents")
        
        server.run(host=args.host, port=args.port, debug=args.debug)
        
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()