"""
Flask HTTP Server for LanceDB Operations

This module provides a RESTful API server for LanceDB operations including:
- Health checks and database statistics
- Table management (list, info, delete)
- Data ingestion from parquet files
- Vector similarity search
- Complete error handling and JSON responses
"""

from flask import Flask, request, jsonify, Response
import logging
from datetime import datetime
from dataclasses import asdict
import traceback
from pathlib import Path
from typing import Optional

# Import database manager and schemas
from .lance_db_manager import LanceDBManager
from .schemas import SearchResult, DatabaseStats

# Configure logging
logger = logging.getLogger(__name__)


class LanceDBFlaskServer:
    """
    Flask HTTP server for LanceDB operations.
    Provides RESTful API endpoints for database management.
    """
    
    def __init__(self, db_manager: LanceDBManager = None):
        """
        Initialize Flask server with LanceDB manager.
        
        Args:
            db_manager: Existing LanceDB manager instance, or None to create new one
        """
        self.app = Flask(__name__)
        self.db_manager = db_manager or LanceDBManager()
        self.setup_routes()
        self.setup_error_handlers()
    
    def setup_routes(self):
        """Setup Flask routes for API endpoints."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            try:
                return jsonify({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'database_path': self.db_manager.db_path,
                    'embedding_dimension': self.db_manager.embedding_dim
                })
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return jsonify({
                    'status': 'unhealthy',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get database statistics."""
            try:
                stats = self.db_manager.get_database_stats()
                return jsonify(asdict(stats))
            except Exception as e:
                logger.error(f"Failed to get database stats: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tables', methods=['GET'])
        def list_tables():
            """List all tables in the database."""
            try:
                tables = self.db_manager.list_tables()
                return jsonify({
                    'tables': tables,
                    'total_tables': len(tables),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                logger.error(f"Failed to list tables: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tables/<table_name>', methods=['GET'])
        def get_table_info(table_name: str):
            """Get information about a specific table."""
            try:
                info = self.db_manager.get_table_info(table_name)
                return jsonify(info)
            except Exception as e:
                logger.error(f"Failed to get table info for {table_name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tables/<table_name>', methods=['DELETE'])
        def delete_table(table_name: str):
            """Delete a table."""
            try:
                success = self.db_manager.delete_table(table_name)
                if success:
                    return jsonify({
                        'message': f'Table {table_name} deleted successfully',
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    return jsonify({
                        'error': f'Failed to delete table {table_name}',
                        'timestamp': datetime.now().isoformat()
                    }), 500
            except Exception as e:
                logger.error(f"Failed to delete table {table_name}: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/ingest', methods=['POST'])
        def ingest_parquet():
            """
            Ingest data from a parquet file.
            
            Expected JSON payload:
            {
                "parquet_path": "path/to/file.parquet",
                "table_name": "optional_table_name",
                "overwrite": false,
                "batch_size": 100
            }
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'JSON payload is required'}), 400
                
                if 'parquet_path' not in data:
                    return jsonify({'error': 'parquet_path is required'}), 400
                
                parquet_path = data['parquet_path']
                table_name = data.get('table_name')
                overwrite = data.get('overwrite', False)
                batch_size = data.get('batch_size', 100)
                
                # Validate parquet path exists
                if not Path(parquet_path).exists():
                    return jsonify({
                        'error': f'Parquet file not found: {parquet_path}',
                        'timestamp': datetime.now().isoformat()
                    }), 404
                
                # Validate batch size
                if batch_size < 1 or batch_size > 1000:
                    return jsonify({
                        'error': 'batch_size must be between 1 and 1000',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                logger.info(f"Starting ingestion request: {parquet_path} -> {table_name}")
                
                # Start ingestion
                stats = self.db_manager.populate_from_parquet(
                    parquet_path=parquet_path,
                    table_name=table_name,
                    batch_size=batch_size,
                    overwrite=overwrite
                )
                
                return jsonify({
                    'message': 'Ingestion completed successfully',
                    'stats': stats,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/search', methods=['POST'])
        def search_documents():
            """
            Search for similar documents.
            
            Expected JSON payload:
            {
                "query": "search text",
                "table_name": "table_to_search",
                "limit": 10,
                "score_threshold": 0.0
            }
            """
            try:
                data = request.get_json()
                if not data:
                    return jsonify({'error': 'JSON payload is required'}), 400
                
                if 'query' not in data or 'table_name' not in data:
                    return jsonify({
                        'error': 'query and table_name are required',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                query = data['query']
                table_name = data['table_name']
                limit = data.get('limit', 10)
                score_threshold = data.get('score_threshold', 0.0)
                
                # Validate parameters
                if not query.strip():
                    return jsonify({
                        'error': 'query cannot be empty',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                if limit < 1 or limit > 100:
                    return jsonify({
                        'error': 'limit must be between 1 and 100',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                if not (0.0 <= score_threshold <= 1.0):
                    return jsonify({
                        'error': 'score_threshold must be between 0.0 and 1.0',
                        'timestamp': datetime.now().isoformat()
                    }), 400
                
                logger.info(f"Search request: '{query[:50]}...' in table {table_name}")
                
                # Perform search
                results = self.db_manager.search_similar(
                    query_text=query,
                    table_name=table_name,
                    limit=limit,
                    score_threshold=score_threshold
                )
                
                # Convert results to dictionaries
                results_dict = [asdict(result) for result in results]
                
                return jsonify({
                    'query': query,
                    'table_name': table_name,
                    'total_results': len(results_dict),
                    'limit': limit,
                    'score_threshold': score_threshold,
                    'results': results_dict,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                logger.error(traceback.format_exc())
                return jsonify({
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }), 500
        
        @self.app.route('/', methods=['GET'])
        def api_info():
            """API information endpoint."""
            return jsonify({
                'name': 'LanceDB PSL RAG API',
                'version': '1.0.0',
                'description': 'RESTful API for LanceDB operations with PSL document schema',
                'endpoints': {
                    'health': 'GET /health - Health check',
                    'stats': 'GET /stats - Database statistics',
                    'tables': 'GET /tables - List tables',
                    'table_info': 'GET /tables/{name} - Table information',
                    'delete_table': 'DELETE /tables/{name} - Delete table',
                    'ingest': 'POST /ingest - Ingest parquet data',
                    'search': 'POST /search - Search documents'
                },
                'timestamp': datetime.now().isoformat()
            })
    
    def setup_error_handlers(self):
        """Setup error handlers for the Flask app."""
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({
                'error': 'Endpoint not found',
                'message': 'The requested endpoint does not exist',
                'timestamp': datetime.now().isoformat()
            }), 404
        
        @self.app.errorhandler(405)
        def method_not_allowed(error):
            return jsonify({
                'error': 'Method not allowed',
                'message': 'The HTTP method is not allowed for this endpoint',
                'timestamp': datetime.now().isoformat()
            }), 405
        
        @self.app.errorhandler(500)
        def internal_error(error):
            logger.error(f"Internal server error: {error}")
            return jsonify({
                'error': 'Internal server error',
                'message': 'An unexpected error occurred',
                'timestamp': datetime.now().isoformat()
            }), 500
        
        @self.app.before_request
        def log_request_info():
            """Log incoming requests."""
            logger.info(f"Request: {request.method} {request.path} from {request.remote_addr}")
        
        @self.app.after_request
        def log_response_info(response):
            """Log outgoing responses."""
            logger.info(f"Response: {response.status_code} for {request.method} {request.path}")
            return response
    
    def run(self, host: str = '0.0.0.0', port: int = 5000, debug: bool = False):
        """
        Run the Flask server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            debug: Enable debug mode
        """
        logger.info(f"Starting LanceDB Flask server on {host}:{port}")
        logger.info(f"Database path: {self.db_manager.db_path}")
        logger.info(f"Debug mode: {debug}")
        
        try:
            self.app.run(host=host, port=port, debug=debug)
        except Exception as e:
            logger.error(f"Failed to start Flask server: {e}")
            raise
        finally:
            # Clean up database manager
            if hasattr(self.db_manager, 'close'):
                self.db_manager.close()
    
    def get_app(self) -> Flask:
        """
        Get the Flask app instance for external use (e.g., with gunicorn).
        
        Returns:
            Flask application instance
        """
        return self.app