"""
LanceDB Server with Flask HTTP API for PSL RAG Project

This module provides a comprehensive LanceDB management class with Flask HTTP endpoints
for creating, populating, and querying vector databases using parquet files.

Features:
- Create new or connect to existing LanceDB databases
- Populate databases from parquet files with PSL document schema
- Vector search and similarity queries
- RESTful API endpoints via Flask
- Support for both sync and async operations
- Comprehensive error handling and logging

Usage:
    # Start the server
    python lance_db_server.py
    
    # Or use as a class
    from lance_db_server import LanceDBManager
    manager = LanceDBManager()
    manager.populate_from_parquet('data.parquet')
"""

import lancedb
import pandas as pd
import pyarrow as pa
import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from flask import Flask, request, jsonify, Response
import logging
from datetime import datetime
import json
from dataclasses import dataclass, asdict
import traceback
from dotenv import load_dotenv
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

# Import our schema definitions
from parquet_schemas import (
    PSLDocumentRecord, 
    PANDAS_DTYPES, 
    validate_document_record,
    analyze_parquet_file
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a search result from LanceDB"""
    id: str
    score: float
    source_file: str
    text: str
    word_count: int
    embedding_summary: str
    
    
@dataclass
class DatabaseStats:
    """Statistics about the LanceDB database"""
    total_tables: int
    table_names: List[str]
    total_documents: int
    database_path: str
    created_at: Optional[str] = None


class LanceDBManager:
    """
    Comprehensive LanceDB management class for PSL RAG project.
    
    Handles database operations, parquet file ingestion, and vector search
    with support for the PSL document schema.
    """
    
    def __init__(self, 
                 db_path: str = None,
                 embedding_dim: int = 384,  # Default for sentence-transformers
                 create_if_not_exists: bool = True):
        """
        Initialize LanceDB Manager.
        
        Args:
            db_path: Path to LanceDB database directory
            embedding_dim: Dimension of embedding vectors
            create_if_not_exists: Whether to create database if it doesn't exist
        """
        self.db_path = db_path or os.getenv('LANCEDB_PATH', './lance_db')
        self.embedding_dim = embedding_dim
        self.db = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Ensure database path exists
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            self.connect_database()
            logger.info(f"Connected to LanceDB at {self.db_path}")
        except Exception as e:
            if create_if_not_exists:
                logger.info(f"Creating new LanceDB at {self.db_path}")
                self.connect_database()
            else:
                raise e
    
    def connect_database(self) -> None:
        """Connect to or create LanceDB database."""
        try:
            self.db = lancedb.connect(self.db_path)
            logger.info(f"Successfully connected to LanceDB: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to LanceDB: {e}")
            raise
    
    def get_database_stats(self) -> DatabaseStats:
        """Get statistics about the current database."""
        try:
            table_names = self.db.table_names()
            total_documents = 0
            
            for table_name in table_names:
                try:
                    table = self.db.open_table(table_name)
                    total_documents += len(table.to_pandas())
                except Exception as e:
                    logger.warning(f"Could not count documents in table {table_name}: {e}")
            
            return DatabaseStats(
                total_tables=len(table_names),
                table_names=table_names,
                total_documents=total_documents,
                database_path=self.db_path,
                created_at=datetime.now().isoformat()
            )
        except Exception as e:
            logger.error(f"Failed to get database stats: {e}")
            raise
    
    def create_table_schema(self) -> pa.Schema:
        """
        Create PyArrow schema for PSL documents with embedding vector.
        
        Returns:
            PyArrow schema with all required fields including vector
        """
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("source_file", pa.string()),
            pa.field("text", pa.string()),
            pa.field("word_count", pa.int64()),
            pa.field("embedding_summary", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=self.embedding_dim)),
            pa.field("created_at", pa.timestamp('ms'))
        ])
    
    def generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate a mock embedding vector for testing purposes.
        In production, replace this with actual embedding model.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            Mock embedding vector
        """
        # Simple hash-based mock embedding for consistent results
        import hashlib
        hash_obj = hashlib.md5(text.encode())
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to numbers and normalize
        numbers = [int(hash_hex[i:i+2], 16) for i in range(0, len(hash_hex), 2)]
        
        # Pad or truncate to desired dimension
        while len(numbers) < self.embedding_dim:
            numbers.extend(numbers[:self.embedding_dim - len(numbers)])
        numbers = numbers[:self.embedding_dim]
        
        # Normalize to [-1, 1] range
        embedding = [(num - 127.5) / 127.5 for num in numbers]
        return embedding
    
    def prepare_document_for_insertion(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare a document record for insertion into LanceDB.
        
        Args:
            record: Document record from parquet file
            
        Returns:
            Prepared record with embedding and metadata
        """
        # Validate the record
        is_valid, errors = validate_document_record(record)
        if not is_valid:
            raise ValueError(f"Invalid document record: {errors}")
        
        # Use embedding_summary for embedding if available, otherwise use text
        text_for_embedding = record.get('embedding_summary', '').strip()
        if not text_for_embedding:
            text_for_embedding = record.get('text', '')[:1000]  # Limit for performance
        
        # Generate embedding (replace with actual embedding model in production)
        embedding = self.generate_mock_embedding(text_for_embedding)
        
        # Prepare record with all required fields
        # Use timestamp with millisecond precision to avoid LanceDB casting errors
        created_at = pd.Timestamp.now().floor('ms')
        
        prepared_record = {
            'id': str(uuid.uuid4()),
            'source_file': record['source_file'],
            'text': record['text'],
            'word_count': int(record['word_count']),
            'embedding_summary': record['embedding_summary'],
            'vector': embedding,
            'created_at': created_at
        }
        
        return prepared_record
    
    def create_or_get_table(self, table_name: str, overwrite: bool = False) -> lancedb.table.Table:
        """
        Create a new table or get existing one.
        
        Args:
            table_name: Name of the table to create/get
            overwrite: Whether to overwrite existing table
            
        Returns:
            LanceDB table instance
        """
        try:
            # Check if table exists
            existing_tables = self.db.table_names()
            
            if table_name in existing_tables:
                if overwrite:
                    logger.info(f"Dropping existing table: {table_name}")
                    self.db.drop_table(table_name)
                else:
                    logger.info(f"Opening existing table: {table_name}")
                    return self.db.open_table(table_name)
            
            # Create new table with schema
            schema = self.create_table_schema()
            table = self.db.create_table(table_name, schema=schema)
            logger.info(f"Created new table: {table_name}")
            
            return table
            
        except Exception as e:
            logger.error(f"Failed to create/get table {table_name}: {e}")
            raise
    
    def populate_from_parquet(self, 
                             parquet_path: str,
                             table_name: str = None,
                             batch_size: int = 100,
                             overwrite: bool = False) -> Dict[str, Any]:
        """
        Populate LanceDB table from a parquet file.
        
        Args:
            parquet_path: Path to parquet file
            table_name: Name for the table (derived from filename if None)
            batch_size: Number of records to process in each batch
            overwrite: Whether to overwrite existing table
            
        Returns:
            Dictionary with ingestion statistics
        """
        try:
            # Validate parquet file exists
            parquet_path = Path(parquet_path)
            if not parquet_path.exists():
                raise FileNotFoundError(f"Parquet file not found: {parquet_path}")
            
            # Determine table name
            if not table_name:
                table_name = parquet_path.stem.replace('_data', '')
            
            logger.info(f"Starting ingestion from {parquet_path} to table {table_name}")
            
            # Load parquet file
            df = pd.read_parquet(parquet_path)
            logger.info(f"Loaded {len(df)} records from parquet file")
            
            # Validate parquet schema
            required_columns = ['source_file', 'text', 'word_count', 'embedding_summary']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns in parquet: {missing_columns}")
            
            # Create or get table
            table = self.create_or_get_table(table_name, overwrite=overwrite)
            
            # Process records in batches
            total_processed = 0
            total_errors = 0
            batch_records = []
            
            for idx, row in df.iterrows():
                try:
                    # Convert row to dict and prepare for insertion
                    record_dict = row.to_dict()
                    prepared_record = self.prepare_document_for_insertion(record_dict)
                    batch_records.append(prepared_record)
                    
                    # Process batch when full
                    if len(batch_records) >= batch_size:
                        try:
                            # Convert to DataFrame and then to PyArrow table for better type control
                            batch_df = pd.DataFrame(batch_records)
                            # Ensure timestamp is in milliseconds
                            if 'created_at' in batch_df.columns:
                                batch_df['created_at'] = pd.to_datetime(batch_df['created_at']).dt.floor('ms')
                            
                            table.add(batch_df)
                            total_processed += len(batch_records)
                            logger.info(f"Processed batch: {total_processed} records")
                        except Exception as batch_error:
                            logger.error(f"Failed to add batch: {batch_error}")
                            total_errors += len(batch_records)
                        
                        batch_records = []
                
                except Exception as record_error:
                    logger.warning(f"Failed to process record {idx}: {record_error}")
                    total_errors += 1
            
            # Process remaining records
            if batch_records:
                try:
                    # Convert to DataFrame and then to PyArrow table for better type control
                    batch_df = pd.DataFrame(batch_records)
                    # Ensure timestamp is in milliseconds
                    if 'created_at' in batch_df.columns:
                        batch_df['created_at'] = pd.to_datetime(batch_df['created_at']).dt.floor('ms')
                    
                    table.add(batch_df)
                    total_processed += len(batch_records)
                    logger.info(f"Processed final batch: {total_processed} total records")
                except Exception as batch_error:
                    logger.error(f"Failed to add final batch: {batch_error}")
                    total_errors += len(batch_records)
            
            # Create vector index for better search performance
            try:
                if total_processed > 50:  # Only create index for larger tables
                    logger.info("Creating vector index for improved search performance...")
                    table.create_index(num_sub_vectors=min(16, total_processed // 10))
                    logger.info("Vector index created successfully")
            except Exception as index_error:
                logger.warning(f"Failed to create vector index: {index_error}")
            
            stats = {
                'table_name': table_name,
                'parquet_file': str(parquet_path),
                'total_records': len(df),
                'processed_records': total_processed,
                'error_records': total_errors,
                'success_rate': (total_processed / len(df)) * 100 if len(df) > 0 else 0,
                'ingestion_time': datetime.now().isoformat()
            }
            
            logger.info(f"Ingestion completed: {stats}")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to populate from parquet {parquet_path}: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def search_similar(self, 
                      query_text: str,
                      table_name: str,
                      limit: int = 10,
                      score_threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for similar documents in the specified table.
        
        Args:
            query_text: Text to search for
            table_name: Name of table to search in
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results
        """
        try:
            # Open table
            table = self.db.open_table(table_name)
            
            # Generate query embedding
            query_embedding = self.generate_mock_embedding(query_text)
            
            # Perform vector search
            results = (table
                      .search(query_embedding)
                      .limit(limit)
                      .to_pandas())
            
            # Convert to SearchResult objects
            search_results = []
            for _, row in results.iterrows():
                if row.get('_distance', 0) >= score_threshold:
                    search_result = SearchResult(
                        id=row['id'],
                        score=1.0 - row.get('_distance', 0),  # Convert distance to similarity
                        source_file=row['source_file'],
                        text=row['text'][:500] + "..." if len(row['text']) > 500 else row['text'],
                        word_count=row['word_count'],
                        embedding_summary=row['embedding_summary']
                    )
                    search_results.append(search_result)
            
            logger.info(f"Search completed: found {len(search_results)} results for '{query_text[:50]}...'")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed in table {table_name}: {e}")
            raise
    
    def list_tables(self) -> List[str]:
        """List all tables in the database."""
        try:
            return self.db.table_names()
        except Exception as e:
            logger.error(f"Failed to list tables: {e}")
            raise
    
    def get_table_info(self, table_name: str) -> Dict[str, Any]:
        """Get information about a specific table."""
        try:
            table = self.db.open_table(table_name)
            df = table.to_pandas()
            
            return {
                'table_name': table_name,
                'total_documents': len(df),
                'schema': str(table.schema),
                'columns': list(df.columns),
                'sample_records': df.head(3).to_dict('records') if len(df) > 0 else []
            }
        except Exception as e:
            logger.error(f"Failed to get table info for {table_name}: {e}")
            raise
    
    def delete_table(self, table_name: str) -> bool:
        """Delete a table from the database."""
        try:
            self.db.drop_table(table_name)
            logger.info(f"Deleted table: {table_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete table {table_name}: {e}")
            return False


class LanceDBFlaskServer:
    """
    Flask HTTP server for LanceDB operations.
    Provides RESTful API endpoints for database management.
    """
    
    def __init__(self, db_manager: LanceDBManager = None):
        """Initialize Flask server with LanceDB manager."""
        self.app = Flask(__name__)
        self.db_manager = db_manager or LanceDBManager()
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for API endpoints."""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint."""
            return jsonify({
                'status': 'healthy',
                'timestamp': datetime.now().isoformat(),
                'database_path': self.db_manager.db_path
            })
        
        @self.app.route('/stats', methods=['GET'])
        def get_stats():
            """Get database statistics."""
            try:
                stats = self.db_manager.get_database_stats()
                return jsonify(asdict(stats))
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tables', methods=['GET'])
        def list_tables():
            """List all tables in the database."""
            try:
                tables = self.db_manager.list_tables()
                return jsonify({'tables': tables})
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tables/<table_name>', methods=['GET'])
        def get_table_info(table_name):
            """Get information about a specific table."""
            try:
                info = self.db_manager.get_table_info(table_name)
                return jsonify(info)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/tables/<table_name>', methods=['DELETE'])
        def delete_table(table_name):
            """Delete a table."""
            try:
                success = self.db_manager.delete_table(table_name)
                if success:
                    return jsonify({'message': f'Table {table_name} deleted successfully'})
                else:
                    return jsonify({'error': f'Failed to delete table {table_name}'}), 500
            except Exception as e:
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
                if not data or 'parquet_path' not in data:
                    return jsonify({'error': 'parquet_path is required'}), 400
                
                parquet_path = data['parquet_path']
                table_name = data.get('table_name')
                overwrite = data.get('overwrite', False)
                batch_size = data.get('batch_size', 100)
                
                # Validate parquet path exists
                if not Path(parquet_path).exists():
                    return jsonify({'error': f'Parquet file not found: {parquet_path}'}), 404
                
                # Start ingestion
                stats = self.db_manager.populate_from_parquet(
                    parquet_path=parquet_path,
                    table_name=table_name,
                    batch_size=batch_size,
                    overwrite=overwrite
                )
                
                return jsonify({
                    'message': 'Ingestion completed successfully',
                    'stats': stats
                })
                
            except Exception as e:
                logger.error(f"Ingestion failed: {e}")
                return jsonify({'error': str(e)}), 500
        
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
                if not data or 'query' not in data or 'table_name' not in data:
                    return jsonify({'error': 'query and table_name are required'}), 400
                
                query = data['query']
                table_name = data['table_name']
                limit = data.get('limit', 10)
                score_threshold = data.get('score_threshold', 0.0)
                
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
                    'results': results_dict
                })
                
            except Exception as e:
                logger.error(f"Search failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server."""
        logger.info(f"Starting LanceDB Flask server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def main():
    """Main function to run the server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='LanceDB Server with Flask API')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    parser.add_argument('--db-path', help='Path to LanceDB database')
    parser.add_argument('--embedding-dim', type=int, default=384, help='Embedding dimension')
    
    args = parser.parse_args()
    
    # Initialize LanceDB manager
    db_manager = LanceDBManager(
        db_path=args.db_path,
        embedding_dim=args.embedding_dim
    )
    
    # Initialize and run Flask server
    server = LanceDBFlaskServer(db_manager)
    server.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()