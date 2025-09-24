"""
LanceDB Manager - Database operations for PSL RAG Project

This module handles all LanceDB database operations including:
- Database connection and management
- Parquet file ingestion with PSL schema validation
- Vector search operations
- Table management (create, delete, info)
- Embedding generation (mock implementation)
"""

import lancedb
import pandas as pd
import pyarrow as pa
import numpy as np
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging
from datetime import datetime
import traceback
import os
import uuid
from concurrent.futures import ThreadPoolExecutor

# Import schema definitions
from .schemas import (
    PSLDocumentRecord,
    SearchResult,
    DatabaseStats,
    validate_document_record,
    PANDAS_DTYPES
)

# Configure logging
logger = logging.getLogger(__name__)


class LanceDBManager:
    """
    Comprehensive LanceDB management class for PSL RAG project.
    
    Handles database operations, parquet file ingestion, and vector search
    with support for the PSL document schema.
    """
    
    def __init__(self, 
                 db_path: str = None,
                 embedding_dim: int = 768,  # Default for sentence-transformers
                 create_if_not_exists: bool = True,
                 embedder=None):
        """
        Initialize LanceDB Manager.
        
        Args:
            db_path: Path to LanceDB database directory
            embedding_dim: Dimension of embedding vectors
            create_if_not_exists: Whether to create database if it doesn't exist
            embedder: Embedding generator instance
        """
        self.db_path = db_path or os.getenv('LANCEDB_PATH', './lance_db')
        self.embedding_dim = embedding_dim
        self.db = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.embedder = embedder
        
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
        
        # Generate embedding using provided embedder
        if self.embedder is not None:
            embedding_response = self.embedder.embed([text_for_embedding], self.embedding_dim)
            embedding = embedding_response.embeddings[0].values
        else:
            raise ValueError("No embedder provided to LanceDBManager")
        
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
            
            # Generate query embedding using embedder
            if self.embedder is not None:
                embedding_response = self.embedder.embed([query_text], self.embedding_dim)
                query_embedding = embedding_response.embeddings[0].values
            else:
                raise ValueError("No embedder provided to LanceDBManager")
            
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
    
    def close(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
            logger.info("Database manager cleanup completed")