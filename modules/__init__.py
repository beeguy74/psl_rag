"""
Database module initialization for PSL RAG project.

This module contains the database operations and schema definitions
for the LanceDB vector database implementation.
"""

from .lance_db_manager import LanceDBManager
from .schemas import (
    PSLDocumentRecord,
    DatabaseStats,
    SearchResult,
    ParquetFileMetadata,
    PANDAS_DTYPES,
    validate_document_record,
    analyze_parquet_file
)

__all__ = [
    'LanceDBManager',
    'PSLDocumentRecord',
    'DatabaseStats', 
    'SearchResult',
    'ParquetFileMetadata',
    'PANDAS_DTYPES',
    'validate_document_record',
    'analyze_parquet_file'
]