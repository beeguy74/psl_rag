# LanceDB Server with Flask HTTP API

A comprehensive LanceDB management system with Flask HTTP endpoints for creating, populating, and querying vector databases using PSL RAG parquet files.

## Features

- **LanceDB Management**: Create, connect to, and manage LanceDB vector databases
- **Parquet Integration**: Seamlessly ingest data from PSL RAG parquet files
- **Vector Search**: Perform similarity searches using embeddings
- **REST API**: Complete Flask HTTP API for remote operations
- **Schema Validation**: Built-in validation for PSL document schema
- **Batch Processing**: Efficient batch ingestion with configurable batch sizes
- **Error Handling**: Comprehensive error handling and logging
- **Performance Optimization**: Automatic vector indexing for improved search performance

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements_lance.txt

# Or install individual packages
pip install lancedb pandas pyarrow flask python-dotenv numpy requests
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit configuration as needed
nano .env
```

### 3. Basic Usage

#### Using as a Python Class

```python
from lance_db_server import LanceDBManager

# Initialize manager
manager = LanceDBManager(db_path="./my_lance_db")

# Populate from parquet file
stats = manager.populate_from_parquet(
    parquet_path="data/processed_parquets/masters_data.parquet",
    table_name="masters",
    overwrite=True
)

# Search for similar documents
results = manager.search_similar(
    query_text="finance program admission requirements",
    table_name="masters",
    limit=5
)

for result in results:
    print(f"Score: {result.score:.3f}")
    print(f"Source: {result.source_file}")
    print(f"Summary: {result.embedding_summary}")
    print("-" * 50)
```

#### Using the Flask HTTP API

Start the server:

```bash
# Start with default settings
python lance_db_server.py

# Or with custom settings
python lance_db_server.py --host 0.0.0.0 --port 8080 --db-path ./custom_db
```

Use the API:

```bash
# Health check
curl http://localhost:5000/health

# Get database statistics
curl http://localhost:5000/stats

# List all tables
curl http://localhost:5000/tables

# Ingest parquet file
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "parquet_path": "data/processed_parquets/masters_data.parquet",
    "table_name": "masters",
    "overwrite": true,
    "batch_size": 100
  }'

# Search documents
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "finance program admission",
    "table_name": "masters",
    "limit": 5,
    "score_threshold": 0.0
  }'
```

## API Endpoints

### Database Management

- `GET /health` - Health check
- `GET /stats` - Database statistics
- `GET /tables` - List all tables
- `GET /tables/{table_name}` - Get table information
- `DELETE /tables/{table_name}` - Delete table

### Data Operations

- `POST /ingest` - Ingest parquet file into database
- `POST /search` - Search for similar documents

### Detailed API Documentation

#### POST /ingest

Ingest data from a parquet file into LanceDB.

**Request Body:**
```json
{
  "parquet_path": "path/to/file.parquet",
  "table_name": "optional_table_name",
  "overwrite": false,
  "batch_size": 100
}
```

**Response:**
```json
{
  "message": "Ingestion completed successfully",
  "stats": {
    "table_name": "masters",
    "parquet_file": "data/processed_parquets/masters_data.parquet",
    "total_records": 334,
    "processed_records": 334,
    "error_records": 0,
    "success_rate": 100.0,
    "ingestion_time": "2024-01-15T10:30:00"
  }
}
```

#### POST /search

Search for similar documents using vector similarity.

**Request Body:**
```json
{
  "query": "search text",
  "table_name": "table_to_search",
  "limit": 10,
  "score_threshold": 0.0
}
```

**Response:**
```json
{
  "query": "finance program admission",
  "table_name": "masters",
  "total_results": 3,
  "results": [
    {
      "id": "uuid-string",
      "score": 0.92,
      "source_file": "/path/to/source.json",
      "text": "Document text preview...",
      "word_count": 456,
      "embedding_summary": "Program summary..."
    }
  ]
}
```

## Configuration Options

### Environment Variables

- `LANCEDB_PATH`: Database storage path (default: `./lance_db`)
- `EMBEDDING_DIMENSION`: Vector dimension (default: `384`)
- `FLASK_HOST`: Flask server host (default: `0.0.0.0`)
- `FLASK_PORT`: Flask server port (default: `5000`)
- `DEFAULT_BATCH_SIZE`: Processing batch size (default: `100`)

### Command Line Arguments

```bash
python lance_db_server.py --help

Options:
  --host TEXT              Host to bind to (default: 0.0.0.0)
  --port INTEGER           Port to bind to (default: 5000)
  --debug                  Run in debug mode
  --db-path TEXT           Path to LanceDB database
  --embedding-dim INTEGER  Embedding dimension (default: 384)
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python test_lance_server.py

# Test only the LanceDBManager class
python -c "from test_lance_server import test_lance_db_manager; test_lance_db_manager()"

# Test only the Flask API (requires server running)
python -c "from test_lance_server import test_flask_api; test_flask_api()"
```

## Data Schema

The system expects parquet files with the PSL RAG schema:

```python
{
    "source_file": str,        # Path to original JSON file
    "text": str,              # Markdown-formatted content
    "word_count": int,        # Number of words in text
    "embedding_summary": str   # AI-generated summary for embeddings
}
```

## Performance Considerations

### Indexing

- Vector indexes are automatically created for tables with >50 documents
- Index creation improves search performance but increases ingestion time
- Customize indexing with `num_sub_vectors` parameter

### Batch Processing

- Use appropriate batch sizes (50-200) for optimal performance
- Larger batches use more memory but may be faster
- Monitor memory usage during ingestion

### Embedding Generation

- Current implementation uses mock embeddings for demonstration
- Replace `generate_mock_embedding()` with actual embedding models
- Consider using sentence-transformers, OpenAI, or other embedding services

## Production Deployment

### Using Gunicorn

```bash
# Install gunicorn
pip install gunicorn

# Run with gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 "lance_db_server:LanceDBFlaskServer().app"
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements_lance.txt .
RUN pip install -r requirements_lance.txt

COPY . .
EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "lance_db_server:LanceDBFlaskServer().app"]
```

### Environment Variables for Production

```bash
export LANCEDB_PATH="/data/lance_db"
export FLASK_HOST="0.0.0.0"
export FLASK_PORT="5000"
export LOG_LEVEL="INFO"
```

## Integration with PSL RAG Project

### Ingesting PSL Data

```python
# Ingest all PSL parquet files
parquet_files = [
    "data/processed_parquets/masters_data.parquet",
    "data/processed_parquets/licences_data.parquet",
    "data/processed_parquets/doctorats_data.parquet",
    "data/processed_parquets/diplomes_etablissements_composantes_data.parquet"
]

manager = LanceDBManager()

for parquet_file in parquet_files:
    table_name = Path(parquet_file).stem.replace('_data', '')
    stats = manager.populate_from_parquet(
        parquet_path=parquet_file,
        table_name=table_name,
        overwrite=True
    )
    print(f"Ingested {stats['processed_records']} documents into {table_name}")
```

### Querying PSL Data

```python
# Search across different program types
queries = [
    "finance and business programs",
    "computer science artificial intelligence",
    "doctoral research opportunities",
    "admission requirements international students"
]

for query in queries:
    print(f"\nSearching for: {query}")
    for table in ["masters", "licences", "doctorats"]:
        try:
            results = manager.search_similar(query, table, limit=2)
            if results:
                print(f"  {table}: {len(results)} results")
                for result in results[:1]:
                    print(f"    - {result.embedding_summary[:100]}...")
        except:
            pass
```

## Troubleshooting

### Common Issues

1. **Import Error**: Ensure all dependencies are installed
   ```bash
   pip install -r requirements_lance.txt
   ```

2. **Parquet File Not Found**: Check file paths are absolute or relative to working directory

3. **Table Already Exists**: Use `overwrite=True` parameter or different table name

4. **Memory Issues**: Reduce batch_size for large parquet files

5. **Search Returns No Results**: Check table exists and has data; try lower score_threshold

### Debugging

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or set environment variable
export LOG_LEVEL=DEBUG
```

Check database status:

```bash
curl http://localhost:5000/stats
curl http://localhost:5000/tables
```

## Contributing

1. Follow PSL RAG project coding standards
2. Add tests for new functionality
3. Update documentation for API changes
4. Consider performance implications for large datasets

## License

Part of the PSL RAG project for Paris Sciences & Lettres university.