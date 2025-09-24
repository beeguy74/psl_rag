# LanceDB Server - Quick Start Guide

## Overview

This LanceDB server provides a complete solution for creating and managing vector databases with Flask HTTP API support, specifically designed for the PSL RAG project. It enables seamless ingestion of parquet files and vector-based similarity search.

## Key Files Created

1. **`lance_db_server.py`** - Main server implementation with LanceDBManager class and Flask API
2. **`test_lance_server.py`** - Comprehensive testing suite for both class and API usage
3. **`demo_lance_server.py`** - Interactive demo showing all features without dependencies
4. **`README_LanceDB.md`** - Complete documentation with examples
5. **`requirements_lance.txt`** - Required Python packages
6. **`.env.example`** - Configuration template

## Quick Installation & Usage

### 1. Install Dependencies
```bash
pip install lancedb flask pandas pyarrow numpy python-dotenv requests
# or
pip install -r requirements_lance.txt
```

### 2. Basic Usage as Python Class
```python
from lance_db_server import LanceDBManager

# Initialize
manager = LanceDBManager(db_path="./my_vector_db")

# Ingest parquet data
stats = manager.populate_from_parquet(
    parquet_path="data/processed_parquets/masters_data.parquet",
    table_name="masters"
)

# Search similar documents
results = manager.search_similar(
    query_text="finance program requirements",
    table_name="masters",
    limit=5
)
```

### 3. Start HTTP Server
```bash
# Default settings (localhost:5000)
python3 lance_db_server.py

# Custom settings
python3 lance_db_server.py --host 0.0.0.0 --port 8080 --db-path ./custom_db
```

### 4. Use HTTP API
```bash
# Health check
curl http://localhost:5000/health

# Ingest data
curl -X POST http://localhost:5000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "parquet_path": "data/processed_parquets/masters_data.parquet",
    "table_name": "masters",
    "overwrite": true
  }'

# Search documents
curl -X POST http://localhost:5000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "finance program admission",
    "table_name": "masters", 
    "limit": 5
  }'
```

## Key Features

### LanceDBManager Class
- **Database Management**: Create, connect, manage LanceDB instances
- **Parquet Ingestion**: Batch processing of PSL document schema
- **Vector Search**: Similarity search with configurable parameters
- **Schema Validation**: Built-in validation for PSL RAG data
- **Performance Optimization**: Automatic indexing and batch processing

### Flask HTTP API  
- **RESTful Endpoints**: Complete CRUD operations via HTTP
- **Error Handling**: Comprehensive error responses and logging
- **JSON Interface**: Clean JSON request/response format
- **Async Support**: Background processing capabilities

### PSL RAG Integration
- **Schema Compatibility**: Designed for PSL parquet file structure
- **Multi-table Support**: Handle different program types (Masters, Licences, etc.)
- **French Content**: Proper handling of French academic terminology
- **Batch Ingestion**: Efficient processing of large datasets

## Testing

```bash
# Run demo (no dependencies required)
python3 demo_lance_server.py

# Run full test suite (requires LanceDB installed)
python3 test_lance_server.py

# Test API (requires running server)
python3 -c "from test_lance_server import test_flask_api; test_flask_api()"
```

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Parquet       │───▶│  LanceDBManager  │───▶│   LanceDB       │
│   Files         │    │                  │    │   Vector DB     │
│  (PSL Data)     │    │ - Validation     │    │                 │
└─────────────────┘    │ - Batch Process  │    └─────────────────┘
                       │ - Embedding Gen  │              │
┌─────────────────┐    │ - Search Logic   │              │
│   Flask HTTP    │◀───┤                  │◀─────────────┘
│   API Server    │    └──────────────────┘
│                 │
│ /health         │
│ /stats          │
│ /ingest         │
│ /search         │
│ /tables         │
└─────────────────┘
```

## Configuration Options

### Environment Variables (.env file)
```bash
LANCEDB_PATH=./lance_db
EMBEDDING_DIMENSION=384
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
DEFAULT_BATCH_SIZE=100
LOG_LEVEL=INFO
```

### Command Line Arguments
```bash
python3 lance_db_server.py \
  --host 0.0.0.0 \
  --port 5000 \
  --db-path ./my_db \
  --embedding-dim 768 \
  --debug
```

## Production Considerations

### Security
- Add authentication/authorization to Flask endpoints
- Validate input parameters and file paths
- Use HTTPS in production environments

### Performance
- Adjust batch sizes based on available memory
- Monitor vector index creation for large datasets  
- Consider distributed deployment for scale

### Monitoring
- Log all operations with appropriate levels
- Monitor database growth and performance metrics
- Set up health check endpoints for load balancers

## Integration Examples

### PSL RAG Workflow
```python
# Complete PSL data ingestion
psl_files = [
    "data/processed_parquets/masters_data.parquet",
    "data/processed_parquets/licences_data.parquet", 
    "data/processed_parquets/doctorats_data.parquet"
]

manager = LanceDBManager(db_path="./psl_vector_db")

for parquet_file in psl_files:
    table_name = Path(parquet_file).stem.replace('_data', '')
    stats = manager.populate_from_parquet(parquet_file, table_name)
    print(f"✅ {table_name}: {stats['processed_records']} documents")

# Multi-table search
def search_all_programs(query, limit=10):
    all_results = []
    for table in ["masters", "licences", "doctorats"]:
        try:
            results = manager.search_similar(query, table, limit)
            for result in results:
                result.table_source = table
                all_results.append(result)
        except:
            continue
    return sorted(all_results, key=lambda x: x.score, reverse=True)[:limit]
```

### Microservice Deployment
```python
# Dedicated microservice
from flask import Flask
from lance_db_server import LanceDBFlaskServer, LanceDBManager

app = Flask(__name__)
manager = LanceDBManager(db_path=os.getenv('VECTOR_DB_PATH'))
server = LanceDBFlaskServer(manager)

# Add custom routes
@app.route('/psl/search/<program_type>')
def psl_search(program_type):
    query = request.args.get('q')
    results = manager.search_similar(query, program_type, limit=10)
    return jsonify([asdict(r) for r in results])

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

## Next Steps

1. **Install Dependencies**: `pip install -r requirements_lance.txt`
2. **Run Demo**: `python3 demo_lance_server.py` 
3. **Test with Real Data**: Use your PSL parquet files
4. **Deploy Server**: Start with `python3 lance_db_server.py`
5. **Integrate**: Use HTTP API or Python class in your application
6. **Scale**: Consider production deployment options

For complete documentation, see `README_LanceDB.md`