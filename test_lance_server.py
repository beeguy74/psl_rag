"""
Example usage and testing script for LanceDB Server.

This script demonstrates how to use the LanceDBManager class and Flask API
for ingesting PSL RAG parquet files and performing vector searches.
"""

import requests
import json
from pathlib import Path
import time
import sys

# Add modules directory to path  
sys.path.insert(0, str(Path(__file__).parent))

from modules.lance_db_manager import LanceDBManager
from modules.flask_server import LanceDBFlaskServer
import pandas as pd


def test_lance_db_manager():
    """Test the LanceDBManager class directly."""
    print("=== Testing LanceDBManager Class ===")
    
    # Initialize manager
    manager = LanceDBManager(db_path="./test_lance_db", embedding_dim=384)
    
    # Get database stats
    stats = manager.get_database_stats()
    print(f"Database stats: {stats}")
    
    # Test with a sample parquet file if available
    parquet_files = list(Path("data/processed_parquets").glob("*.parquet"))
    
    if parquet_files:
        sample_parquet = parquet_files[0]
        print(f"Testing with parquet file: {sample_parquet}")
        
        # Ingest parquet file
        ingestion_stats = manager.populate_from_parquet(
            parquet_path=str(sample_parquet),
            table_name="test_table",
            batch_size=50,
            overwrite=True
        )
        print(f"Ingestion stats: {ingestion_stats}")
        
        # Test search
        search_results = manager.search_similar(
            query_text="master finance program",
            table_name="test_table",
            limit=5
        )
        
        print(f"Search results ({len(search_results)} found):")
        for i, result in enumerate(search_results[:3], 1):
            print(f"  {i}. Score: {result.score:.3f}")
            print(f"     Source: {result.source_file}")
            print(f"     Summary: {result.embedding_summary[:100]}...")
            print()
    else:
        print("No parquet files found in data/processed_parquets/")
    
    # List tables
    tables = manager.list_tables()
    print(f"Available tables: {tables}")


def test_flask_api():
    """Test the Flask API endpoints."""
    print("=== Testing Flask API ===")
    
    base_url = "http://localhost:5000"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health check: {response.status_code} - {response.json()}")
    except requests.exceptions.ConnectionError:
        print("Flask server not running. Please start it first with:")
        print("python lance_server.py")
        return
    
    # Test stats endpoint
    response = requests.get(f"{base_url}/stats")
    print(f"Stats: {response.json()}")
    
    # Test list tables
    response = requests.get(f"{base_url}/tables")
    print(f"Tables: {response.json()}")
    
    # Test ingestion (if parquet files exist)
    parquet_files = list(Path("data/processed_parquets").glob("*.parquet"))
    
    if parquet_files:
        sample_parquet = parquet_files[0]
        
        ingestion_payload = {
            "parquet_path": str(sample_parquet.absolute()),
            "table_name": "api_test_table",
            "overwrite": True,
            "batch_size": 25
        }
        
        print(f"Testing ingestion with payload: {ingestion_payload}")
        response = requests.post(f"{base_url}/ingest", json=ingestion_payload)
        print(f"Ingestion response: {response.status_code}")
        if response.status_code == 200:
            print(f"Ingestion result: {response.json()}")
        else:
            print(f"Ingestion error: {response.text}")
        
        # Test search
        if response.status_code == 200:
            search_payload = {
                "query": "finance program admission requirements",
                "table_name": "api_test_table",
                "limit": 3,
                "score_threshold": 0.0
            }
            
            response = requests.post(f"{base_url}/search", json=search_payload)
            print(f"Search response: {response.status_code}")
            if response.status_code == 200:
                search_results = response.json()
                print(f"Search found {search_results['total_results']} results:")
                for i, result in enumerate(search_results['results'][:2], 1):
                    print(f"  {i}. Score: {result['score']:.3f}")
                    print(f"     Source: {result['source_file']}")
                    print(f"     Text preview: {result['text'][:100]}...")
            else:
                print(f"Search error: {response.text}")


def create_sample_data():
    """Create sample parquet data for testing if none exists."""
    parquet_dir = Path("data/processed_parquets")
    
    if not parquet_dir.exists() or not list(parquet_dir.glob("*.parquet")):
        print("Creating sample data for testing...")
        
        # Create directory if it doesn't exist
        parquet_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample data matching PSL schema
        sample_data = [
            {
                "source_file": "/test/data/MASTERS/MASTER_FINANCE/MASTER_FINANCE.json",
                "text": "# Formation\n# Master Finance\n\nLe master Finance forme des cadres financiers de haut niveau pour les entreprises et institutions financières. Ce programme développe une expertise approfondie en finance quantitative, gestion des risques, et marchés financiers internationaux.",
                "word_count": 45,
                "embedding_summary": "Master's program in Finance training high-level financial executives with expertise in quantitative finance, risk management, and international financial markets."
            },
            {
                "source_file": "/test/data/MASTERS/MASTER_INFORMATIQUE/MASTER_INFORMATIQUE.json", 
                "text": "# Formation\n# Master Informatique\n\nLe master Informatique offre une formation complète en sciences informatiques, couvrant l'intelligence artificielle, le développement logiciel, les systèmes distribués et la cybersécurité. Les étudiants acquièrent les compétences nécessaires pour devenir des experts en technologie.",
                "word_count": 52,
                "embedding_summary": "Computer Science Master's program covering AI, software development, distributed systems and cybersecurity, preparing technology experts."
            },
            {
                "source_file": "/test/data/LICENCES/LICENCE_MATHEMATIQUES/LICENCE_MATHEMATIQUES.json",
                "text": "# Formation\n# Licence Mathématiques\n\nLa licence Mathématiques propose une formation rigoureuse en mathématiques fondamentales et appliquées. Elle prépare les étudiants à poursuivre en master ou à intégrer le monde professionnel dans les domaines de l'enseignement, de la recherche ou de l'industrie.",
                "word_count": 48,
                "embedding_summary": "Mathematics Bachelor's degree with rigorous training in fundamental and applied mathematics, preparing for graduate studies or careers in teaching, research, or industry."
            }
        ]
        
        # Create sample parquet file
        df = pd.DataFrame(sample_data)
        sample_parquet_path = parquet_dir / "sample_test_data.parquet"
        df.to_parquet(sample_parquet_path, index=False)
        
        print(f"Created sample data at: {sample_parquet_path}")
        return sample_parquet_path
    
    return None


def main():
    """Main testing function."""
    print("LanceDB Server Testing Suite")
    print("=" * 50)
    
    # Create sample data if needed
    create_sample_data()
    
    # Test direct manager usage
    test_lance_db_manager()
    
    print("\n" + "=" * 50)
    print("To test the Flask API, run the following in another terminal:")
    print("python lance_server.py")
    print("Then run this script again or call test_flask_api() directly")
    print("=" * 50)
    
    # Optionally test Flask API if server is running
    print("\nAttempting to test Flask API...")
    test_flask_api()


if __name__ == "__main__":
    main()