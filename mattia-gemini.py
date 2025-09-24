import pandas as pd
import numpy as np
from pathlib import Path
import time
import json
from google import genai
from google.genai import types
from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
import re
from os import getenv

# Configuration
load_dotenv()
API_KEY = getenv("PROCESSING_GEMINI_API_KEY")
MODEL = getenv("PROCESSING_GEMINI_MODEL_NAME")

class DocumentSummarizer:
    """Generate embedding-friendly summaries for document chunks using Gemini API."""
    
    def __init__(self, api_key: str):
        """Initialize the Gemini API client."""
        print("Initializing Gemini API client...")
        try:
            self.genai_client = genai.Client(api_key=api_key)
            print("✓ Gemini API client initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing Gemini API client: {e}")
            raise
    
    def generate_summary(self, text_chunk: str, max_retries: int = 3) -> str:
        """
        Generate a concise summary suitable for embedding generation.
        
        Args:
            text_chunk: The original text to summarize
            max_retries: Number of retry attempts for API calls
            
        Returns:
            Summary text or error message
        """
        
        # Create the summarization prompt
        prompt = f"""Please create a concise, informative summary of the following text that will be used for semantic search and embedding generation.

The summary should:
1. Capture the key information and main topics
2. Be 2-4 sentences long
3. Use clear, natural language (avoid tables/lists)
4. Focus on the most important content for search purposes
5. Include relevant keywords and concepts

Original text:
{text_chunk}

Please provide only the summary, without any additional formatting or explanations."""

        for attempt in range(max_retries):
            try:
                contents = [
                    types.Content(
                        role="user",
                        parts=[types.Part.from_text(text=prompt)],
                    ),
                ]
                
                config = types.GenerateContentConfig(
                    response_mime_type="text/plain",
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                )
                
                response = self.genai_client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=config,
                )
                
                summary = response.text.strip()
                
                # Basic validation of the summary
                if len(summary) < 10:
                    print(f"Warning: Very short summary generated (length: {len(summary)})")
                
                return summary
                
            except Exception as e:
                print(f"API call attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2  # Exponential backoff
                    print(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    error_msg = f"Failed to generate summary after {max_retries} attempts: {str(e)}"
                    print(f"❌ {error_msg}")
                    return f"[SUMMARY_ERROR: {error_msg}]"
        
        return "[SUMMARY_ERROR: Unknown error]"


class ProgressManager:
    """Handles saving and loading progress for document processing."""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        
    def get_progress_path(self, file_name: str) -> Path:
        """Get the progress file path for a given parquet file."""
        clean_name = file_name.replace('.parquet', '')
        return self.base_path / f"progress_summarization_{clean_name}.json"
    
    def get_temp_df_path(self, file_name: str, output_path: Path) -> Path:
        """Get the temporary dataframe path."""
        clean_name = file_name.replace('.parquet', '')
        return output_path.parent / f"temp_summarization_{clean_name}.parquet"
    
    def save_progress(self, file_name: str, df: pd.DataFrame, current_index: int, 
                     output_path: Path, start_time: float) -> bool:
        """Save current progress to disk."""
        try:
            # Save temporary dataframe
            temp_df_path = self.get_temp_df_path(file_name, output_path)
            temp_df_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(temp_df_path, index=False)
            
            # Save progress metadata
            progress_data = {
                'file_name': file_name,
                'current_index': current_index,
                'total_documents': len(df),
                'temp_df_path': str(temp_df_path),
                'output_path': str(output_path),
                'start_time': start_time,
                'last_updated': time.strftime('%Y-%m-%d %H:%M:%S'),
                'elapsed_time': time.time() - start_time
            }
            
            progress_path = self.get_progress_path(file_name)
            with open(progress_path, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to save progress: {e}")
            return False
    
    def load_progress(self, file_name: str) -> Optional[Dict]:
        """Load existing progress for a file."""
        progress_path = self.get_progress_path(file_name)
        
        if not progress_path.exists():
            return None
        
        try:
            with open(progress_path, 'r', encoding='utf-8') as f:
                progress_data = json.load(f)
            
            # Verify temp file exists
            temp_df_path = Path(progress_data['temp_df_path'])
            if not temp_df_path.exists():
                print(f"⚠️ Temp file missing for {file_name}, starting fresh")
                self.clean_progress(file_name)
                return None
            
            return progress_data
            
        except Exception as e:
            print(f"⚠️ Failed to load progress for {file_name}: {e}")
            return None
    
    def clean_progress(self, file_name: str) -> None:
        """Remove all progress files for a completed file."""
        try:
            # Remove progress JSON
            progress_path = self.get_progress_path(file_name)
            if progress_path.exists():
                progress_path.unlink()
            
            # Remove temp dataframe - we need to check all possible locations
            # since we don't know the exact output path structure
            temp_pattern = f"temp_summarization_{file_name.replace('.parquet', '')}.parquet"
            for temp_file in self.base_path.rglob(temp_pattern):
                if temp_file.exists():
                    temp_file.unlink()
                    
        except Exception as e:
            print(f"⚠️ Failed to clean progress files for {file_name}: {e}")


def load_and_process_file(input_path: Path, output_path: Path, summarizer: DocumentSummarizer, 
                         progress_manager: ProgressManager, delay_per_doc: float = 0.5) -> bool:
    """Load and process a single parquet file with robust progress saving."""
    
    file_name = input_path.name
    print(f"\n{'='*80}")
    print(f"PROCESSING: {file_name}")
    print(f"{'='*80}")
    
    # Check for existing progress
    progress_data = progress_manager.load_progress(file_name)
    
    if progress_data:
        # Resume from saved progress
        try:
            temp_df_path = Path(progress_data['temp_df_path'])
            df = pd.read_parquet(temp_df_path)
            start_index = progress_data['current_index']
            start_time = progress_data['start_time']
            
            print(f"✓ RESUMING: Loaded {len(df)} documents")
            print(f"✓ Progress: {start_index}/{len(df)} documents completed")
            print(f"✓ Elapsed: {progress_data['elapsed_time']:.1f} seconds")
            print(f"✓ Resuming from document {start_index + 1}")
            
        except Exception as e:
            print(f"❌ Failed to load progress: {e}")
            print("Starting fresh...")
            df = pd.read_parquet(input_path)
            start_index = 0
            start_time = time.time()
            
    else:
        # Start fresh
        print("Starting fresh processing...")
        try:
            df = pd.read_parquet(input_path)
            start_index = 0
            start_time = time.time()
            print(f"✓ Loaded {len(df)} documents from {file_name}")
        except Exception as e:
            print(f"❌ Failed to load {file_name}: {e}")
            return False
    
    # Ensure required columns exist
    if 'text' not in df.columns:
        print(f"❌ Missing 'text' column in {file_name}")
        return False
    
    # Add summary column if it doesn't exist
    if 'embedding_summary' not in df.columns:
        df['embedding_summary'] = ""
    
    # Process documents one by one
    total_docs = len(df)
    successful = 0
    failed = 0
    skipped = 0
    
    print(f"\nProcessing documents {start_index + 1} to {total_docs}...")
    print("-" * 80)
    
    for i in range(start_index, total_docs):
        # Show progress
        progress_pct = ((i + 1) / total_docs) * 100
        elapsed = time.time() - start_time
        
        print(f"[{i+1:4d}/{total_docs}] ({progress_pct:5.1f}%) ", end="")
        
        # Check if already has valid summary
        current_summary = df.iloc[i]['embedding_summary']
        if (pd.notna(current_summary) and 
            current_summary and 
            not current_summary.startswith('[') and 
            len(current_summary) > 10):
            skipped += 1
            print("SKIP - Already completed")
            continue
        
        # Get document text
        doc_text = df.iloc[i]['text']
        
        # Skip empty documents
        if pd.isna(doc_text) or len(str(doc_text).strip()) < 10:
            df.iloc[i, df.columns.get_loc('embedding_summary')] = "[EMPTY_TEXT]"
            failed += 1
            print("SKIP - Empty text")
        else:
            # Generate summary
            print("Summarizing... ", end="")
            summary = summarizer.generate_summary(str(doc_text))
            
            # Store result
            df.iloc[i, df.columns.get_loc('embedding_summary')] = summary
            
            if summary.startswith("[SUMMARY_ERROR"):
                failed += 1
                print("FAILED")
            else:
                successful += 1
                print("OK")
        
        # Save progress after EVERY document
        save_success = progress_manager.save_progress(
            file_name, df, i + 1, output_path, start_time
        )
        
        # Add delay to avoid rate limiting
        time.sleep(delay_per_doc)
        
        # Print status every 25 documents
        if (i + 1) % 25 == 0:
            rate = (i + 1 - start_index) / elapsed if elapsed > 0 else 0
            eta_seconds = (total_docs - i - 1) / rate if rate > 0 else 0
            eta_minutes = eta_seconds / 60
            
            print(f"\n  STATUS: {successful} success, {failed} failed, {skipped} skipped")
            print(f"  RATE: {rate:.1f} docs/sec, ETA: {eta_minutes:.1f} minutes")
    
    # Final statistics
    print(f"\n{'='*80}")
    print(f"PROCESSING COMPLETE: {file_name}")
    print(f"{'='*80}")
    print(f"Total documents: {total_docs}")
    print(f"Successful summaries: {successful}")
    print(f"Failed summaries: {failed}")
    print(f"Skipped (already done): {skipped}")
    print(f"Success rate: {(successful/(total_docs-skipped))*100:.1f}%" if (total_docs-skipped) > 0 else "N/A")
    print(f"Total time: {(time.time()-start_time)/60:.1f} minutes")
    
    # Save final result
    try:
        print(f"\nSaving final result to {output_path}...")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)
        print(f"✓ Successfully saved {len(df)} documents with summaries")
        
        # Show sample summaries
        valid_summaries = df[~df['embedding_summary'].str.startswith('[')]['embedding_summary']
        if len(valid_summaries) > 0:
            print(f"\nSample summaries ({len(valid_summaries)} total valid):")
            print("-" * 60)
            for i, summary in enumerate(valid_summaries.head(2)):
                print(f"{i+1}. {summary[:150]}{'...' if len(summary) > 150 else ''}")
        
        # Clean up progress files
        progress_manager.clean_progress(file_name)
        print("✓ Cleaned up progress files")
        
        return True
        
    except Exception as e:
        print(f"❌ Error saving final file: {e}")
        return False


def process_all_parquets(base_path: str, output_dir: str, api_key: str):
    """Process all parquet files with comprehensive progress tracking."""
    
    print("=" * 100)
    print("PSL DOCUMENT SUMMARIZATION WITH FREQUENT PROGRESS SAVING")
    print("=" * 100)
    print(f"Input directory: {base_path}")
    print(f"Output directory: {output_dir}")
    print("Progress saved after EVERY document processed")
    print("=" * 100)
    
    # Initialize components
    try:
        summarizer = DocumentSummarizer(api_key)
        progress_manager = ProgressManager(base_path)
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        return
    
    # Define files to process
    parquet_files = [
        # "masters_data.parquet",
        # "licences_data.parquet", 
        # "doctorats_data.parquet",
        "diplomes_etablissements_composantes_data.parquet"
    ]
    
    base_path = Path(base_path)
    output_path = Path(output_dir)
    
    # Process each file
    results = {}
    total_start_time = time.time()
    
    for i, parquet_file in enumerate(parquet_files):
        print(f"\n{'='*100}")
        print(f"FILE {i+1}/{len(parquet_files)}: {parquet_file}")
        print(f"{'='*100}")
        
        input_file_path = base_path / parquet_file
        output_file_path = output_path / parquet_file
        
        if not input_file_path.exists():
            print(f"⚠️ Input file not found: {input_file_path}")
            results[parquet_file] = "FILE_NOT_FOUND"
            continue
        
        # Process the file
        file_start_time = time.time()
        success = load_and_process_file(
            input_file_path,
            output_file_path,
            summarizer,
            progress_manager,
            delay_per_doc=0.5  # Half second delay per document
        )
        file_time = time.time() - file_start_time
        
        if success:
            results[parquet_file] = f"SUCCESS ({file_time/60:.1f} min)"
            print(f"✅ {parquet_file} completed successfully in {file_time/60:.1f} minutes")
        else:
            results[parquet_file] = f"FAILED ({file_time/60:.1f} min)"
            print(f"❌ {parquet_file} failed after {file_time/60:.1f} minutes")
    
    # Final summary
    total_time = time.time() - total_start_time
    print(f"\n{'='*100}")
    print("FINAL PROCESSING SUMMARY")
    print(f"{'='*100}")
    print(f"Total processing time: {total_time/60:.1f} minutes ({total_time/3600:.1f} hours)")
    print("\nFile processing results:")
    
    for file, result in results.items():
        status_icon = "✅" if "SUCCESS" in result else "❌" if "FAILED" in result else "⚠️"
        print(f"  {status_icon} {file}: {result}")
    
    successful_files = sum(1 for result in results.values() if "SUCCESS" in result)
    print(f"\nSuccessfully processed {successful_files}/{len(parquet_files)} files")
    
    if successful_files > 0:
        print(f"\nProcessed files saved to: {output_path}")
        print("These files now contain an 'embedding_summary' column suitable for semantic search.")


def test_single_document(base_path: str, api_key: str, file_name: str = "masters_data.parquet", doc_index: int = 0):
    """Test summarization on a single document."""
    
    print("=" * 80)
    print("TESTING SINGLE DOCUMENT SUMMARIZATION")
    print("=" * 80)
    
    # Load test file
    test_file = Path(base_path) / file_name
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    try:
        df = pd.read_parquet(test_file)
    except Exception as e:
        print(f"❌ Failed to load {file_name}: {e}")
        return
        
    if len(df) <= doc_index:
        print(f"❌ Document index {doc_index} out of range (file has {len(df)} documents)")
        return
    
    # Get test document
    test_doc = df.iloc[doc_index]
    original_text = test_doc['text']
    
    print(f"Testing document {doc_index} from {file_name}")
    print(f"Original text length: {len(original_text)} characters")
    print(f"Word count: {test_doc.get('word_count', 'N/A')}")
    print("-" * 80)
    print("ORIGINAL TEXT (first 500 chars):")
    print(original_text[:500] + "..." if len(original_text) > 500 else original_text)
    print("-" * 80)
    
    # Test summarization
    try:
        summarizer = DocumentSummarizer(api_key)
        print("Generating summary...")
        
        start_time = time.time()
        summary = summarizer.generate_summary(original_text)
        generation_time = time.time() - start_time
        
        print(f"\nSUMMARY (generated in {generation_time:.2f}s):")
        print("-" * 80)
        print(summary)
        print("-" * 80)
        print(f"Summary length: {len(summary)} characters")
        print(f"Compression ratio: {len(summary)/len(original_text):.3f}")
        
        if summary.startswith("[SUMMARY_ERROR"):
            print("❌ Summary generation failed")
        else:
            print("✅ Summary generation successful")
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")


if __name__ == "__main__":
    # Configuration
    base_path = "./data/processed_parquets"
    output_dir = "./data/test_processed_parquets"
    api_key = API_KEY
    
    # Test mode - uncomment to test with a single document first
    test_single_document(base_path, api_key, "masters_data.parquet", 0)
    
    # Full processing - uncomment to process all files
    # process_all_parquets(base_path, output_dir, api_key)