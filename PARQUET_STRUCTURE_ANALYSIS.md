# PSL RAG Parquet File Structure Analysis

## Overview

The `mattia-gemini.py` script processes parquet files from the `data/processed_parquets/` directory and generates enhanced versions with AI summaries. This document describes the exact structure of both the original files and the result files.

## File Schema

All parquet files (both original and result) follow the same schema:

### PyArrow Schema
```
source_file     -> string (nullable: True)
text           -> string (nullable: True)  
word_count     -> int64 (nullable: True)
embedding_summary -> string (nullable: True)
```

### Column Specifications

| Column | Type | Description | Typical Range |
|--------|------|-------------|---------------|
| `source_file` | string | Absolute path to original JSON file | 70-320 chars |
| `text` | string | Markdown-formatted content from JSON | 1,700-21,400 chars |
| `word_count` | int64 | Number of words in text content | 278-3,342 words |
| `embedding_summary` | string | AI-generated summary for embeddings | 0-950 chars |

## Files in processed_parquets Directory

| File | Documents | Size | Description |
|------|-----------|------|-------------|
| `masters_data.parquet` | 334 | 5.2 MB | Master's degree programs |
| `licences_data.parquet` | 68 | 1.2 MB | Bachelor's degree programs |
| `doctorats_data.parquet` | 25 | 0.16 MB | Doctoral programs |
| `diplomes_etablissements_composantes_data.parquet` | 49 | 0.68 MB | Institution-specific diplomas |

## Script Processing Behavior

### Input Files (Original)
- **Location**: `data/processed_parquets/`
- **Status**: All files already contain `embedding_summary` values
- **Content**: Pre-processed with summaries from previous runs

### Output Files (Result)
- **Naming**: `temp_summarization_{original_filename}`  
- **Example**: `temp_summarization_masters_data.parquet`
- **Location**: Same directory during processing
- **Purpose**: Incremental progress saving

### Processing States

The `embedding_summary` field can have three states:

1. **Empty** (`""`) - Not yet processed
2. **Valid** - Contains 2-4 sentence AI summary
3. **Error** - Starts with `[SUMMARY_ERROR:` or `[EMPTY_TEXT]`

## Text Content Structure

The `text` field contains Markdown-formatted content with this structure:

```markdown
# Formation
# Niveau Diplome Id
{level_id}

# Diplome Delivre
{diploma_name}

# Etablissements Operateurs
- {institution_name}

# Detail Formation
## Langue Id
{language_id}

## Nom
{program_name}

## Presentation
{description}

## Objectifs
{objectives}

## Conditions Acces
{admission_requirements}

[... additional sections ...]
```

## Source File Patterns

The `source_file` field contains paths following these patterns:

### Main Program Files
```
*/PROGRAM_NAME/PROGRAM_NAME.json
```
Example: `/Users/mattia/Desktop/PSL/DATA/MASTERS/MASTER FINANCE/MASTER FINANCE.json`

### Academic Track Files (Parcours)
```
*/PROGRAM_NAME/PARCOURS/PROGRAM_NAME_Parcours_TRACK_NAME.json
```
Example: `/Users/mattia/Desktop/PSL/DATA/MASTERS/MASTER FINANCE/PARCOURS/MASTER FINANCE_Parcours Audit and financial advisory (M2).json`

## Python Type Definitions

A complete Python schema definition is provided in `parquet_schemas.py`:

```python
@dataclass
class PSLDocumentRecord:
    source_file: str        # Path to original JSON
    text: str              # Markdown content  
    word_count: int        # Word count
    embedding_summary: str # AI summary
```

## Current State Analysis

Based on inspection of existing files:

- ✅ **All files already have summaries**: Every document in all 4 parquet files contains valid `embedding_summary` values
- 📊 **Total documents**: 476 across all files (334 + 68 + 25 + 49)
- 💾 **Total size**: ~7.3 MB
- 🔄 **Script status**: Currently set to test mode, would re-process existing summaries

## Key Insights for Development

1. **File Consistency**: All files follow identical schema
2. **Content Format**: Text is consistently formatted as Markdown with structured headers
3. **Bilingual Content**: Programs contain both French and English information
4. **Hierarchical Structure**: Both main programs and academic tracks (parcours) are included
5. **Processing Complete**: All current files already have AI summaries generated

## Usage Recommendations

For working with these files:
- Use `parquet_schemas.py` for type definitions and validation
- The script can process files without existing summaries (empty string values)
- Progress is saved after every document to enable resumption
- Files are ready for embedding generation using the `embedding_summary` column