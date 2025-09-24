# PSL RAG Project - GitHub Copilot Instructions

## Project Overview

This project builds a student-facing Q&A chatbot using Retrieval-Augmented Generation (RAG) for Paris Sciences & Lettres (PSL) university. The system aggregates and structures official university information including formations, admissions, calendars, and housing data, with improved retrieval quality through synthetic knowledge representations.

## Project Structure

### Core Data Organization

The project contains scraped university data organized by program types:

- **`data/DIPLOMES_ETABLISSEMENTS_COMPOSANTES/`**: Diploma and institutional component data
- **`data/DOCTORATS/`**: Doctoral program information
- **`data/LICENCES/`**: Bachelor's degree programs  
- **`data/MASTERS/`**: Master's degree programs

### Data Structure Pattern

Each program type follows a consistent structure:
```
PROGRAM_TYPE/
├── PROGRAM_NAME/
│   ├── PROGRAM_NAME.json          # Main program information (when available)
│   └── PARCOURS/                  # Academic tracks/pathways (when available)
│       ├── PROGRAM_NAME_Track1.json
│       └── PROGRAM_NAME_Track2.json
```

**Important Notes:**
- Not every program has main program info files
- Not every program has parcours (academic tracks)
- File structure varies by program availability

### Processed Data

**`data/processed_parquets/`**: Contains processed versions of the raw data:
- Original JSON text converted to Markdown format
- Synthetic summaries generated for each chunk
- Used for embeddings in the RAG system
- Files: `diplomes_etablissements_composantes_data.parquet`, `doctorats_data.parquet`, `licences_data.parquet`, `masters_data.parquet`

## Data Schema

### JSON Structure
Each program JSON file contains:
- **FORMATION**: Core program metadata including:
  - Diploma level and type
  - Operating institutions (Dauphine-PSL, ENSAD-PSL, etc.)
  - Responsible faculty/coordinators
  - Languages of instruction
  - Academic disciplines
- **DETAIL_FORMATION**: Detailed program information by language
- **PARCOURS**: Academic tracks and specializations (when applicable)

### Key Identifiers
- Programs are identified by their full names (e.g., "MASTER FINANCE")
- Parcours files include both program and track names in filename
- Institution codes reference PSL member schools

## Development Guidelines

### Working with Data

1. **Data Access**: Always check if both main program files and parcours exist before processing
2. **File Naming**: Preserve exact naming conventions including accents and special characters
3. **Missing Data**: Handle cases where either main program info or parcours may be missing
4. **Encoding**: Ensure proper UTF-8 handling for French text and special characters

### RAG System Development

1. **Chunking Strategy**: Respect the existing processed parquet structure with synthetic summaries
2. **Embeddings**: Use synthetic summaries for improved retrieval quality
3. **Context**: Consider hierarchical relationships between programs and their parcours
4. **Multilingual Support**: Handle both French and English content appropriately

### Code Organization

When creating new code:
- **Data Processing**: Create scripts in a `src/` or `scripts/` directory
- **RAG Components**: Organize embedding, retrieval, and generation logic separately
- **Configuration**: Use environment variables for model settings and API keys
- **Documentation**: Document data transformations and processing steps

### Dependencies & Tools

Expected tech stack (infer from context):
- **Python**: Primary language (evidenced by .gitignore)
- **Pandas/PyArrow**: For parquet file processing
- **Transformers/LangChain**: For RAG implementation
- **Vector Databases**: For embedding storage and retrieval
- **FastAPI/Flask**: For chatbot API endpoints

### Best Practices

1. **Data Integrity**: Preserve original French program names and academic terminology
2. **Performance**: Leverage processed parquet files for efficient data access
3. **Scalability**: Design for potential addition of new PSL schools and programs
4. **Error Handling**: Account for inconsistent data availability across programs
5. **Testing**: Create test cases covering various program types and data completeness scenarios

### PSL Context

Paris Sciences & Lettres is a collegiate university comprising:
- Dauphine - PSL (business and economics)
- École normale supérieure (humanities and sciences)
- ESPCI Paris (physics and chemistry)
- Mines Paris - PSL (engineering)
- And other member institutions

When working on the chatbot, ensure responses reflect PSL's multidisciplinary nature and institutional diversity.

### File Operations

- Use proper path handling for nested directory structures
- Handle long filenames with spaces and special characters
- Implement robust file existence checks before processing
- Consider case sensitivity in program name matching
