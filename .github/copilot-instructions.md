# PSL RAG Project - GitHub Copilot Instructions

## Project Overview

This project builds a student-facing Q&A chatbot using Retrieval-Augmented Generation (RAG) for Paris Sciences & Lettres (PSL) university. The system aggregates and structures official university information including formations, admissions, calendars, and housing data, with improved retrieval quality through synthetic knowledge representations.

## Poetry Dependency Management

This project uses [Poetry](https://python-poetry.org/) for Python dependency management and environment setup. To install dependencies or run scripts, use:

- `poetry install` to install all dependencies
- `poetry run python <script.py>` to run scripts in the Poetry environment

Refer to `pyproject.toml` for package configuration.

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

## Database Manager & API Server Functionality

### LanceDB Manager (`modules/lance_db_manager.py`)

This module provides a comprehensive interface for managing the LanceDB vector database, including:

- Database connection and initialization
- Parquet file ingestion with PSL schema validation
- Table creation, deletion, and info retrieval
- Batch insertion with embedding generation (requires an embedder instance)
- Vector similarity search for document retrieval
- Database statistics and resource cleanup

Usage: Instantiate `LanceDBManager`, configure with a path and embedder, and use its methods for ingestion, search, and table management.

### Flask REST API Server (`modules/flask_server.py`)

This module exposes LanceDB operations via a RESTful HTTP API, including:

- Health check and database stats endpoints
- Table listing, info, and deletion endpoints
- Parquet ingestion endpoint (`POST /ingest`)
- Vector similarity search endpoint (`POST /search`)
- Error handling and JSON responses

Usage: Instantiate `LanceDBFlaskServer` with a `LanceDBManager` and call `.run()` to start the server. See endpoint documentation in the root API response (`GET /`).

**Note:** All database operations and ingestion require a valid embedder instance for embedding generation.

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
- **Database & API**: Place database management and API server code in `modules/`.


### Dependencies & Tools

Expected tech stack (infer from context):
- **Poetry**: For dependency management and environment setup
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
6. **API Usage**: Use the Flask server endpoints for database operations and document retrieval.
7. **Resource Cleanup**: Ensure database manager resources are closed after use.

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

## Benchmark Evaluation System

### Benchmark Overview

The project includes a comprehensive benchmark dataset at **`data/questions-benchmark.txt`** containing 70 real student questions that evaluate the RAG chatbot's ability to provide accurate, relevant responses about PSL programs and procedures.

### Question Categories & Patterns

The benchmark covers six main categories of student inquiries:

#### 1. Program Discovery & Recommendations
- **Pattern**: Students seeking program suggestions based on career goals or interests
- **Examples**: "Quelles formation pouvez-vous me conseiller pour devenir comédien ?", "Je souhaite travailler dans l'industrie pharmaceutique, quelle formation me conseillez-vous ?"
- **Key Challenge**: Requires understanding career pathways and matching them to specific PSL programs

#### 2. Admission Requirements & Prerequisites
- **Pattern**: Questions about BAC specialties, language requirements, and eligibility criteria
- **Examples**: "Quelles sont spécialités recommandées au BAC pour s'inscrire en Science pour un Monde durable ?", "Faut-il maitriser l'anglais pour postuler au master Science cognitive ?"
- **Key Challenge**: Extracting specific admission criteria from program data

#### 3. Program Structure & Curriculum Details
- **Pattern**: Specific questions about course content, schedules, and academic organization
- **Examples**: "Combien d'heures de maths en science pour un monde durable ?", "C'est quels jours les cours du tronc commun humanité numérique de première année ?"
- **Key Challenge**: Accessing detailed curriculum information within program parcours

#### 4. Career Outcomes & Progression
- **Pattern**: Questions about post-graduation opportunities and further studies
- **Examples**: "Je fais quoi après un master Science cognitive ?", "Est-ce que je peux faire un doctorat après un master SCIENCES DU VIVANT ?"
- **Key Challenge**: Connecting program content to career prospects and academic pathways

#### 5. Administrative Procedures
- **Pattern**: Questions about contact information, deadlines, and bureaucratic processes
- **Examples**: "Qui dois-je contacter pour avoir plus d'informations sur le Master Sciences de l'univers et technologies spatiales ?", "Quelle est la date limite du dépôt des dossiers de mon master ?"
- **Key Challenge**: Providing accurate administrative details and contact points

#### 6. International Student Support
- **Pattern**: Questions from international students about procedures and requirements
- **Examples**: "Je suis un étudiant grec, comment puis-je candidater pau CPES ?", "Can you tell me if master's admissions for this year are still open or not for international students?"
- **Key Challenge**: Handling multilingual queries and international-specific procedures

### Benchmark Quality Indicators

#### Language Complexity
- **Multilingual Support**: Questions in both French and English
- **Natural Language Variations**: Informal phrasing, abbreviations (e.g., "Ca mène à quoi")
- **Typos & Colloquialisms**: Realistic student language including spelling errors

#### Content Specificity
- **Named Entities**: Specific programs, professors (e.g., "Thérèse COLLINS"), institutions
- **Technical Terms**: Academic terminology mixing French and English
- **Contextual References**: Questions requiring understanding of French higher education system

### Evaluation Guidelines for AI Agents

When developing or testing the RAG system against this benchmark:

#### Response Quality Metrics
1. **Accuracy**: Factual correctness of information provided
2. **Completeness**: Coverage of all relevant aspects of the question
3. **Relevance**: Appropriate scope and focus of the response
4. **Clarity**: Understandable language appropriate for student audience
5. **Actionability**: Practical guidance with specific next steps when appropriate

#### Common Failure Patterns to Avoid
1. **Program Confusion**: Mixing up similar program names or institutions
2. **Outdated Information**: Providing information that may not be current
3. **Language Mixing**: Inappropriate code-switching between French and English
4. **Overgeneralization**: Providing generic advice when specific information is available
5. **Missing Context**: Failing to consider the broader PSL ecosystem and inter-institutional relationships

#### Testing Strategies
1. **Systematic Coverage**: Ensure all question categories are addressed in testing
2. **Edge Case Handling**: Test with questions containing typos or ambiguous phrasing
3. **Multilingual Consistency**: Verify consistent quality across French and English queries
4. **Cross-Reference Validation**: Check answers against multiple data sources when available
5. **User Journey Mapping**: Test sequences of related questions that students might ask

### Implementation Considerations

#### For Retrieval Systems
- Questions often reference programs by informal names or partial titles
- Students may ask about programs that don't exist or have been renamed
- Cross-institutional queries require understanding PSL member school relationships

#### For Generation Systems
- Responses should maintain appropriate academic tone while being accessible
- Include relevant disclaimers about contacting official sources for definitive information
- Handle uncertainty gracefully when information is not available in the dataset

#### For Evaluation Scripts
- Implement automated metrics for response quality assessment
- Create rubrics for manual evaluation of complex queries
- Track performance across different question categories and languages
- Monitor for bias in responses across different program types or institutions
