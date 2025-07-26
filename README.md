# Adobe Hackathon Round 1B: Persona-Driven Document Intelligence

## Overview

This solution implements a persona-driven document intelligence system that extracts and ranks the most relevant sections from a collection of documents based on a specific persona and their job-to-be-done.

## Features

- **Generic Solution**: Works across diverse domains (academic, business, technical, medical)
- **Persona-Aware**: Adapts to different professional roles and expertise levels
- **Task-Oriented**: Prioritizes content based on specific job requirements
- **Efficient Processing**: CPU-only execution under 1GB model size constraint
- **Robust Extraction**: Handles various PDF formats and document structures

## Directory Structure

```
Round1B/
├── app/
│   └── main.py                 # Main application logic
├── input/                      # Input PDFs and configuration
├── output/                     # Generated JSON outputs
├── Dockerfile                  # Docker configuration
├── requirements.txt            # Python dependencies
├── approach_explanation.md     # Technical methodology
└── README.md                   # This file
```

## Input Format

### Document Collection
Place 3-10 related PDF files in the `input/` directory.

### Configuration (Optional)
Create a `config.json` file in the `input/` directory:

```json
{
  "persona": "PhD Researcher in Computational Biology",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
}
```

If no configuration file is provided, the system uses default values suitable for academic research.

## Output Format

The system generates a `challenge1b_output.json` file with the following structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a comprehensive literature review...",
    "processing_timestamp": "2025-01-20T10:30:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "section_title": "Methodology",
      "importance_rank": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc1.pdf",
      "page_number": 3,
      "refined_text": "Summary of key findings...",
      "subsection_id": "Methodology_sub_1"
    }
  ]
}
```

## Technical Architecture

### Core Components

1. **PersonaDrivenExtractor**: Main class handling document processing and relevance scoring
2. **PDF Text Extraction**: Uses PyMuPDF for robust text extraction with page-level granularity
3. **Heading Detection**: Rule-based system for identifying document structure
4. **Relevance Scoring**: TF-IDF similarity combined with domain-aware keyword boosting
5. **Subsection Analysis**: Granular content extraction with text refinement

### Models and Libraries

- **spaCy (en_core_web_sm)**: Lightweight NLP model for text processing
- **DistilBERT**: Compact transformer for semantic understanding
- **Scikit-learn**: TF-IDF vectorization and similarity calculations
- **PyMuPDF**: PDF text extraction and structure analysis

## Building and Running

### Docker Build
```bash
docker build --platform linux/amd64 -t round1b-extractor.
```

### Docker Run
```bash
docker run --rm -v %cd%/input:/app/input -v %cd%/output:/app/output round1b-extractor
```
### Notes:
Ensure you are inside the Round1B directory when running these commands.

### Local Development
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app/main.py
```

## Performance Characteristics

- **Model Size**: ~800MB (under 1GB constraint)
- **Processing Time**: Typically 40-45 seconds for 5 documents
- **CPU Optimization**: Multi-threaded text processing where applicable

## Algorithm Details

### Relevance Scoring
1. **Content Similarity**: TF-IDF cosine similarity between section content and persona+job description
2. **Keyword Boosting**: Domain-specific keyword matching with importance weights
3. **Position Weighting**: Earlier sections receive slight priority boost
4. **Adaptive Thresholding**: Dynamic relevance cutoff based on content quality

### Section Selection
- Top 25 most relevant sections selected for output (if 25 are there based on similarity)
- Minimum relevance threshold of 0.1 to ensure quality

### Subsection Extraction
- Paragraph-level granularity for detailed analysis
- Text refinement using first 2 sentences or 200 characters
- Full text preservation for comprehensive review

## Supported Use Cases

### Academic Research
- Literature reviews and methodology analysis
- Dataset and benchmark comparisons
- Research gap identification

### Business Analysis
- Financial report analysis
- Market positioning studies
- Revenue trend analysis

### Educational Content
- Exam preparation and key concept extraction
- Curriculum planning and content organization
- Study guide generation

## Error Handling

- Graceful fallbacks for missing or corrupted PDF files
- Model loading failure recovery with basic text processing
- Empty document collection handling
- Invalid configuration file management

## Limitations

- English language focus (can be extended for multilingual support)
- Dependent on document structure quality for optimal heading detection
- Performance scales linearly with document collection size
- Requires minimum content length for meaningful section analysis
