#!/usr/bin/env python3
"""
Adobe Hackathon Round 1B: Enhanced Persona-Driven Document Intelligence
Improved application for extracting and ranking relevant sections based on persona and job-to-be-done
with advanced duplicate handling, document relevance scoring, and multi-format support
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple, Set, Optional
import fitz  # PyMuPDF
import re
from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import spacy
from transformers import pipeline
from hashlib import md5
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPersonaDrivenExtractor:
    def __init__(self):
        """Initialize with robust error handling and fallback mechanisms"""
        self._init_nlp_models()
        self.duplicate_threshold = 0.85  # Similarity threshold for duplicate detection
        self.min_section_length = 50    # Minimum characters for a valid section
        self.max_sections_per_doc = 15  # Maximum sections to extract per document
        
    def _init_nlp_models(self):
        """Initialize NLP models with fallback mechanisms"""
        # Initialize spaCy
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("spaCy model loaded successfully")
        except OSError:
            logger.warning("spaCy model not found. Using basic text processing.")
            self.nlp = None

        # Initialize sentiment analyzer
        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                return_all_scores=True
            )
            logger.info("Sentiment analyzer loaded successfully")
        except Exception as e:
            logger.warning(f"Sentiment analyzer not available: {e}")
            self.sentiment_analyzer = None

        # Initialize semantic similarity model
        try:
            self.semantic_similarity = pipeline(
                "feature-extraction",
                model="sentence-transformers/all-mpnet-base-v2",
                return_tensors="np"
            )
            logger.info("Semantic similarity model loaded successfully")
        except Exception as e:
            logger.warning(f"Semantic similarity model not available: {e}")
            self.semantic_similarity = None

    def extract_text_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Enhanced PDF text extraction with improved section detection and metadata"""
        try:
            doc = fitz.open(pdf_path)
            content = {
                'title': '',
                'sections': [],
                'full_text': '',
                'page_count': len(doc),
                'metadata': doc.metadata,
                'filename': os.path.basename(pdf_path),
                'file_size': os.path.getsize(pdf_path),
                'extraction_quality': 'high'
            }
            
            full_text_parts = []
            current_section = None
            font_sizes = []
            seen_headings = set()
            page_texts = {}

            # First pass: collect all text and font information
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text()
                page_texts[page_num] = page_text
                full_text_parts.append(page_text)

                # Get detailed text blocks with formatting
                try:
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" not in block:
                            continue
                        for line in block["lines"]:
                            for span in line["spans"]:
                                if span.get("size"):
                                    font_sizes.append(span["size"])
                except Exception as e:
                    logger.warning(f"Error processing page {page_num}: {e}")
                    continue

            # Determine average and large font sizes for heading detection
            if font_sizes:
                avg_font_size = np.median(font_sizes)
                large_font_threshold = np.percentile(font_sizes, 75)
            else:
                avg_font_size = 12
                large_font_threshold = 14

            # Second pass: extract sections with improved heading detection
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                try:
                    blocks = page.get_text("dict")["blocks"]
                    for block in blocks:
                        if "lines" not in block:
                            continue
                        
                        for line in block["lines"]:
                            line_text = ""
                            line_font_size = 0
                            
                            for span in line["spans"]:
                                text = span["text"].strip()
                                if text:
                                    line_text += text + " "
                                    line_font_size = max(line_font_size, span.get("size", 12))
                            
                            line_text = line_text.strip()
                            if not line_text or len(line_text) < 3:
                                continue
                            
                            # Enhanced heading detection
                            is_heading = self._is_enhanced_heading(
                                line_text, line_font_size, large_font_threshold, avg_font_size
                            )
                            
                            if is_heading:
                                # Check for duplicates using normalized text
                                normalized_heading = re.sub(r'\s+', ' ', line_text.lower().strip())
                                heading_hash = md5(normalized_heading.encode()).hexdigest()
                                
                                if heading_hash in seen_headings:
                                    continue
                                seen_headings.add(heading_hash)
                                
                                # Save previous section
                                if current_section and len(current_section['content'].strip()) > self.min_section_length:
                                    current_section['content'] = self._clean_section_content(current_section['content'])
                                    content['sections'].append(current_section)
                                
                                # Start new section
                                current_section = {
                                    'title': line_text,
                                    'page': page_num + 1,
                                    'content': '',
                                    'font_size': line_font_size,
                                    'section_id': f"sec_{len(content['sections']) + 1}",
                                    'word_count': 0
                                }
                            elif current_section:
                                current_section['content'] += line_text + '\n'
                
                except Exception as e:
                    logger.warning(f"Error processing page {page_num} blocks: {e}")
                    # Fallback: add page text as content to current section
                    if current_section:
                        current_section['content'] += page_texts.get(page_num, '') + '\n'

            # Add final section
            if current_section and len(current_section['content'].strip()) > self.min_section_length:
                current_section['content'] = self._clean_section_content(current_section['content'])
                content['sections'].append(current_section)

            # If no sections found, create sections from pages
            if not content['sections']:
                content['sections'] = self._create_page_based_sections(page_texts)
                content['extraction_quality'] = 'medium'

            # Set document title
            content['full_text'] = '\n'.join(full_text_parts)
            content['title'] = self._extract_document_title(content['sections'], content['metadata'])
            
            # Add word counts and clean up sections
            for section in content['sections']:
                section['word_count'] = len(section['content'].split())
                section['char_count'] = len(section['content'])

            doc.close()
            logger.info(f"Extracted {len(content['sections'])} sections from {pdf_path}")
            return content
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return {
                'title': os.path.basename(pdf_path),
                'sections': [],
                'full_text': '',
                'page_count': 0,
                'metadata': {},
                'filename': os.path.basename(pdf_path),
                'file_size': 0,
                'extraction_quality': 'low',
                'error': str(e)
            }

    def _is_enhanced_heading(self, text: str, font_size: float, large_threshold: float, avg_size: float) -> bool:
        """Enhanced heading detection with multiple criteria"""
        if not text or len(text.strip()) < 3:
            return False
            
        text = text.strip()
        
        # Font size check
        font_size_check = font_size > large_threshold or font_size > avg_size * 1.15
        
        # Pattern-based checks
        heading_patterns = [
            r'^[0-9]+\.?\s+[A-Z]',        # 1. Introduction
            r'^[IVX]+\.?\s+[A-Z]',        # II. Background  
            r'^[A-Z][A-Z\s]+$',           # ALL CAPS
            r'^\d+\.\d+',                 # 1.1, 2.3
            r'^[A-Za-z]\)',               # A) Section
            r'^•\s+[A-Z]',                # • Bullet
            r'^[A-Z][^.!?]*$',            # Starts capital, no sentence ending
            r'^(Chapter|Section|Part|Appendix)\s+',  # Common prefixes
            r'^(Introduction|Conclusion|Summary|Abstract|References)$',  # Common sections
        ]
        
        pattern_match = any(re.match(pattern, text, re.IGNORECASE) for pattern in heading_patterns)
        
        # Heuristic checks
        is_short = len(text) < 120 and len(text.split()) < 20
        starts_capital = text[0].isupper()
        no_sentence_ending = not text.endswith(('.', '!', '?'))
        has_multiple_words = len(text.split()) > 1
        not_too_long = len(text.split()) < 15
        
        # Combined scoring
        score = 0
        if font_size_check: score += 2
        if pattern_match: score += 3
        if is_short and starts_capital: score += 1
        if no_sentence_ending: score += 1
        if has_multiple_words and not_too_long: score += 1
        
        return score >= 3

    def _clean_section_content(self, content: str) -> str:
        """Clean and normalize section content"""
        # Remove excessive whitespace
        content = re.sub(r'\n\s*\n', '\n\n', content)
        content = re.sub(r' +', ' ', content)
        
        # Remove page numbers and common artifacts
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)
        content = re.sub(r'\n\s*Page \d+.*?\n', '\n', content, flags=re.IGNORECASE)
        
        return content.strip()

    def _create_page_based_sections(self, page_texts: Dict[int, str]) -> List[Dict[str, Any]]:
        """Create sections based on pages when heading detection fails"""
        sections = []
        for page_num, text in page_texts.items():
            if len(text.strip()) > self.min_section_length:
                # Try to extract a title from the first line
                lines = [line.strip() for line in text.split('\n') if line.strip()]
                title = lines[0][:80] + "..." if lines and len(lines[0]) > 80 else lines[0] if lines else f"Page {page_num + 1}"
                
                sections.append({
                    'title': title,
                    'page': page_num + 1,
                    'content': text,
                    'font_size': 12,
                    'section_id': f"page_{page_num + 1}",
                    'word_count': len(text.split()),
                    'char_count': len(text)
                })
        return sections

    def _extract_document_title(self, sections: List[Dict], metadata: Dict) -> str:
        """Extract document title from sections or metadata"""
        # Try metadata first
        if metadata.get('title') and metadata['title'].strip() != '':
            return metadata['title']
        
        # Try first section
        if sections and sections[0]['title']:
            return sections[0]['title']
        
        # Try subject from metadata
        if metadata.get('subject'):
            return metadata['subject']
            
        return "Untitled Document"

    def calculate_document_relevance(self, doc_content: Dict[str, Any], persona: str, job: str) -> float:
        """Calculate overall document relevance score"""
        # Combine title, first few sections, and metadata
        doc_text = f"{doc_content['title']} "
        
        # Add content from top sections
        for section in doc_content['sections'][:5]:
            doc_text += f"{section['title']} {section['content'][:200]} "
        
        if doc_content['metadata'].get('subject'):
            doc_text += doc_content['metadata']['subject']
        if doc_content['metadata'].get('keywords'):
            doc_text += doc_content['metadata']['keywords']
            
        query_text = f"{persona} {job}"
        
        # Calculate similarity
        similarity = self._calculate_advanced_similarity(doc_text[:2000], query_text)
        
        # Apply quality bonus
        quality_bonus = {
            'high': 1.0,
            'medium': 0.9,
            'low': 0.7
        }.get(doc_content.get('extraction_quality', 'medium'), 0.8)
        
        # Page count consideration (more pages might be more comprehensive)
        page_bonus = min(1.2, 1.0 + (doc_content['page_count'] * 0.01))
        
        return min(similarity * quality_bonus * page_bonus, 1.0)

    def calculate_section_relevance(self, section: Dict[str, Any], persona: str, job: str, doc_relevance: float = 1.0) -> float:
        """Enhanced section relevance calculation with context awareness"""
        section_text = f"{section['title']} {section['content']}"
        query_text = f"{persona} {job}"
        
        # Base similarity calculation
        similarity = self._calculate_advanced_similarity(section_text, query_text)
        
        # Context-aware boosting
        boost_factors = 1.0
        
        # Keyword matching with importance weighting
        important_keywords = self._extract_contextual_keywords(query_text)
        keyword_matches = 0
        for keyword, weight in important_keywords.items():
            if keyword.lower() in section_text.lower():
                keyword_matches += weight
        
        boost_factors += min(keyword_matches * 0.1, 0.5)
        
        # Position boost (earlier sections often more important)
        position_boost = max(0.8, 1.2 - (section.get('page', 1) * 0.03))
        
        # Content quality boost
        word_count = section.get('word_count', len(section.get('content', '').split()))
        quality_boost = min(1.2, 0.7 + (word_count / 500))  # Optimal around 500 words
        
        # Font size importance (larger headings = more important)
        font_boost = 1.0 + max(0, (section.get('font_size', 12) - 12) * 0.02)
        
        # Document context boost
        doc_context_boost = 0.8 + (doc_relevance * 0.4)
        
        final_score = similarity * boost_factors * position_boost * quality_boost * font_boost * doc_context_boost
        return min(final_score, 1.0)

    def _calculate_advanced_similarity(self, text1: str, text2: str) -> float:
        """Advanced similarity calculation with fallback mechanisms"""
        # Try semantic similarity first
        if self.semantic_similarity:
            try:
                # Truncate texts to avoid token limits
                text1 = text1[:1000]
                text2 = text2[:1000]
                
                emb1 = self.semantic_similarity(text1)
                emb2 = self.semantic_similarity(text2)
                
                # Handle different output formats
                if isinstance(emb1, list):
                    emb1 = np.array(emb1)
                if isinstance(emb2, list):
                    emb2 = np.array(emb2)
                
                # Average pooling for sentence embeddings
                if len(emb1.shape) > 2:
                    emb1 = np.mean(emb1, axis=1)
                if len(emb2.shape) > 2:
                    emb2 = np.mean(emb2, axis=1)
                
                emb1 = np.mean(emb1, axis=0, keepdims=True)
                emb2 = np.mean(emb2, axis=0, keepdims=True)
                
                semantic_sim = cosine_similarity(emb1, emb2)[0][0]
                if not np.isnan(semantic_sim):
                    # Combine with TF-IDF for robustness
                    tfidf_sim = self._calculate_tfidf_similarity(text1, text2)
                    return semantic_sim * 0.7 + tfidf_sim * 0.3
                    
            except Exception as e:
                logger.warning(f"Semantic similarity failed: {e}")
        
        # Fallback to TF-IDF
        return self._calculate_tfidf_similarity(text1, text2)

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF cosine similarity with error handling"""
        try:
            vectorizer = TfidfVectorizer(
                stop_words='english', 
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(similarity) if not np.isnan(similarity) else 0.0
        except Exception as e:
            logger.warning(f"TF-IDF similarity failed: {e}")
            return self._fallback_keyword_similarity(text1.lower(), text2.lower())

    def _fallback_keyword_similarity(self, text1: str, text2: str) -> float:
        """Fallback keyword-based similarity"""
        words1 = set(re.findall(r'\b\w{3,}\b', text1))
        words2 = set(re.findall(r'\b\w{3,}\b', text2))
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Boost for important word matches
        important_matches = 0
        important_words = {'challenge', 'problem', 'solution', 'data', 'research', 'analysis', 'business', 'technical'}
        for word in intersection:
            if word in important_words:
                important_matches += 1
                
        importance_boost = 1.0 + (important_matches * 0.1)
        return min(jaccard * importance_boost, 1.0)

    def _extract_contextual_keywords(self, text: str) -> Dict[str, float]:
        """Extract keywords with contextual importance weights"""
        keywords = {}
        
        # High-importance domain patterns
        domain_patterns = {
            r'\b(healthcare|medical|clinical|pharmaceutical|drug|patient|hospital)\w*\b': 2.0,
            r'\b(startup|entrepreneur|funding|venture|investment|revenue|business)\w*\b': 2.0,
            r'\b(technology|technical|engineering|software|digital|data|AI|ML)\w*\b': 1.8,
            r'\b(research|analysis|study|methodology|findings|results|evidence)\w*\b': 1.5,
            r'\b(compliance|regulation|privacy|security|legal|policy)\w*\b': 1.7,
            r'\b(challenge|problem|issue|difficulty|barrier|limitation)\w*\b': 1.6,
            r'\b(solution|approach|method|strategy|framework|model)\w*\b': 1.6,
            r'\b(market|customer|user|client|consumer|audience)\w*\b': 1.4,
        }
        
        for pattern, weight in domain_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                keywords[match.lower()] = weight
        
        # Extract important nouns and adjectives using spaCy
        if self.nlp:
            try:
                doc = self.nlp(text[:500])  # Limit to avoid processing issues
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                        len(token.text) > 3 and 
                        not token.is_stop and 
                        not token.is_punct):
                        lemma = token.lemma_.lower()
                        if lemma not in keywords:
                            keywords[lemma] = 1.0
            except Exception as e:
                logger.warning(f"spaCy processing failed: {e}")
        
        return keywords

    def detect_and_remove_duplicates(self, sections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Advanced duplicate detection and removal"""
        if not sections:
            return sections
            
        unique_sections = []
        seen_hashes = set()
        
        for section in sections:
            # Create multiple hash signatures
            title_hash = md5(re.sub(r'\s+', ' ', section['title'].lower().strip()).encode()).hexdigest()
            content_preview = section.get('content', '')[:200]
            content_hash = md5(re.sub(r'\s+', ' ', content_preview.lower().strip()).encode()).hexdigest()
            combined_hash = md5((section['title'] + content_preview).lower().encode()).hexdigest()
            
            # Check for exact duplicates
            if combined_hash in seen_hashes:
                continue
                
            # Check for near-duplicates using similarity
            is_duplicate = False
            for existing_section in unique_sections:
                existing_content = existing_section.get('content', '')[:200]
                similarity = self._calculate_tfidf_similarity(content_preview, existing_content)
                
                if similarity > self.duplicate_threshold:
                    # Keep the one with higher relevance score
                    if section.get('relevance_score', 0) > existing_section.get('relevance_score', 0):
                        unique_sections.remove(existing_section)
                        break
                    else:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_hashes.add(combined_hash)
                unique_sections.append(section)
        
        logger.info(f"Removed {len(sections) - len(unique_sections)} duplicate sections")
        return unique_sections

    def extract_enhanced_subsections(self, section: Dict[str, Any], persona: str, job: str) -> List[Dict[str, Any]]:
        """Extract and analyze subsections with enhanced relevance scoring"""
        content = section.get('content', '')
        if not content or len(content.strip()) < 100:
            return []
        
        # Split into meaningful paragraphs
        paragraphs = []
        for para in content.split('\n\n'):
            para = para.strip()
            if len(para) > 80:  # Minimum meaningful paragraph length
                paragraphs.append(para)
        
        if not paragraphs:
            # Fallback: split by sentences
            sentences = re.split(r'[.!?]+', content)
            paragraphs = [s.strip() for s in sentences if len(s.strip()) > 80]
        
        subsections = []
        seen_hashes = set()
        query_text = f"{persona} {job}"
        
        for i, paragraph in enumerate(paragraphs[:10]):  # Limit for performance
            # Create hash for duplicate detection
            para_hash = md5(paragraph[:100].lower().encode()).hexdigest()
            if para_hash in seen_hashes:
                continue
            seen_hashes.add(para_hash)
            
            # Calculate relevance to query
            relevance_score = self._calculate_advanced_similarity(paragraph, query_text)
            
            # Only include if sufficiently relevant
            if relevance_score > 0.15:
                # Create refined text snippet
                sentences = re.split(r'[.!?]+', paragraph)
                refined_sentences = [s.strip() for s in sentences if s.strip()]
                
                if len(refined_sentences) >= 2:
                    refined_text = '. '.join(refined_sentences[:3]) + '.'
                else:
                    refined_text = paragraph[:300] + ('...' if len(paragraph) > 300 else '')
                
                subsections.append({
                    'document': section.get('document', ''),
                    'page_number': section.get('page', 1),
                    'section_title': section['title'],
                    'refined_text': refined_text,
                    'full_text': paragraph,
                    'subsection_id': f"{section.get('section_id', 'unknown')}_sub_{i+1}",
                    'relevance_score': relevance_score,
                    'word_count': len(paragraph.split()),
                    'hash': para_hash
                })
        
        # Sort by relevance and return top subsections
        subsections.sort(key=lambda x: x['relevance_score'], reverse=True)
        return subsections[:5]  # Top 5 most relevant subsections per section

    def process_documents(self, input_dir: str, persona: str, job: str) -> Dict[str, Any]:
        """Enhanced main processing pipeline"""
        pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
        
        if not pdf_files:
            raise ValueError("No PDF files found in input directory")
        
        logger.info(f"Processing {len(pdf_files)} PDF files")
        
        # Phase 1: Extract content from all documents
        all_documents = {}
        document_relevance_scores = {}
        
        for pdf_file in pdf_files:
            path = os.path.join(input_dir, pdf_file)
            logger.info(f"Extracting content from {pdf_file}")
            
            content = self.extract_text_from_pdf(path)
            content['filename'] = pdf_file
            all_documents[pdf_file] = content
            
            # Calculate document-level relevance
            doc_relevance = self.calculate_document_relevance(content, persona, job)
            document_relevance_scores[pdf_file] = doc_relevance
            logger.info(f"Document relevance for {pdf_file}: {doc_relevance:.3f}")
        
        # Phase 2: Process sections from all documents
        all_sections = []
        sections_per_document = {}  # Track sections per document for transparency

        for pdf_file, doc_content in all_documents.items():
            doc_relevance = document_relevance_scores[pdf_file]
            doc_sections = []
            
            for section in doc_content['sections']:
                section['document'] = pdf_file
                section['document_relevance'] = doc_relevance
                
                # Calculate section relevance
                section_score = self.calculate_section_relevance(section, persona, job, doc_relevance)
                section['relevance_score'] = section_score
                
                # Include all sections initially
                all_sections.append(section)
                doc_sections.append(section)
            
            sections_per_document[pdf_file] = len(doc_sections)
            
            # Ensure each document has at least one representative section in final output
            if not doc_sections:
                # Create a placeholder section if document has no extractable sections
                placeholder_section = {
                    'document': pdf_file,
                    'document_relevance': doc_relevance,
                    'title': f"Document Overview - {doc_content['title']}",
                    'content': doc_content['full_text'][:500] + '...' if len(doc_content['full_text']) > 500 else doc_content['full_text'],
                    'page': 1,
                    'relevance_score': max(0.1, doc_relevance * 0.5),  # Ensure minimum visibility
                    'section_id': f"placeholder_{pdf_file}",
                    'word_count': len(doc_content['full_text'].split()),
                    'is_placeholder': True
                }
                all_sections.append(placeholder_section)
                sections_per_document[pdf_file] = 1

        # Log document representation
        for pdf_file, section_count in sections_per_document.items():
            logger.info(f"Document {pdf_file}: {section_count} sections extracted")
        
        logger.info(f"Found {len(all_sections)} total sections across all documents")
        
        # Phase 3: Remove duplicates and rank sections
        unique_sections = self.detect_and_remove_duplicates(all_sections)
        unique_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Phase 4: Select top sections ensuring document representation
        # First, ensure each document gets at least one section if possible
        represented_docs = set()
        guaranteed_sections = []
        remaining_sections = []

        for section in unique_sections:
            if section['document'] not in represented_docs and len(guaranteed_sections) < len(all_documents):
                guaranteed_sections.append(section)
                represented_docs.add(section['document'])
            else:
                remaining_sections.append(section)

        # Fill remaining slots with highest scoring sections
        available_slots = 25 - len(guaranteed_sections)
        top_sections = guaranteed_sections + remaining_sections[:available_slots]

        # Log document representation in final output
        logger.info(f"Final output represents {len(represented_docs)} out of {len(all_documents)} documents")
        for doc in all_documents.keys():
            if doc not in represented_docs:
                logger.warning(f"Document {doc} not represented in final output due to low relevance")
                extracted_sections.append({
                    'document': doc,
                    'section_title': "No relevant section found in analysis",
                    'importance_rank': None,
                    'relevance_score': 0.0,
                    'page_number': None,
                    'content_preview': "",
                    'section_id': f"no_relevant_{doc}"
        })
        extracted_sections = []
        subsection_analysis = []
        
        for i, section in enumerate(top_sections):
            extracted_sections.append({
                'document': section['document'],
                'document_relevance': section['document_relevance'],
                'page_number': section['page'],
                'section_title': section['title'],
                'importance_rank': i + 1,
                'relevance_score': section['relevance_score'],
                'word_count': section.get('word_count', 0),
                'content_preview': section['content'][:150] + '...' if len(section['content']) > 150 else section['content'],
                'section_id': section.get('section_id', f'sec_{i+1}')
            })
            
            # Extract subsections from top 15 sections only (performance consideration)
            if i < 15:
                subsections = self.extract_enhanced_subsections(section, persona, job)
                for sub in subsections:
                    sub['section_importance_rank'] = i + 1
                    subsection_analysis.append(sub)
        
        # Phase 5: Rank documents by overall relevance
        ranked_documents = [
            {
                'filename': pdf_file,
                'title': doc_content['title'],
                'relevance_score': document_relevance_scores[pdf_file],
                'page_count': doc_content['page_count'],
                'section_count': len([s for s in all_sections if s['document'] == pdf_file]),
                'extraction_quality': doc_content.get('extraction_quality', 'medium')
            }
            for pdf_file, doc_content in all_documents.items()
        ]
        ranked_documents.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        # Sort subsections by relevance
        subsection_analysis.sort(key=lambda x: x['relevance_score'], reverse=True)
        
        logger.info(f"Final output: {len(extracted_sections)} sections, {len(subsection_analysis)} subsections")
        
        return {
            'metadata': {
                'input_documents': pdf_files,
                'persona': persona,
                'job_to_be_done': job,
                'processing_timestamp': datetime.now().isoformat(),
                'total_documents_processed': len(pdf_files),
                'total_sections_found': len(all_sections),
                'unique_sections_returned': len(extracted_sections),
                'subsections_returned': len(subsection_analysis),
                'average_document_relevance': np.mean(list(document_relevance_scores.values())),
                'processing_quality': 'enhanced'
            },
            'document_rankings': ranked_documents,
            'extracted_sections': extracted_sections,
            'subsection_analysis': subsection_analysis,
            'summary': self._generate_enhanced_summary(
                extracted_sections, subsection_analysis, persona, job, ranked_documents
            )
        }

    def _generate_enhanced_summary(self, sections: List[Dict], subsections: List[Dict], 
                                 persona: str, job: str, documents: List[Dict]) -> Dict:
        """Generate comprehensive summary with insights"""
        top_documents = [doc['filename'] for doc in documents[:3]]
        top_section_titles = [s['section_title'] for s in sections[:5]]
        key_insights = [s['refined_text'][:100] + '...' for s in subsections[:3]]
        
        # Calculate distribution statistics
        doc_relevance_scores = [doc['relevance_score'] for doc in documents]
        section_relevance_scores = [s['relevance_score'] for s in sections]
        
        return {
            'persona': persona,
            'job_to_be_done': job,
            'most_relevant_documents': top_documents,
            'top_section_titles': top_section_titles,
            'key_insights': key_insights,
            'statistics': {
                'total_documents_analyzed': len(documents),
                'total_relevant_sections': len(sections),
                'total_subsections': len(subsections),
                'avg_document_relevance': round(np.mean(doc_relevance_scores), 3),
                'avg_section_relevance': round(np.mean(section_relevance_scores), 3),
                'max_document_relevance': round(max(doc_relevance_scores), 3),
                'max_section_relevance': round(max(section_relevance_scores), 3)
            },
            'recommendations': self._generate_recommendations(documents, sections, persona, job)
        }

    def _generate_recommendations(self, documents: List[Dict], sections: List[Dict], 
                                persona: str, job: str) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        # Document-level recommendations
        high_relevance_docs = [doc for doc in documents if doc['relevance_score'] > 0.6]
        if high_relevance_docs:
            recommendations.append(
                f"Focus on {len(high_relevance_docs)} highly relevant documents: " + 
                ", ".join([doc['filename'] for doc in high_relevance_docs[:3]])
            )
        
        # Section-level recommendations
        high_relevance_sections = [s for s in sections if s['relevance_score'] > 0.7]
        if high_relevance_sections:
            recommendations.append(
                f"Prioritize {len(high_relevance_sections)} high-relevance sections for detailed review"
            )
        
        # Content recommendations based on persona
        if 'startup' in persona.lower() or 'founder' in persona.lower():
            business_sections = [s for s in sections if any(keyword in s['section_title'].lower() 
                               for keyword in ['business', 'funding', 'market', 'strategy', 'financial'])]
            if business_sections:
                recommendations.append(
                    f"Found {len(business_sections)} business-focused sections relevant to startup needs"
                )
        
        if 'healthcare' in persona.lower() or 'medical' in persona.lower():
            health_sections = [s for s in sections if any(keyword in s['section_title'].lower() 
                             for keyword in ['health', 'medical', 'clinical', 'patient', 'drug'])]
            if health_sections:
                recommendations.append(
                    f"Identified {len(health_sections)} healthcare-related sections for domain expertise"
                )
        
        # Quality recommendations
        low_quality_docs = [doc for doc in documents if doc['extraction_quality'] == 'low']
        if low_quality_docs:
            recommendations.append(
                f"Consider manual review of {len(low_quality_docs)} documents with lower extraction quality"
            )
        
        return recommendations[:5]  # Limit to top 5 recommendations


def main():
    """Enhanced main function with better error handling and logging"""
    input_dir = '/app/input'
    output_dir = '/app/output'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Load configuration
        config_path = os.path.join(input_dir, 'config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                persona = config.get('persona', 'Researcher')
                job = config.get('job_to_be_done', 'Analyze documents for insights')
        else:
            logger.warning("No config.json found, using default persona and job")
            persona = "Researcher"
            job = "Analyze documents for insights"

        logger.info(f"Starting document processing")
        logger.info(f"Persona: {persona}")
        logger.info(f"Job to be done: {job}")

        # Initialize extractor and process documents
        extractor = EnhancedPersonaDrivenExtractor()
        result = extractor.process_documents(input_dir, persona, job)

        # Save main results
        output_file = os.path.join(output_dir, 'challenge1b_output.json')
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        # Save additional analytics file
        analytics_file = os.path.join(output_dir, 'document_analytics.json')
        analytics_data = {
            'document_rankings': result['document_rankings'],
            'processing_metadata': result['metadata'],
            'summary_statistics': result['summary']['statistics'],
            'recommendations': result['summary']['recommendations']
        }
        with open(analytics_file, 'w', encoding='utf-8') as f:
            json.dump(analytics_data, f, indent=2, ensure_ascii=False)

        # Print comprehensive summary
        print("\n" + "="*80)
        print("DOCUMENT PROCESSING RESULTS")
        print("="*80)
        
        print(f"\n📊 PROCESSING SUMMARY:")
        print(f"   • Persona: {persona}")
        print(f"   • Job: {job}")
        print(f"   • Documents processed: {result['metadata']['total_documents_processed']}")
        print(f"   • Total sections found: {result['metadata']['total_sections_found']}")
        print(f"   • Unique sections returned: {result['metadata']['unique_sections_returned']}")
        print(f"   • Subsections analyzed: {result['metadata']['subsections_returned']}")
        
        print(f"\n📈 RELEVANCE STATISTICS:")
        stats = result['summary']['statistics']
        print(f"   • Average document relevance: {stats['avg_document_relevance']}")
        print(f"   • Average section relevance: {stats['avg_section_relevance']}")
        print(f"   • Highest document relevance: {stats['max_document_relevance']}")
        print(f"   • Highest section relevance: {stats['max_section_relevance']}")
        
        print(f"\n🏆 TOP DOCUMENTS:")
        for i, doc in enumerate(result['document_rankings'][:5], 1):
            print(f"   {i}. {doc['filename']} (relevance: {doc['relevance_score']:.3f})")
        
        print(f"\n📋 TOP SECTIONS:")
        for i, section in enumerate(result['extracted_sections'][:5], 1):
            print(f"   {i}. {section['section_title'][:60]}... (score: {section['relevance_score']:.3f})")
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(result['summary']['recommendations'], 1):
            print(f"   {i}. {rec}")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"   • Main results: {output_file}")
        print(f"   • Analytics: {analytics_file}")
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETED SUCCESSFULLY")
        print("="*80)

    except Exception as e:
        logger.error(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        
        # Create error output
        error_output = {
            'error': str(e),
            'timestamp': datetime.now().isoformat(),
            'persona': locals().get('persona', 'Unknown'),
            'job_to_be_done': locals().get('job', 'Unknown'),
            'traceback': traceback.format_exc()
        }
        
        error_file = os.path.join(output_dir, 'error_log.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_output, f, indent=2, ensure_ascii=False)
        
        print(f"\nError occurred. Details saved to: {error_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()