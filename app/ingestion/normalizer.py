"""Content normalizer for email ingestion pipeline."""

import re
from typing import Dict, Any, List, Tuple
import structlog
from bs4 import BeautifulSoup
import hashlib
from datetime import datetime

from .models import EmailContent

logger = structlog.get_logger(__name__)


class HTMLNormalizer:
    """Normalizes HTML content to clean text."""
    
    def __init__(self):
        """Initialize HTML normalizer."""
        self.soup = None
    
    def normalize_html(self, html_content: str) -> str:
        """Convert HTML to clean, readable text."""
        try:
            # Parse HTML with BeautifulSoup
            self.soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in self.soup(["script", "style"]):
                script.decompose()
            
            # Extract text content
            text = self.soup.get_text()
            
            # Clean up whitespace
            text = self._clean_whitespace(text)
            
            # Preserve some structure
            text = self._preserve_structure(text)
            
            return text
            
        except Exception as exc:
            logger.error("Failed to normalize HTML", exc_info=exc)
            # Fallback: return raw HTML stripped of tags
            return self._fallback_html_strip(html_content)
    
    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace."""
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Replace multiple newlines with double newline
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _preserve_structure(self, text: str) -> str:
        """Preserve some structural elements."""
        # Preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Preserve list items
        text = re.sub(r'^\s*[-*]\s+', '\n• ', text, flags=re.MULTILINE)
        
        # Preserve headers (simplified)
        text = re.sub(r'^\s*(#{1,6})\s+', '\n', text, flags=re.MULTILINE)
        
        return text
    
    def _fallback_html_strip(self, html_content: str) -> str:
        """Fallback HTML stripping when BeautifulSoup fails."""
        # Simple regex-based HTML tag removal
        text = re.sub(r'<[^>]+>', '', html_content)
        text = re.sub(r'&[a-zA-Z]+;', ' ', text)  # Remove HTML entities
        text = self._clean_whitespace(text)
        return text


class ContentNormalizer:
    """Normalizes email content for processing."""
    
    def __init__(self):
        """Initialize content normalizer."""
        self.html_normalizer = HTMLNormalizer()
    
    def normalize_content(self, raw_content: bytes | str, content_type: str, charset: str = None) -> EmailContent:
        """Normalize email content and create EmailContent object."""
        try:
            # Handle both bytes and str input safely
            if isinstance(raw_content, str):
                text_content = raw_content
                input_type = "str"
                # Convert str to bytes for hash calculation
                raw_bytes = raw_content.encode('utf-8')
            else:
                input_type = "bytes"
                raw_bytes = raw_content
                # Decode content
                if charset:
                    try:
                        text_content = raw_content.decode(charset)
                    except UnicodeDecodeError:
                        text_content = raw_content.decode('utf-8', errors='replace')
                else:
                    text_content = raw_content.decode('utf-8', errors='replace')
            
            # Normalize based on content type
            if content_type == 'text/html':
                normalized_text = self.html_normalizer.normalize_html(text_content)
            else:
                normalized_text = self._normalize_plain_text(text_content)
            
            # Apply signature and quote stripping
            normalized_text = self._strip_signatures_and_quotes(normalized_text)
            
            # Detect language
            detected_language = self._detect_language(normalized_text)
            
            # Calculate content metrics
            word_count = self._count_words(normalized_text)
            sentence_count = self._count_sentences(normalized_text)
            paragraph_count = self._count_paragraphs(normalized_text)
            
            # Calculate hashes
            raw_content_hash = self._calculate_hash(raw_bytes)
            normalized_text_hash = self._calculate_hash(normalized_text.encode('utf-8'))
            
            # Create normalization manifest
            normalization_manifest = self._create_normalization_manifest(
                input_type, content_type, charset, detected_language, normalized_text
            )
            
            return EmailContent(
                raw_content=raw_content,
                raw_content_hash=raw_content_hash,
                normalized_text=normalized_text,
                normalized_text_hash=normalized_text_hash,
                content_length=len(raw_content),
                word_count=word_count,
                sentence_count=sentence_count,
                paragraph_count=paragraph_count,
                detected_language=detected_language,
                language_confidence=0.8,  # Placeholder confidence
                normalization_manifest=normalization_manifest
            )
            
        except Exception as exc:
            logger.error("Failed to normalize content", exc_info=exc)
            raise ValueError(f"Failed to normalize content: {str(exc)}")
    
    def _normalize_plain_text(self, text: str) -> str:
        """Normalize plain text content."""
        # Clean up whitespace
        text = re.sub(r'\r\n', '\n', text)  # Normalize line endings
        text = re.sub(r'\r', '\n', text)    # Handle old Mac line endings
        
        # Clean up excessive whitespace
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs to single space
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Multiple newlines to double newline
        
        return text.strip()
    
    def _strip_signatures_and_quotes(self, text: str) -> str:
        """Strip email signatures and quoted text."""
        lines = text.split('\n')
        cleaned_lines = []
        
        in_signature = False
        in_quoted_text = False
        
        for line in lines:
            # Check for signature indicators
            if self._is_signature_line(line):
                in_signature = True
                continue
            
            # Check for quoted text indicators
            if self._is_quoted_line(line):
                in_quoted_text = True
                continue
            
            # If we're in signature or quoted text, skip the line
            if in_signature or in_quoted_text:
                continue
            
            # Check if we're exiting quoted text
            if in_quoted_text and not self._is_quoted_line(line):
                in_quoted_text = False
            
            # Check if we're exiting signature
            if in_signature and not self._is_signature_line(line):
                in_signature = False
            
            # Add the line if we're not in signature or quoted text
            if not in_signature and not in_quoted_text:
                cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _is_signature_line(self, line: str) -> bool:
        """Check if a line indicates the start of a signature."""
        line = line.strip()
        
        # Common signature patterns
        signature_patterns = [
            r'^--\s*$',  # Standard signature separator
            r'^Best regards,?$',
            r'^Sincerely,?$',
            r'^Thanks,?$',
            r'^Regards,?$',
            r'^Yours truly,?$',
            r'^Kind regards,?$',
            r'^Cheers,?$',
            r'^Sent from my iPhone$',
            r'^Sent from my iPad$',
            r'^Sent from my Android$',
            r'^Get Outlook for .*$',
            r'^Sent from .*$'
        ]
        
        for pattern in signature_patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return True
        
        return False
    
    def _is_quoted_line(self, line: str) -> bool:
        """Check if a line is quoted text."""
        line = line.strip()
        
        # Check for quote indicators
        if line.startswith('>'):
            return True
        
        # Check for forwarded message indicators
        if re.match(r'^From:.*$', line):
            return True
        
        if re.match(r'^Sent:.*$', line):
            return True
        
        if re.match(r'^To:.*$', line):
            return True
        
        if re.match(r'^Subject:.*$', line):
            return True
        
        if re.match(r'^Date:.*$', line):
            return True
        
        return False
    
    def _detect_language(self, text: str) -> str:
        """Detect the language of the text."""
        # Simple language detection based on common words
        # This is a placeholder implementation - in production, use a proper language detection library
        
        # Count common English words
        english_words = ['the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that', 'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they', 'at', 'be', 'this', 'have', 'from', 'or', 'one', 'had', 'by', 'word', 'but', 'not', 'what', 'all', 'were', 'we', 'when', 'your', 'can', 'said', 'there', 'use', 'an', 'each', 'which', 'she', 'do', 'how', 'their', 'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more', 'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get', 'come', 'made', 'may', 'part']
        
        # Count common Spanish words
        spanish_words = ['el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se', 'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para', 'al', 'del', 'como', 'más', 'o', 'pero', 'sus', 'me', 'hasta', 'hay', 'donde', 'han', 'quien', 'están', 'estado', 'desde', 'todo', 'nos', 'durante', 'todos', 'uno', 'les', 'ni', 'contra', 'otros', 'ese', 'eso', 'ante', 'ellos', 'e', 'esto', 'mí', 'antes', 'algunos', 'qué', 'unos', 'yo', 'otro', 'otras', 'otra', 'él', 'tanto', 'esa', 'estos', 'mucho', 'quienes', 'nada', 'muchos', 'cual', 'poco', 'ella', 'estar', 'estas', 'algunas', 'algo', 'nosotros']
        
        # Count common French words
        french_words = ['le', 'de', 'un', 'à', 'être', 'et', 'en', 'avoir', 'ne', 'je', 'son', 'que', 'se', 'qui', 'ce', 'il', 'pas', 'sur', 'faire', 'plus', 'dire', 'me', 'du', 'tout', 'pouvoir', 'autre', 'on', 'devoir', 'même', 'prendre', 'nous', 'comme', 'leur', 'dans', 'bien', 'elle', 'si', 'par', 'tout', 'y', 'devoir', 'encore', 'grand', 'peu', 'mon', 'meilleur', 'donner', 'bon', 'ce', 'monde', 'si', 'y', 'faire', 'regarder', 'autre', 'peu', 'alors', 'venir', 'comprendre', 'notre', 'deux', 'depuis', 'parfait', 'lire', 'cela', 'jamais', 'aussi', 'quand', 'très', 'toujours', 'peut', 'attendre', 'même', 'partir', 'joli', 'bonheur', 'semble', 'surtout', 'encore', 'nouveau', 'donner', 'côté', 'continuer', 'début', 'guère', 'trop', 'tôt', 'vouloir', 'demander', 'grand', 'trois', 'cela', 'jamais', 'aussi', 'quand', 'très', 'toujours', 'peut', 'attendre', 'même', 'partir', 'joli', 'bonheur', 'semble', 'surtout', 'encore', 'nouveau', 'donner', 'côté', 'continuer', 'début', 'guère', 'trop', 'tôt', 'vouloir', 'demander', 'grand', 'trois']
        
        # Count words in text (case insensitive)
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        # Count matches for each language
        english_count = sum(1 for word in words if word in english_words)
        spanish_count = sum(1 for word in words if word in spanish_words)
        french_count = sum(1 for word in words if word in french_words)
        
        # Determine language based on highest count
        if english_count > spanish_count and english_count > french_count:
            return 'en'
        elif spanish_count > french_count:
            return 'es'
        elif french_count > 0:
            return 'fr'
        else:
            return 'en'  # Default to English
    
    def _count_words(self, text: str) -> int:
        """Count words in text."""
        words = re.findall(r'\b\w+\b', text)
        return len(words)
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)
    
    def _count_paragraphs(self, text: str) -> int:
        """Count paragraphs in text."""
        paragraphs = re.split(r'\n\s*\n', text)
        # Filter out empty paragraphs
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        return len(paragraphs)
    
    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    def _count_stripped_blocks(self, text: str) -> int:
        """Count the number of stripped blocks (signatures, quotes)."""
        lines = text.split('\n')
        stripped_count = 0
        
        for line in lines:
            if self._is_signature_line(line) or self._is_quoted_line(line):
                stripped_count += 1
        
        return stripped_count
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple approximation: 1 token ≈ 4 characters
        return max(1, len(text) // 4)
    
    def _create_normalization_manifest(self, input_type: str, content_type: str, charset: str, language: str, normalized_text: str) -> Dict[str, Any]:
        """Create normalization manifest."""
        return {
            'input_type': input_type,
            'content_type': content_type,
            'charset': charset or 'utf-8',
            'detected_language': language,
            'stripped_blocks': self._count_stripped_blocks(normalized_text),
            'lang': language,
            'length_chars': len(normalized_text),
            'token_estimate': self._estimate_tokens(normalized_text),
            'normalization_steps': [
                'html_to_text' if content_type == 'text/html' else 'plain_text_cleanup',
                'signature_stripping',
                'quote_stripping',
                'whitespace_normalization'
            ],
            'timestamp': str(datetime.utcnow())
        }
