"""Semantic chunking for email ingestion pipeline."""

import asyncio
from typing import List, Optional, Dict, Any
import structlog
import re
import hashlib

from .models import ChunkInfo, generate_chunk_uid

logger = structlog.get_logger(__name__)


class SemanticChunker:
    """Creates semantic chunks from text content."""
    
    def __init__(self):
        """Initialize semantic chunker."""
        self.sentence_boundary_pattern = re.compile(r'[.!?]+[\s\n]+')
        self.paragraph_boundary_pattern = re.compile(r'\n\s*\n')
    
    async def create_chunks(self, text: str, email_id: str, 
                           attachment_id: Optional[str] = None,
                           chunk_size: int = 1000, 
                           overlap: int = 200) -> List[ChunkInfo]:
        """Create semantic chunks from text content."""
        try:
            if not text or len(text.strip()) == 0:
                return []
            
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Split into sentences
            sentences = self._split_into_sentences(cleaned_text)
            
            if not sentences:
                return []
            
            # Create chunks
            chunks = []
            current_chunk = ""
            chunk_index = 0
            start_position = 0
            
            for i, sentence in enumerate(sentences):
                # Check if adding this sentence would exceed chunk size
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    # Create chunk
                    chunk = self._create_chunk(
                        current_chunk, email_id, attachment_id, chunk_index,
                        start_position, start_position + len(current_chunk)
                    )
                    chunks.append(chunk)
                    
                    # Start new chunk with overlap
                    if overlap > 0:
                        # Find overlap point
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                        current_chunk = overlap_text + sentence
                        start_position = start_position + len(current_chunk) - len(overlap_text) - len(sentence)
                    else:
                        current_chunk = sentence
                        start_position = start_position + len(current_chunk) - len(sentence)
                    
                    chunk_index += 1
                else:
                    current_chunk += sentence
                    if not current_chunk:  # First sentence
                        start_position = 0
            
            # Add final chunk
            if current_chunk:
                chunk = self._create_chunk(
                    current_chunk, email_id, attachment_id, chunk_index,
                    start_position, start_position + len(current_chunk)
                )
                chunks.append(chunk)
            
            logger.info("Chunks created successfully",
                       email_id=email_id,
                       total_chunks=len(chunks),
                       chunk_size=chunk_size,
                       overlap=overlap)
            
            return chunks
            
        except Exception as exc:
            logger.error("Failed to create chunks",
                        email_id=email_id,
                        exc_info=exc)
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for chunking."""
        try:
            # Remove excessive whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Normalize line endings
            text = text.replace('\r\n', '\n').replace('\r', '\n')
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception:
            return text
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences while preserving structure."""
        try:
            # Split by sentence boundaries
            sentences = self.sentence_boundary_pattern.split(text)
            
            # Clean up sentences
            cleaned_sentences = []
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    # Add back the sentence ending punctuation
                    if not sentence.endswith(('.', '!', '?')):
                        sentence += '.'
                    cleaned_sentences.append(sentence)
            
            return cleaned_sentences
            
        except Exception:
            # Fallback: simple split by periods
            return [s.strip() + '.' for s in text.split('.') if s.strip()]
    
    def _create_chunk(self, content: str, email_id: str, 
                      attachment_id: Optional[str], chunk_index: int,
                      start_position: int, end_position: int) -> ChunkInfo:
        """Create a ChunkInfo object for a chunk."""
        try:
            # Calculate content hash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Generate chunk UID
            chunk_uid = generate_chunk_uid(email_id, chunk_index, content_hash)
            
            # Count tokens (simple approximation)
            token_count = self._count_tokens(content)
            
            # Determine chunk type
            chunk_type = "attachment" if attachment_id else "text"
            
            # Check if chunk respects semantic boundaries
            semantic_boundary = self._respects_semantic_boundaries(content)
            
            return ChunkInfo(
                chunk_uid=chunk_uid,
                chunk_index=chunk_index,
                chunk_hash=content_hash,
                content=content,
                content_length=len(content),
                token_count=token_count,
                start_position=start_position,
                end_position=end_position,
                email_id=email_id,
                attachment_id=attachment_id,
                chunk_type=chunk_type,
                semantic_boundary=semantic_boundary
            )
            
        except Exception as exc:
            logger.error("Failed to create chunk info", exc_info=exc)
            raise
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text (simple approximation)."""
        try:
            # Simple token counting: split by whitespace and count
            words = text.split()
            return len(words)
            
        except Exception:
            # Fallback: approximate by character count
            return len(text) // 4  # Rough approximation: 4 characters per token
    
    def _respects_semantic_boundaries(self, content: str) -> bool:
        """Check if chunk respects semantic boundaries."""
        try:
            # Check if chunk starts and ends at sentence boundaries
            starts_with_capital = content[0].isupper() if content else False
            ends_with_punctuation = content[-1] in '.!?' if content else False
            
            # Check for incomplete sentences
            incomplete_sentence = re.search(r'[.!?]\s*$', content) is None
            
            return starts_with_capital and (ends_with_punctuation or incomplete_sentence)
            
        except Exception:
            return False


class ChunkManager:
    """Manages chunks and their relationships."""
    
    def __init__(self):
        """Initialize chunk manager."""
        self.chunks_by_email = {}  # email_id -> List[ChunkInfo]
        self.chunks_by_attachment = {}  # attachment_id -> List[ChunkInfo]
        self.chunk_index = {}  # chunk_uid -> ChunkInfo
    
    async def add_chunks(self, chunks: List[ChunkInfo]):
        """Add chunks to the manager."""
        try:
            for chunk in chunks:
                # Index by email
                if chunk.email_id not in self.chunks_by_email:
                    self.chunks_by_email[chunk.email_id] = []
                self.chunks_by_email[chunk.email_id].append(chunk)
                
                # Index by attachment if applicable
                if chunk.attachment_id:
                    if chunk.attachment_id not in self.chunks_by_attachment:
                        self.chunks_by_attachment[chunk.attachment_id] = []
                    self.chunks_by_attachment[chunk.attachment_id].append(chunk)
                
                # Index by chunk UID
                self.chunk_index[chunk.chunk_uid] = chunk
            
            logger.info("Chunks added to manager",
                       total_chunks=len(chunks),
                       total_emails=len(self.chunks_by_email))
            
        except Exception as exc:
            logger.error("Failed to add chunks to manager", exc_info=exc)
    
    async def get_chunks_by_email(self, email_id: str) -> List[ChunkInfo]:
        """Get all chunks for a specific email."""
        return self.chunks_by_email.get(email_id, [])
    
    async def get_chunks_by_attachment(self, attachment_id: str) -> List[ChunkInfo]:
        """Get all chunks for a specific attachment."""
        return self.chunks_by_attachment.get(attachment_id, [])
    
    async def get_chunk_by_uid(self, chunk_uid: str) -> Optional[ChunkInfo]:
        """Get a specific chunk by UID."""
        return self.chunk_index.get(chunk_uid)
    
    async def get_chunk_stats(self) -> Dict[str, Any]:
        """Get statistics about chunks."""
        try:
            total_chunks = len(self.chunk_index)
            total_emails = len(self.chunks_by_email)
            total_attachments = len(self.chunks_by_attachment)
            
            # Calculate average chunks per email
            avg_chunks_per_email = total_chunks / total_emails if total_emails > 0 else 0
            
            # Calculate token statistics
            all_tokens = [chunk.token_count for chunk in self.chunk_index.values()]
            avg_tokens_per_chunk = sum(all_tokens) / len(all_tokens) if all_tokens else 0
            max_tokens_per_chunk = max(all_tokens) if all_tokens else 0
            min_tokens_per_chunk = min(all_tokens) if all_tokens else 0
            
            return {
                'total_chunks': total_chunks,
                'total_emails': total_emails,
                'total_attachments': total_attachments,
                'avg_chunks_per_email': avg_chunks_per_email,
                'avg_tokens_per_chunk': avg_tokens_per_chunk,
                'max_tokens_per_chunk': max_tokens_per_chunk,
                'min_tokens_per_chunk': min_tokens_per_chunk
            }
            
        except Exception as exc:
            logger.error("Failed to get chunk stats", exc_info=exc)
            return {}
    
    async def cleanup_chunks(self, email_id: str):
        """Remove chunks for a specific email."""
        try:
            if email_id in self.chunks_by_email:
                chunks = self.chunks_by_email[email_id]
                
                # Remove from all indexes
                for chunk in chunks:
                    if chunk.chunk_uid in self.chunk_index:
                        del self.chunk_index[chunk.chunk_uid]
                    
                    if chunk.attachment_id and chunk.attachment_id in self.chunks_by_attachment:
                        self.chunks_by_attachment[chunk.attachment_id] = [
                            c for c in self.chunks_by_attachment[chunk.attachment_id] 
                            if c.chunk_uid != chunk.chunk_uid
                        ]
                
                # Remove from email index
                del self.chunks_by_email[email_id]
                
                logger.info("Chunks cleaned up for email",
                           email_id=email_id,
                           chunks_removed=len(chunks))
            
        except Exception as exc:
            logger.error("Failed to cleanup chunks for email",
                        email_id=email_id,
                        exc_info=exc)
    
    async def validate_chunks(self, email_id: str) -> Dict[str, Any]:
        """Validate chunks for an email."""
        try:
            chunks = await self.get_chunks_by_email(email_id)
            
            if not chunks:
                return {'valid': False, 'error': 'No chunks found'}
            
            validation_results = []
            total_tokens = 0
            
            for chunk in chunks:
                # Validate chunk structure
                chunk_valid = True
                errors = []
                
                if not chunk.chunk_uid:
                    chunk_valid = False
                    errors.append('Missing chunk UID')
                
                if not chunk.content:
                    chunk_valid = False
                    errors.append('Empty content')
                
                if chunk.token_count <= 0:
                    chunk_valid = False
                    errors.append('Invalid token count')
                
                if chunk.start_position < 0 or chunk.end_position < chunk.start_position:
                    chunk_valid = False
                    errors.append('Invalid position')
                
                validation_results.append({
                    'chunk_uid': chunk.chunk_uid,
                    'valid': chunk_valid,
                    'errors': errors
                })
                
                total_tokens += chunk.token_count
            
            # Overall validation
            all_valid = all(result['valid'] for result in validation_results)
            
            return {
                'valid': all_valid,
                'total_chunks': len(chunks),
                'total_tokens': total_tokens,
                'chunk_validations': validation_results
            }
            
        except Exception as exc:
            logger.error("Failed to validate chunks",
                        email_id=email_id,
                        exc_info=exc)
            return {'valid': False, 'error': str(exc)}
