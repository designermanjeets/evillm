"""Deduplication engine for email ingestion pipeline."""

import asyncio
from typing import Optional, Dict, Any, List
import structlog
import hashlib
import re
from datetime import datetime

from .models import DedupResult

logger = structlog.get_logger(__name__)


class DeduplicationEngine:
    """Engine for detecting duplicate content."""
    
    def __init__(self):
        """Initialize deduplication engine."""
        self.content_hash_index = {}  # content_hash -> metadata
        self.simhash_processor = SimHashProcessor()
    
    def compute_content_hash(self, content: bytes) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content).hexdigest()
    
    async def check_exact_duplicate(self, content_hash: str) -> Optional[str]:
        """Check for exact duplicate content."""
        try:
            if content_hash in self.content_hash_index:
                # Update reference count
                self.content_hash_index[content_hash]['reference_count'] += 1
                self.content_hash_index[content_hash]['last_seen_at'] = datetime.utcnow()
                
                logger.info("Exact duplicate detected",
                           content_hash=content_hash[:8],
                           reference_count=self.content_hash_index[content_hash]['reference_count'])
                
                return content_hash
            else:
                # Add new content to index
                self.content_hash_index[content_hash] = {
                    'first_seen_at': datetime.utcnow(),
                    'last_seen_at': datetime.utcnow(),
                    'reference_count': 1
                }
                
                return None
                
        except Exception as exc:
            logger.error("Failed to check exact duplicate",
                        content_hash=content_hash[:8],
                        exc_info=exc)
            return None
    
    async def check_near_duplicate(self, content: str, threshold: float = 0.85) -> Optional[Dict[str, Any]]:
        """Check for near-duplicate content using SimHash."""
        try:
            if not content or len(content.strip()) < 100:  # Skip very short content
                return None
            
            # Generate SimHash for current content
            current_simhash = self.simhash_processor.generate_simhash(content)
            
            # Check against existing SimHashes
            for existing_hash, metadata in self.content_hash_index.items():
                if 'simhash' in metadata:
                    similarity = self.simhash_processor.calculate_similarity(
                        current_simhash, metadata['simhash']
                    )
                    
                    if similarity >= threshold:
                        # Update reference count
                        metadata['reference_count'] += 1
                        metadata['last_seen_at'] = datetime.utcnow()
                        
                        logger.info("Near duplicate detected",
                                   content_hash=existing_hash[:8],
                                   similarity=similarity,
                                   threshold=threshold,
                                   reference_count=metadata['reference_count'])
                        
                        return {
                            'content_hash': existing_hash,
                            'similarity_score': similarity,
                            'algorithm': 'simhash'
                        }
            
            # Add new content with SimHash
            content_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            self.content_hash_index[content_hash] = {
                'first_seen_at': datetime.utcnow(),
                'last_seen_at': datetime.utcnow(),
                'reference_count': 1,
                'simhash': current_simhash
            }
            
            return None
            
        except Exception as exc:
            logger.error("Failed to check near duplicate", exc_info=exc)
            return None
    
    async def persist_dedup_lineage(self, dedup_result: DedupResult, email_id: str, tenant_id: str):
        """Persist deduplication lineage to database."""
        try:
            from ..database.session import get_db_session
            from ..database.models import DedupLineage
            
            if not dedup_result.is_duplicate:
                return
            
            async with get_db_session() as session:
                # Create dedup lineage record
                lineage = DedupLineage(
                    tenant_id=tenant_id,
                    email_id=email_id,
                    content_hash=dedup_result.content_hash,
                    duplicate_type=dedup_result.duplicate_type,
                    original_content_hash=dedup_result.original_content_hash,
                    similarity_score=getattr(dedup_result, 'similarity_score', None),
                    dedup_algorithm=getattr(dedup_result, 'dedup_algorithm', None),
                    reference_count=1
                )
                
                session.add(lineage)
                await session.commit()
                
                logger.info("Dedup lineage persisted",
                           email_id=email_id,
                           duplicate_type=dedup_result.duplicate_type)
                
        except Exception as exc:
            logger.error("Failed to persist dedup lineage", exc_info=exc)
            # Don't fail the main process for lineage persistence issues
    
    async def get_dedup_stats(self) -> Dict[str, Any]:
        """Get deduplication statistics."""
        try:
            total_content = len(self.content_hash_index)
            duplicate_content = sum(1 for metadata in self.content_hash_index.values() 
                                 if metadata['reference_count'] > 1)
            
            total_references = sum(metadata['reference_count'] 
                                 for metadata in self.content_hash_index.values())
            
            dedup_ratio = (total_references - total_content) / total_references if total_references > 0 else 0
            
            return {
                'total_unique_content': total_content,
                'duplicate_content': duplicate_content,
                'total_references': total_references,
                'dedup_ratio': dedup_ratio,
                'storage_saved_bytes': 0  # Placeholder for actual storage savings calculation
            }
            
        except Exception as exc:
            logger.error("Failed to get dedup stats", exc_info=exc)
            return {}
    
    async def cleanup_old_content(self, max_age_days: int = 30):
        """Clean up old content from the index."""
        try:
            from datetime import datetime, timedelta
            
            cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
            old_content = []
            
            for content_hash, metadata in self.content_hash_index.items():
                if metadata['last_seen_at'] < cutoff_date:
                    old_content.append(content_hash)
            
            for content_hash in old_content:
                del self.content_hash_index[content_hash]
            
            if old_content:
                logger.info("Cleaned up old content",
                           removed_count=len(old_content),
                           max_age_days=max_age_days)
                
        except Exception as exc:
            logger.error("Failed to cleanup old content", exc_info=exc)


class SimHashProcessor:
    """Processor for SimHash-based near-duplicate detection."""
    
    def __init__(self, hash_bits: int = 64):
        """Initialize SimHash processor."""
        self.hash_bits = hash_bits
        self.feature_extractor = FeatureExtractor()
    
    def generate_simhash(self, content: str) -> int:
        """Generate SimHash for content."""
        try:
            # Extract features (words)
            features = self.feature_extractor.extract_features(content)
            
            if not features:
                return 0
            
            # Initialize hash vector
            hash_vector = [0] * self.hash_bits
            
            # Process each feature
            for feature in features:
                # Generate hash for feature
                feature_hash = self._hash_feature(feature)
                
                # Update hash vector
                for i in range(self.hash_bits):
                    if feature_hash & (1 << i):
                        hash_vector[i] += 1
                    else:
                        hash_vector[i] -= 1
            
            # Convert to SimHash
            simhash = 0
            for i in range(self.hash_bits):
                if hash_vector[i] > 0:
                    simhash |= (1 << i)
            
            return simhash
            
        except Exception as exc:
            logger.error("Failed to generate SimHash", exc_info=exc)
            return 0
    
    def compute_simhash(self, content: str) -> int:
        """Alias for generate_simhash for backward compatibility."""
        return self.generate_simhash(content)
    
    def calculate_similarity(self, simhash1: int, simhash2: int) -> float:
        """Calculate similarity between two SimHashes."""
        try:
            # Calculate Hamming distance
            hamming_distance = bin(simhash1 ^ simhash2).count('1')
            
            # Convert to similarity score (0.0 to 1.0)
            similarity = 1.0 - (hamming_distance / self.hash_bits)
            
            return max(0.0, min(1.0, similarity))
            
        except Exception as exc:
            logger.error("Failed to calculate SimHash similarity", exc_info=exc)
            return 0.0
    
    def _hash_feature(self, feature: str) -> int:
        """Generate hash for a feature."""
        try:
            # Use built-in hash function and convert to positive integer
            hash_value = hash(feature)
            return abs(hash_value) % (1 << self.hash_bits)
            
        except Exception:
            # Fallback to simple hash
            return hash(feature.encode('utf-8')) % (1 << self.hash_bits)


class FeatureExtractor:
    """Extract features from text for SimHash processing."""
    
    def __init__(self):
        """Initialize feature extractor."""
        self.stop_words = self._load_stop_words()
    
    def extract_features(self, content: str) -> List[str]:
        """Extract features from text content."""
        try:
            if not content:
                return []
            
            # Clean and normalize text
            cleaned_text = self._clean_text(content)
            
            # Extract words
            words = re.findall(r'\b\w+\b', cleaned_text.lower())
            
            # Filter out stop words and short words
            features = [word for word in words 
                       if word not in self.stop_words and len(word) > 2]
            
            # Limit features to prevent excessive processing
            return features[:1000]
            
        except Exception as exc:
            logger.error("Failed to extract features", exc_info=exc)
            return []
    
    def _clean_text(self, text: str) -> str:
        """Clean text for feature extraction."""
        try:
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', text)
            
            # Remove special characters but keep alphanumeric and spaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()
            
        except Exception:
            return text
    
    def _load_stop_words(self) -> set:
        """Load common stop words."""
        # Common English stop words
        stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'she', 'do', 'how', 'their',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'time', 'two', 'more',
            'go', 'no', 'way', 'could', 'my', 'than', 'first', 'been', 'call',
            'who', 'its', 'now', 'find', 'long', 'down', 'day', 'did', 'get',
            'come', 'made', 'may', 'part', 'over', 'new', 'sound', 'take', 'only',
            'little', 'work', 'know', 'place', 'year', 'live', 'me', 'back', 'give',
            'most', 'very', 'after', 'thing', 'our', 'just', 'name', 'good', 'sentence',
            'man', 'think', 'say', 'great', 'where', 'help', 'through', 'much',
            'before', 'line', 'right', 'too', 'mean', 'old', 'any', 'same', 'tell',
            'boy', 'follow', 'came', 'want', 'show', 'also', 'around', 'form', 'three',
            'small', 'set', 'put', 'end', 'does', 'another', 'well', 'large', 'must',
            'big', 'even', 'such', 'because', 'turn', 'here', 'why', 'ask', 'went',
            'men', 'read', 'need', 'land', 'different', 'home', 'us', 'move', 'try',
            'kind', 'hand', 'picture', 'again', 'change', 'off', 'play', 'spell',
            'air', 'away', 'animal', 'house', 'point', 'page', 'letter', 'mother',
            'answer', 'found', 'study', 'still', 'learn', 'should', 'America', 'world'
        }
        
        return stop_words


class MinHashProcessor:
    """Processor for MinHash-based near-duplicate detection."""
    
    def __init__(self, num_permutations: int = 128):
        """Initialize MinHash processor."""
        self.num_permutations = num_permutations
        self.feature_extractor = FeatureExtractor()
        self.permutations = self._generate_permutations()
    
    def generate_minhash(self, content: str) -> List[int]:
        """Generate MinHash for content."""
        try:
            # Extract features
            features = self.feature_extractor.extract_features(content)
            
            if not features:
                return [float('inf')] * self.num_permutations
            
            # Initialize MinHash
            minhash = [float('inf')] * self.num_permutations
            
            # Process each feature
            for feature in features:
                feature_hash = hash(feature)
                
                for i in range(self.num_permutations):
                    # Apply permutation
                    permuted_hash = (feature_hash * self.permutations[i]['a'] + 
                                   self.permutations[i]['b']) % self.permutations[i]['prime']
                    
                    # Update MinHash
                    if permuted_hash < minhash[i]:
                        minhash[i] = permuted_hash
            
            return minhash
            
        except Exception as exc:
            logger.error("Failed to generate MinHash", exc_info=exc)
            return [float('inf')] * self.num_permutations
    
    def calculate_similarity(self, minhash1: List[int], minhash2: List[int]) -> float:
        """Calculate similarity between two MinHashes."""
        try:
            if len(minhash1) != len(minhash2):
                return 0.0
            
            # Count matching positions
            matches = sum(1 for i in range(len(minhash1)) 
                         if minhash1[i] == minhash2[i])
            
            # Calculate Jaccard similarity estimate
            similarity = matches / len(minhash1)
            
            return similarity
            
        except Exception as exc:
            logger.error("Failed to calculate MinHash similarity", exc_info=exc)
            return 0.0
    
    def _generate_permutations(self) -> List[Dict[str, int]]:
        """Generate random permutations for MinHash."""
        import random
        
        permutations = []
        for _ in range(self.num_permutations):
            # Generate random coefficients for linear hash function
            a = random.randint(1, 1000000)
            b = random.randint(0, 1000000)
            prime = 1000000007  # Large prime number
            
            permutations.append({'a': a, 'b': b, 'prime': prime})
        
        return permutations
