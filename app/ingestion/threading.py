"""Threading manager for email ingestion pipeline."""

import asyncio
from typing import Optional, Dict, Any, List
import structlog
import re
from datetime import datetime

from .models import EmailMetadata, ThreadInfo, ProcessingStatus

logger = structlog.get_logger(__name__)


class ThreadManager:
    """Manages email threading relationships."""
    
    def __init__(self):
        """Initialize thread manager."""
        self.threads = {}  # thread_id -> ThreadInfo
        self.email_thread_map = {}  # email_id -> thread_id
        self.message_id_thread_map = {}  # message_id -> thread_id
        self.subject_thread_map = {}  # normalized_subject -> thread_id
    
    async def process_email_threading(self, metadata: EmailMetadata, email_id: str) -> ThreadInfo:
        """Process threading for an email."""
        try:
            # Try to find existing thread
            thread_info = await self._find_existing_thread(metadata)
            
            if thread_info:
                # Update existing thread
                await self._update_thread(thread_info, metadata, email_id)
                return thread_info
            else:
                # Create new thread
                thread_info = await self._create_new_thread(metadata, email_id)
                return thread_info
                
        except Exception as exc:
            logger.error("Failed to process email threading",
                        email_id=email_id,
                        message_id=metadata.message_id,
                        exc_info=exc)
            # Create a fallback thread
            return await self._create_fallback_thread(metadata, email_id)
    
    async def _find_existing_thread(self, metadata: EmailMetadata) -> Optional[ThreadInfo]:
        """Find existing thread for an email."""
        try:
            # Method 1: Check Message-ID threading
            if metadata.message_id:
                thread_id = self.message_id_thread_map.get(metadata.message_id)
                if thread_id and thread_id in self.threads:
                    return self.threads[thread_id]
            
            # Method 2: Check In-Reply-To threading
            if metadata.in_reply_to:
                thread_id = self.message_id_thread_map.get(metadata.in_reply_to)
                if thread_id and thread_id in self.threads:
                    return self.threads[thread_id]
            
            # Method 3: Check References threading
            if metadata.references:
                for ref in metadata.references.split():
                    ref = ref.strip()
                    if ref:
                        thread_id = self.message_id_thread_map.get(ref)
                        if thread_id and thread_id in self.threads:
                            return self.threads[thread_id]
            
            # Method 4: Check subject similarity
            if metadata.subject:
                normalized_subject = self._normalize_subject(metadata.subject)
                thread_id = self.subject_thread_map.get(normalized_subject)
                if thread_id and thread_id in self.threads:
                    return self.threads[thread_id]
            
            return None
            
        except Exception as exc:
            logger.error("Failed to find existing thread", exc_info=exc)
            return None
    
    async def _create_new_thread(self, metadata: EmailMetadata, email_id: str) -> ThreadInfo:
        """Create a new thread for an email."""
        try:
            # Generate thread ID
            thread_id = f"thread_{hash(metadata.message_id)}_{hash(metadata.subject)}"
            
            # Normalize subject
            normalized_subject = self._normalize_subject(metadata.subject)
            
            # Create thread info
            thread_info = ThreadInfo(
                thread_id=thread_id,
                thread_subject=metadata.subject,
                thread_subject_normalized=normalized_subject,
                email_count=1,
                first_email_date=metadata.date,
                last_email_date=metadata.date,
                root_message_id=metadata.message_id,
                thread_depth=1,
                thread_breadth=1,
                tenant_id=metadata.tenant_id
            )
            
            # Store thread
            self.threads[thread_id] = thread_info
            
            # Update mappings
            self.email_thread_map[email_id] = thread_id
            if metadata.message_id:
                self.message_id_thread_map[metadata.message_id] = thread_id
            self.subject_thread_map[normalized_subject] = thread_id
            
            logger.info("New thread created",
                       thread_id=thread_id,
                       subject=metadata.subject,
                       email_id=email_id)
            
            return thread_info
            
        except Exception as exc:
            logger.error("Failed to create new thread", exc_info=exc)
            raise
    
    async def _update_thread(self, thread_info: ThreadInfo, metadata: EmailMetadata, email_id: str):
        """Update existing thread with new email."""
        try:
            # Update thread metadata
            thread_info.email_count += 1
            thread_info.last_email_date = metadata.date
            
            # Update thread depth and breadth
            if metadata.in_reply_to:
                thread_info.thread_depth = max(thread_info.thread_depth, 2)
            else:
                thread_info.thread_depth = max(thread_info.thread_depth, 1)
            
            thread_info.thread_breadth = max(thread_info.thread_breadth, thread_info.email_count)
            
            # Update mappings
            self.email_thread_map[email_id] = thread_info.thread_id
            if metadata.message_id:
                self.message_id_thread_map[metadata.message_id] = thread_info.thread_id
            
            logger.info("Thread updated",
                       thread_id=thread_info.thread_id,
                       email_id=email_id,
                       email_count=thread_info.email_count)
            
        except Exception as exc:
            logger.error("Failed to update thread", exc_info=exc)
            raise
    
    async def _create_fallback_thread(self, metadata: EmailMetadata, email_id: str) -> ThreadInfo:
        """Create a fallback thread when threading fails."""
        try:
            # Generate fallback thread ID
            thread_id = f"fallback_thread_{hash(email_id)}"
            
            # Normalize subject
            normalized_subject = self._normalize_subject(metadata.subject) if metadata.subject else "No Subject"
            
            # Create fallback thread info
            thread_info = ThreadInfo(
                thread_id=thread_id,
                thread_subject=metadata.subject or "No Subject",
                thread_subject_normalized=normalized_subject,
                email_count=1,
                first_email_date=metadata.date,
                last_email_date=metadata.date,
                root_message_id=metadata.message_id,
                thread_depth=1,
                thread_breadth=1,
                tenant_id=metadata.tenant_id
            )
            
            # Store thread
            self.threads[thread_id] = thread_info
            
            # Update mappings
            self.email_thread_map[email_id] = thread_id
            if metadata.message_id:
                self.message_id_thread_map[metadata.message_id] = thread_id
            self.subject_thread_map[normalized_subject] = thread_id
            
            logger.warning("Fallback thread created",
                          thread_id=thread_id,
                          email_id=email_id,
                          reason="Threading processing failed")
            
            return thread_info
            
        except Exception as exc:
            logger.error("Failed to create fallback thread", exc_info=exc)
            raise
    
    def _normalize_subject(self, subject: str) -> str:
        """Normalize email subject for threading."""
        try:
            if not subject:
                return "no_subject"
            
            # Convert to lowercase
            normalized = subject.lower()
            
            # Remove common prefixes
            prefixes_to_remove = [
                're:', 're :', 're : ', 're: ',
                'fw:', 'fw :', 'fw : ', 'fw: ',
                'fwd:', 'fwd :', 'fwd : ', 'fwd: ',
                'fyi:', 'fyi :', 'fyi : ', 'fyi: '
            ]
            
            for prefix in prefixes_to_remove:
                if normalized.startswith(prefix):
                    normalized = normalized[len(prefix):].strip()
                    break
            
            # Remove extra whitespace
            normalized = re.sub(r'\s+', ' ', normalized)
            
            # Remove special characters
            normalized = re.sub(r'[^\w\s]', '', normalized)
            
            # Trim and return
            return normalized.strip() or "no_subject"
            
        except Exception:
            return "no_subject"
    
    async def get_thread_info(self, thread_id: str) -> Optional[ThreadInfo]:
        """Get thread information by ID."""
        return self.threads.get(thread_id)
    
    async def get_thread_by_email(self, email_id: str) -> Optional[ThreadInfo]:
        """Get thread information for an email."""
        try:
            thread_id = self.email_thread_map.get(email_id)
            if thread_id:
                return self.threads.get(thread_id)
            return None
            
        except Exception as exc:
            logger.error("Failed to get thread by email",
                        email_id=email_id,
                        exc_info=exc)
            return None
    
    async def get_threads_by_subject(self, subject: str) -> List[ThreadInfo]:
        """Get threads by subject similarity."""
        try:
            normalized_subject = self._normalize_subject(subject)
            matching_threads = []
            
            for thread_info in self.threads.values():
                if thread_info.thread_subject_normalized == normalized_subject:
                    matching_threads.append(thread_info)
            
            return matching_threads
            
        except Exception as exc:
            logger.error("Failed to get threads by subject",
                        subject=subject,
                        exc_info=exc)
            return []
    
    async def get_thread_stats(self) -> Dict[str, Any]:
        """Get threading statistics."""
        try:
            total_threads = len(self.threads)
            total_emails = sum(thread.email_count for thread in self.threads.values())
            
            # Calculate thread depth statistics
            depths = [thread.thread_depth for thread in self.threads.values()]
            avg_depth = sum(depths) / len(depths) if depths else 0
            max_depth = max(depths) if depths else 0
            
            # Calculate thread breadth statistics
            breadths = [thread.thread_breadth for thread in self.threads.values()]
            avg_breadth = sum(breadths) / len(breadths) if breadths else 0
            max_breadth = max(breadths) if breadths else 0
            
            # Calculate emails per thread
            avg_emails_per_thread = total_emails / total_threads if total_threads > 0 else 0
            
            return {
                'total_threads': total_threads,
                'total_emails': total_emails,
                'avg_emails_per_thread': avg_emails_per_thread,
                'avg_thread_depth': avg_depth,
                'max_thread_depth': max_depth,
                'avg_thread_breadth': avg_breadth,
                'max_thread_breadth': max_breadth
            }
            
        except Exception as exc:
            logger.error("Failed to get thread stats", exc_info=exc)
            return {}
    
    async def cleanup_thread(self, thread_id: str):
        """Remove a thread and its mappings."""
        try:
            if thread_id in self.threads:
                thread_info = self.threads[thread_id]
                
                # Remove from mappings
                for email_id, tid in list(self.email_thread_map.items()):
                    if tid == thread_id:
                        del self.email_thread_map[email_id]
                
                for message_id, tid in list(self.message_id_thread_map.items()):
                    if tid == thread_id:
                        del self.message_id_thread_map[message_id]
                
                for subject, tid in list(self.subject_thread_map.items()):
                    if tid == thread_id:
                        del self.subject_thread_map[subject]
                
                # Remove thread
                del self.threads[thread_id]
                
                logger.info("Thread cleaned up",
                           thread_id=thread_id,
                           subject=thread_info.thread_subject)
            
        except Exception as exc:
            logger.error("Failed to cleanup thread",
                        thread_id=thread_id,
                        exc_info=exc)


class ThreadLinker:
    """Utility class for linking emails to threads."""
    
    @staticmethod
    def extract_threading_headers(metadata: EmailMetadata) -> Dict[str, Any]:
        """Extract threading information from email headers."""
        try:
            threading_info = {
                'message_id': metadata.message_id,
                'in_reply_to': metadata.in_reply_to,
                'references': metadata.references,
                'subject': metadata.subject,
                'date': metadata.date
            }
            
            # Parse references into list
            if metadata.references:
                references_list = [ref.strip() for ref in metadata.references.split() if ref.strip()]
                threading_info['references_list'] = references_list
            else:
                threading_info['references_list'] = []
            
            return threading_info
            
        except Exception as exc:
            logger.error("Failed to extract threading headers", exc_info=exc)
            return {}
    
    @staticmethod
    def calculate_thread_position(threading_info: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate the position of an email in a thread."""
        try:
            position_info = {
                'is_reply': False,
                'is_forward': False,
                'thread_level': 0,
                'has_children': False
            }
            
            # Check if this is a reply
            if threading_info.get('in_reply_to'):
                position_info['is_reply'] = True
                position_info['thread_level'] = 1
            
            # Check if this has references (part of a longer thread)
            if threading_info.get('references_list'):
                position_info['thread_level'] = len(threading_info['references_list'])
            
            # Check if this might be a forward (subject starts with Fwd:)
            subject = threading_info.get('subject', '').lower()
            if subject.startswith(('fwd:', 'fw:', 'forward:')):
                position_info['is_forward'] = True
            
            return position_info
            
        except Exception as exc:
            logger.error("Failed to calculate thread position", exc_info=exc)
            return {'is_reply': False, 'is_forward': False, 'thread_level': 0, 'has_children': False}
    
    @staticmethod
    def validate_threading_consistency(threading_info: Dict[str, Any]) -> Dict[str, Any]:
        """Validate threading information for consistency."""
        try:
            validation = {
                'is_consistent': True,
                'warnings': [],
                'errors': []
            }
            
            # Check for missing Message-ID
            if not threading_info.get('message_id'):
                validation['warnings'].append('Missing Message-ID')
            
            # Check for inconsistent references
            if threading_info.get('in_reply_to') and threading_info.get('references_list'):
                if threading_info['in_reply_to'] not in threading_info['references_list']:
                    validation['warnings'].append('In-Reply-To not in References')
            
            # Check for circular references
            message_id = threading_info.get('message_id')
            if message_id and message_id in threading_info.get('references_list', []):
                validation['errors'].append('Circular reference detected')
                validation['is_consistent'] = False
            
            # Check for future dates
            if threading_info.get('date'):
                if threading_info['date'] > datetime.utcnow():
                    validation['warnings'].append('Future date detected')
            
            return validation
            
        except Exception as exc:
            logger.error("Failed to validate threading consistency", exc_info=exc)
            return {'is_consistent': False, 'warnings': [], 'errors': ['Validation failed']}
