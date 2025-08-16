"""MIME parser for email ingestion pipeline."""

import email
from email import policy
from email.parser import BytesParser
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.image import MIMEImage
from email.mime.application import MIMEApplication
from typing import List, Optional, Dict, Any, Tuple
import structlog
import re
from datetime import datetime

from .models import EmailMetadata, AttachmentInfo, ErrorType

logger = structlog.get_logger(__name__)


class MIMEParser:
    """Parser for MIME email messages."""
    
    def __init__(self):
        """Initialize MIME parser."""
        self.policy = policy.default
        self.parser = BytesParser(policy=self.policy)
    
    def parse_email(self, raw_content: bytes, tenant_id: str, batch_id: str) -> Tuple[EmailMetadata, List[AttachmentInfo], bytes]:
        """Parse raw email content and extract metadata and attachments."""
        try:
            # Parse the email message
            message = self.parser.parsebytes(raw_content)
            
            # Extract email metadata
            metadata = self._extract_metadata(message, tenant_id, batch_id)
            
            # Extract attachments
            attachments = self._extract_attachments(message)
            
            # Get the main email body
            body_content = self._extract_body_content(message)
            
            return metadata, attachments, body_content
            
        except Exception as exc:
            logger.error("Failed to parse email", exc_info=exc)
            raise ValueError(f"Failed to parse MIME email: {str(exc)}")
    
    def _extract_metadata(self, message: email.message.Message, tenant_id: str, batch_id: str) -> EmailMetadata:
        """Extract metadata from email message."""
        try:
            # Basic email information
            message_id = message.get('Message-ID', '')
            subject = message.get('Subject', '')
            from_address = message.get('From', '')
            to_addresses = self._parse_address_list(message.get('To', ''))
            cc_addresses = self._parse_address_list(message.get('CC', ''))
            bcc_addresses = self._parse_address_list(message.get('BCC', ''))
            
            # Date parsing
            date_str = message.get('Date', '')
            date = self._parse_date(date_str)
            
            # Threading headers
            in_reply_to = message.get('In-Reply-To', '')
            references = message.get('References', '')
            
            # Content type and charset
            content_type = message.get_content_type()
            charset = message.get_content_charset()
            
            # Generate message ID if missing
            if not message_id:
                message_id = self._generate_message_id(from_address, date, subject)
            
            return EmailMetadata(
                message_id=message_id,
                subject=subject,
                from_address=from_address,
                to_addresses=to_addresses,
                cc_addresses=cc_addresses,
                bcc_addresses=bcc_addresses,
                date=date,
                in_reply_to=in_reply_to if in_reply_to else None,
                references=references if references else None,
                content_type=content_type,
                charset=charset,
                tenant_id=tenant_id,
                batch_id=batch_id
            )
            
        except Exception as exc:
            logger.error("Failed to extract email metadata", exc_info=exc)
            raise ValueError(f"Failed to extract email metadata: {str(exc)}")
    
    def _extract_attachments(self, message: email.message.Message) -> List[AttachmentInfo]:
        """Extract attachments from email message."""
        attachments = []
        
        try:
            for part in message.walk():
                if part.is_multipart():
                    continue
                
                # Check if this part is an attachment
                if self._is_attachment(part):
                    attachment = self._create_attachment_info(part)
                    if attachment:
                        attachments.append(attachment)
            
            return attachments
            
        except Exception as exc:
            logger.error("Failed to extract attachments", exc_info=exc)
            # Return empty list on attachment extraction failure
            return []
    
    def _extract_body_content(self, message: email.message.Message) -> bytes:
        """Extract the main email body content."""
        try:
            # Find the main text content
            if message.is_multipart():
                # Look for text/plain first, then text/html
                for part in message.walk():
                    if part.get_content_maintype() == 'text':
                        if part.get_content_subtype() == 'plain':
                            return part.get_content()
                        elif part.get_content_subtype() == 'html':
                            return part.get_content()
                
                # If no text parts found, return the first part
                for part in message.walk():
                    if not part.is_multipart():
                        return part.get_content()
            else:
                # Single part message
                return message.get_content()
            
            # Fallback: return empty content
            return b""
            
        except Exception as exc:
            logger.error("Failed to extract body content", exc_info=exc)
            return b""
    
    def _is_attachment(self, part: email.message.Message) -> bool:
        """Check if a message part is an attachment."""
        # Check content disposition
        content_disposition = part.get('Content-Disposition', '')
        if content_disposition and 'attachment' in content_disposition.lower():
            return True
        
        # Check if it's not the main text content
        content_type = part.get_content_type()
        if content_type in ['text/plain', 'text/html']:
            # Check if it's the main body or an attachment
            if part.get_filename():
                return True
        
        # Check for specific attachment indicators
        if part.get_filename() or part.get('Content-ID'):
            return True
        
        return False
    
    def _create_attachment_info(self, part: email.message.Message) -> Optional[AttachmentInfo]:
        """Create attachment info from message part."""
        try:
            filename = part.get_filename()
            if not filename:
                # Generate filename from content type
                content_type = part.get_content_type()
                extension = self._get_extension_from_mimetype(content_type)
                filename = f"attachment_{hash(part.get_content())}{extension}"
            
            content_type = part.get_content_type()
            content_disposition = part.get('Content-Disposition', '')
            content_id = part.get('Content-ID', '')
            
            # Get attachment content
            content = part.get_content()
            if isinstance(content, str):
                content = content.encode('utf-8')
            
            # Calculate content hash
            content_hash = self._calculate_content_hash(content)
            content_length = len(content)
            
            # Determine if OCR is required
            ocr_required = self._requires_ocr(content_type, content)
            
            return AttachmentInfo(
                filename=filename,
                content_type=content_type,
                content_disposition=content_disposition,
                content_id=content_id if content_id else None,
                content=content,
                content_hash=content_hash,
                content_length=content_length,
                ocr_required=ocr_required
            )
            
        except Exception as exc:
            logger.error("Failed to create attachment info", exc_info=exc)
            return None
    
    def _parse_address_list(self, address_string: str) -> List[str]:
        """Parse email address list from string."""
        if not address_string:
            return []
        
        try:
            # Simple parsing - split by comma and clean up
            addresses = []
            for addr in address_string.split(','):
                addr = addr.strip()
                if addr:
                    # Extract email address from "Name <email@domain.com>" format
                    email_match = re.search(r'<(.+?)>', addr)
                    if email_match:
                        addresses.append(email_match.group(1))
                    else:
                        addresses.append(addr)
            
            return addresses
            
        except Exception as exc:
            logger.warning("Failed to parse address list", address_string=address_string, exc_info=exc)
            return []
    
    def _parse_date(self, date_string: str) -> datetime:
        """Parse email date string."""
        if not date_string:
            return datetime.utcnow()
        
        try:
            # Try to parse the date string
            parsed_date = email.utils.parsedate_to_datetime(date_string)
            if parsed_date:
                return parsed_date
        except Exception:
            pass
        
        try:
            # Fallback: try different date formats
            import dateutil.parser
            return dateutil.parser.parse(date_string)
        except Exception:
            pass
        
        # Final fallback: return current time
        logger.warning("Failed to parse email date", date_string=date_string)
        return datetime.utcnow()
    
    def _generate_message_id(self, from_address: str, date: datetime, subject: str) -> str:
        """Generate a message ID if one is missing."""
        import hashlib
        
        # Create a hash from sender, date, and subject
        content = f"{from_address}{date.isoformat()}{subject}"
        hash_value = hashlib.md5(content.encode()).hexdigest()
        
        # Generate a domain from the sender
        domain = "unknown.domain"
        if '@' in from_address:
            domain = from_address.split('@')[-1]
        
        return f"<{hash_value}@{domain}>"
    
    def _get_extension_from_mimetype(self, mimetype: str) -> str:
        """Get file extension from MIME type."""
        extension_map = {
            'text/plain': '.txt',
            'text/html': '.html',
            'application/pdf': '.pdf',
            'application/msword': '.doc',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'application/vnd.ms-excel': '.xls',
            'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': '.xlsx',
            'image/jpeg': '.jpg',
            'image/png': '.png',
            'image/gif': '.gif',
            'image/bmp': '.bmp',
            'image/tiff': '.tiff'
        }
        
        return extension_map.get(mimetype, '.bin')
    
    def _calculate_content_hash(self, content: bytes) -> str:
        """Calculate SHA-256 hash of content."""
        import hashlib
        return hashlib.sha256(content).hexdigest()
    
    def _requires_ocr(self, content_type: str, content: bytes) -> bool:
        """Determine if content requires OCR processing."""
        # Check if it's an image
        if content_type.startswith('image/'):
            return True
        
        # Check if it's a PDF that might need OCR
        if content_type == 'application/pdf':
            # Simple heuristic: if content is mostly binary, it might need OCR
            text_ratio = self._calculate_text_ratio(content)
            return text_ratio < 0.1  # Less than 10% text content
        
        return False
    
    def _calculate_text_ratio(self, content: bytes) -> float:
        """Calculate the ratio of text content in binary data."""
        try:
            # Count printable ASCII characters
            printable_count = sum(1 for byte in content if 32 <= byte <= 126)
            total_count = len(content)
            
            if total_count == 0:
                return 0.0
            
            return printable_count / total_count
            
        except Exception:
            return 0.0


class EmailParser:
    """High-level email parser that coordinates the parsing process."""
    
    def __init__(self):
        """Initialize email parser."""
        self.mime_parser = MIMEParser()
    
    async def parse_email_file(self, file_path: str, tenant_id: str, batch_id: str) -> Tuple[EmailMetadata, List[AttachmentInfo], bytes]:
        """Parse email from file."""
        try:
            with open(file_path, 'rb') as f:
                raw_content = f.read()
            
            return await self.parse_email_content(raw_content, tenant_id, batch_id)
            
        except Exception as exc:
            logger.error("Failed to parse email file", file_path=file_path, exc_info=exc)
            raise
    
    async def parse_email_content(self, raw_content: bytes, tenant_id: str, batch_id: str) -> Tuple[EmailMetadata, List[AttachmentInfo], bytes]:
        """Parse email from raw content."""
        try:
            # Validate content size
            if len(raw_content) > 50 * 1024 * 1024:  # 50MB limit
                raise ValueError("Email content exceeds maximum size limit")
            
            # Parse the email
            metadata, attachments, body_content = self.mime_parser.parse_email(
                raw_content, tenant_id, batch_id
            )
            
            logger.info("Email parsed successfully",
                       message_id=metadata.message_id,
                       subject=metadata.subject,
                       attachment_count=len(attachments))
            
            return metadata, attachments, body_content
            
        except Exception as exc:
            logger.error("Failed to parse email content", exc_info=exc)
            raise
