"""OCR service for text extraction from images and documents."""

import asyncio
import hashlib
import time
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class OCRStatus(Enum):
    """OCR processing status."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    QUARANTINED = "quarantined"


class OCRBackendType(Enum):
    """OCR backend types."""
    LOCAL = "local"
    TESSERACT = "tesseract"
    AWS_TEXTRACT = "aws_textract"
    GOOGLE_VISION = "google_vision"
    AZURE_VISION = "azure_vision"


@dataclass
class OCRResult:
    """OCR processing result."""
    success: bool
    text: str
    confidence: float
    processing_time: float
    backend: str
    language_detected: Optional[str] = None
    pages_processed: int = 1
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class OCRTask:
    """OCR processing task."""
    task_id: str
    attachment_id: str
    tenant_id: str
    content_hash: str
    mimetype: str
    file_size: int
    language_hint: Optional[str] = None
    status: OCRStatus = OCRStatus.PENDING
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Optional[OCRResult] = None
    retry_count: int = 0
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


class OCRProvider(ABC):
    """Abstract OCR provider interface."""
    
    @abstractmethod
    async def extract_text(self, 
                          image_bytes: bytes, 
                          mimetype: str, 
                          language_hint: Optional[str] = None,
                          timeout: int = 30) -> OCRResult:
        """Extract text from image/document bytes."""
        pass
    
    @abstractmethod
    def supports_mimetype(self, mimetype: str) -> bool:
        """Check if provider supports the given mimetype."""
        pass
    
    @abstractmethod
    def get_backend_type(self) -> OCRBackendType:
        """Get the backend type."""
        pass


class StubOCRProvider(OCRProvider):
    """Stub OCR provider for development and testing."""
    
    def __init__(self):
        """Initialize stub provider."""
        self.backend_type = OCRBackendType.LOCAL
    
    async def extract_text(self, 
                          image_bytes: bytes, 
                          mimetype: str, 
                          language_hint: Optional[str] = None,
                          timeout: int = 30) -> OCRResult:
        """Simulate OCR text extraction."""
        start_time = time.time()
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        processing_time = time.time() - start_time
        
        # Generate stub text based on mimetype
        if mimetype.startswith("image/"):
            stub_text = "This is sample OCR text extracted from an image.\nIt contains multiple lines of text.\nThe OCR engine has processed this content successfully."
        elif mimetype == "application/pdf":
            stub_text = "Sample PDF document content extracted via OCR.\nThis text was not available in the native PDF format.\nOCR processing has made it searchable."
        else:
            stub_text = "Sample OCR text from document.\nContent extracted using optical character recognition.\nReady for indexing and search."
        
        return OCRResult(
            success=True,
            text=stub_text,
            confidence=0.85,
            processing_time=processing_time,
            backend="stub",
            language_detected="en",
            pages_processed=1,
            metadata={"stub": True, "content_hash": hashlib.sha256(image_bytes).hexdigest()[:8]}
        )
    
    def supports_mimetype(self, mimetype: str) -> bool:
        """Stub provider supports all mimetypes."""
        return True
    
    def get_backend_type(self) -> OCRBackendType:
        """Get backend type."""
        return self.backend_type


class TesseractOCRProvider(OCRProvider):
    """Tesseract OCR provider for local image processing."""
    
    def __init__(self):
        """Initialize Tesseract provider."""
        self.backend_type = OCRBackendType.TESSERACT
        self._check_tesseract_available()
    
    def _check_tesseract_available(self):
        """Check if Tesseract is available."""
        try:
            import pytesseract
            self.tesseract = pytesseract
            logger.info("Tesseract OCR provider initialized successfully")
        except ImportError:
            logger.warning("Tesseract not available, falling back to stub provider")
            raise ImportError("pytesseract not installed")
    
    async def extract_text(self, 
                          image_bytes: bytes, 
                          mimetype: str, 
                          language_hint: Optional[str] = None,
                          timeout: int = 30) -> OCRResult:
        """Extract text using Tesseract OCR."""
        start_time = time.time()
        
        try:
            # Convert bytes to PIL Image
            from PIL import Image
            import io
            
            image = Image.open(io.BytesIO(image_bytes))
            
            # Preprocess image if needed
            image = await self._preprocess_image(image)
            
            # Configure Tesseract
            config = '--oem 3 --psm 6'  # Default OCR Engine Mode, Assume uniform block of text
            if language_hint:
                config += f' -l {language_hint}'
            
            # Extract text
            text = self.tesseract.image_to_string(image, config=config)
            
            processing_time = time.time() - start_time
            
            return OCRResult(
                success=True,
                text=text.strip(),
                confidence=0.8,  # Tesseract doesn't provide confidence scores
                processing_time=processing_time,
                backend="tesseract",
                language_detected=language_hint or "en",
                pages_processed=1,
                metadata={"tesseract_config": config}
            )
            
        except Exception as exc:
            processing_time = time.time() - start_time
            logger.error("Tesseract OCR failed", exc_info=exc)
            
            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                processing_time=processing_time,
                backend="tesseract",
                error_message=str(exc)
            )
    
    async def _preprocess_image(self, image):
        """Preprocess image for better OCR results."""
        try:
            # Convert to grayscale if needed
            if image.mode != 'L':
                image = image.convert('L')
            
            # Apply basic preprocessing (can be enhanced)
            import cv2
            import numpy as np
            
            # Convert PIL to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_GRAY2BGR)
            
            # Apply Gaussian blur to reduce noise
            cv_image = cv2.GaussianBlur(cv_image, (1, 1), 0)
            
            # Apply adaptive thresholding
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Convert back to PIL
            return Image.fromarray(thresh)
            
        except Exception as exc:
            logger.warning("Image preprocessing failed, using original", exc_info=exc)
            return image
    
    def supports_mimetype(self, mimetype: str) -> bool:
        """Tesseract supports image mimetypes."""
        return mimetype.startswith("image/")
    
    def get_backend_type(self) -> OCRBackendType:
        """Get backend type."""
        return self.backend_type


class OCRServiceRegistry:
    """Registry for OCR service providers."""
    
    def __init__(self):
        """Initialize provider registry."""
        self.providers: Dict[str, OCRProvider] = {}
        self.default_provider: Optional[OCRProvider] = None
    
    def register_provider(self, name: str, provider: OCRProvider):
        """Register an OCR provider."""
        self.providers[name] = provider
        logger.info("OCR provider registered", name=name, backend_type=provider.get_backend_type().value)
    
    def set_default_provider(self, name: str):
        """Set the default OCR provider."""
        if name in self.providers:
            self.default_provider = self.providers[name]
            logger.info("Default OCR provider set", name=name)
        else:
            logger.error("Provider not found", name=name)
    
    def get_provider(self, name: Optional[str] = None) -> OCRProvider:
        """Get OCR provider by name or default."""
        if name and name in self.providers:
            return self.providers[name]
        
        if self.default_provider:
            return self.default_provider
        
        # Fallback to first available provider
        if self.providers:
            first_provider = list(self.providers.values())[0]
            logger.warning("No default provider set, using first available", provider=first_provider.__class__.__name__)
            return first_provider
        
        raise RuntimeError("No OCR providers available")


class OCRService:
    """Main OCR service for text extraction."""
    
    def __init__(self, registry: Optional[OCRServiceRegistry] = None):
        """Initialize OCR service."""
        self.registry = registry or OCRServiceRegistry()
        self._setup_default_providers()
    
    def _setup_default_providers(self):
        """Setup default OCR providers."""
        # Register stub provider
        stub_provider = StubOCRProvider()
        self.registry.register_provider("stub", stub_provider)
        
        # Try to register Tesseract provider
        try:
            tesseract_provider = TesseractOCRProvider()
            self.registry.register_provider("tesseract", tesseract_provider)
            logger.info("Tesseract OCR provider registered")
        except ImportError:
            logger.info("Tesseract not available, using stub provider only")
        
        # Set default provider
        if "tesseract" in self.registry.providers:
            self.registry.set_default_provider("tesseract")
        else:
            self.registry.set_default_provider("stub")
    
    async def extract_text(self, 
                          image_bytes: bytes, 
                          mimetype: str, 
                          language_hint: Optional[str] = None,
                          timeout: int = 30,
                          provider_name: Optional[str] = None) -> OCRResult:
        """Extract text from image/document using specified or default provider."""
        try:
            # Get appropriate provider
            provider = self.registry.get_provider(provider_name)
            
            # Check if provider supports mimetype
            if not provider.supports_mimetype(mimetype):
                logger.warning("Provider does not support mimetype", 
                             provider=provider.__class__.__name__, 
                             mimetype=mimetype)
                # Try to find a provider that supports this mimetype
                for name, p in self.registry.providers.items():
                    if p.supports_mimetype(mimetype):
                        provider = p
                        logger.info("Switched to compatible provider", provider=name)
                        break
                else:
                    return OCRResult(
                        success=False,
                        text="",
                        confidence=0.0,
                        processing_time=0.0,
                        backend="none",
                        error_message=f"No provider supports mimetype: {mimetype}"
                    )
            
            # Extract text with timeout
            result = await asyncio.wait_for(
                provider.extract_text(image_bytes, mimetype, language_hint, timeout),
                timeout=timeout
            )
            
            logger.info("OCR text extraction completed",
                       success=result.success,
                       backend=result.backend,
                       processing_time=result.processing_time,
                       text_length=len(result.text))
            
            return result
            
        except asyncio.TimeoutError:
            logger.error("OCR processing timed out", timeout=timeout)
            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                processing_time=timeout,
                backend=provider_name or "unknown",
                error_message=f"OCR processing timed out after {timeout} seconds"
            )
        except Exception as exc:
            logger.error("OCR processing failed", exc_info=exc)
            return OCRResult(
                success=False,
                text="",
                confidence=0.0,
                processing_time=0.0,
                backend=provider_name or "unknown",
                error_message=str(exc)
            )
    
    def get_available_providers(self) -> List[str]:
        """Get list of available OCR providers."""
        return list(self.registry.providers.keys())
    
    def get_provider_info(self, name: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific provider."""
        if name in self.registry.providers:
            provider = self.registry.providers[name]
            return {
                "name": name,
                "backend_type": provider.get_backend_type().value,
                "supports_images": provider.supports_mimetype("image/jpeg"),
                "supports_pdfs": provider.supports_mimetype("application/pdf")
            }
        return None


# Global OCR service instance
_ocr_service: Optional[OCRService] = None


def get_ocr_service() -> OCRService:
    """Get global OCR service instance."""
    global _ocr_service
    if _ocr_service is None:
        _ocr_service = OCRService()
    return _ocr_service
