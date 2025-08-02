import os
import requests
import PyPDF2
import io
import docx2txt
import aiofiles
import aiohttp
import uuid
import hashlib
import time
import asyncio
import gc
import psutil
import ipaddress
import socket
import tempfile
from typing import List, Dict, Any, Optional, Set
from urllib.parse import urlparse, unquote
from pathlib import Path
from email import policy
from email.parser import BytesParser

from app.config import get_settings
from app.utils.chunk_strategies import semantic_chunking

class DocumentProcessor:
    def __init__(self):
        self.settings = get_settings()
        self.document_dir = Path(self.settings.DOCUMENT_DIR)
        
        # Ensure document directory exists
        os.makedirs(self.document_dir, exist_ok=True)
        
        self.supported_extensions = {
            '.pdf': self._process_pdf,
            '.docx': self._process_docx,
            '.eml': self._process_email
        }
        
        # Allowed MIME types for security
        self.allowed_mime_types = {
            'application/pdf': '.pdf',
            'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            'message/rfc822': '.eml',
            'application/octet-stream': None,  # Will be validated by extension
        }
        
        # Maximum document size (10MB default)
        self.max_document_size = getattr(self.settings, 'MAX_DOCUMENT_SIZE_MB', 10) * 1024 * 1024
        
        # Maximum PDF pages to process
        self.max_pdf_pages = getattr(self.settings, 'MAX_PDF_PAGES', 200)
    
    async def process_document(self, document_url: str) -> List[Dict[str, Any]]:
        """Process a document from a URL and return chunked content
        
        Args:
            document_url: URL of the document to process
            
        Returns:
            List of document chunks with text and metadata
        """
        start_time = time.time()
        print(f"🚀 Starting document processing: {document_url}")
        
        try:
            # Generate a unique document ID based on URL
            doc_id = self._generate_doc_id(document_url)
            print(f"📄 Document ID: {doc_id}")
            
            # Check if document is already downloaded
            local_path = self.document_dir / f"{doc_id}"
            if not local_path.exists():
                # Download document with timeout
                content = await asyncio.wait_for(
                    self._download_document_async(document_url),
                    timeout=60  # 60-second timeout for download
                )
                
                # Check document size
                if len(content) > self.max_document_size:
                    raise ValueError(f"Document size ({len(content)/1024/1024:.2f} MB) exceeds maximum allowed size ({self.max_document_size/1024/1024} MB)")
                
                # Save document locally
                with open(local_path, 'wb') as f:
                    f.write(content)
            else:
                # Check file size before loading
                if local_path.stat().st_size > self.max_document_size:
                    raise ValueError(f"Document size ({local_path.stat().st_size/1024/1024:.2f} MB) exceeds maximum allowed size ({self.max_document_size/1024/1024} MB)")
                    
                # Load document from local storage
                with open(local_path, 'rb') as f:
                    content = f.read()
            
            # Process based on document type
            file_extension = self._get_extension(document_url)
            # Before processing document type
            print(f"🔍 Processing document with extension: {file_extension}")
            self._log_memory_usage("Before document processing")
            
            if file_extension in self.supported_extensions:
                # Process with timeout
                text_chunks = await asyncio.wait_for(
                    self._process_document_with_type(content, file_extension),
                    timeout=300  # Increased timeout to 5 minutes for processing
                )
            else:
                raise ValueError(f"Unsupported document type: {file_extension}")
            
            print(f"✅ Document processed: {len(text_chunks)} text segments extracted")
            self._log_memory_usage("After document processing")
            
            # Before chunking
            print(f"🧩 Starting semantic chunking...")
            
            # Apply semantic chunking with timeout
            chunked_data = await asyncio.wait_for(
                self._apply_chunking(text_chunks, document_url, doc_id),
                timeout=300  # Increased timeout to 5 minutes for chunking
            )
            
            print(f"✅ Chunking complete: {len(chunked_data)} chunks generated")
            self._log_memory_usage("After chunking")
            
            processing_time = time.time() - start_time
            print(f"⏱️ Document processing completed in {processing_time:.2f} seconds")
            
            return chunked_data
        
        except asyncio.TimeoutError:
            print(f"⏰ TIMEOUT: Document processing timed out after {time.time() - start_time:.2f} seconds")
            self._log_memory_usage("At timeout")
            raise TimeoutError(f"Document processing timed out after {time.time() - start_time:.2f} seconds")
        except ValueError as e:
            print(f"❌ VALIDATION ERROR: {str(e)}")
            self._log_memory_usage("At validation error")
            # Preserve the original error type for validation errors
            raise 
        except Exception as e:
            print(f"❌ PROCESSING ERROR: {str(e)}")
            self._log_memory_usage("At processing error")
            # Wrap unknown errors to avoid exposing internal details
            raise ValueError(f"Error processing document: Document processing failed")
    
    def _is_internal_hostname(self, hostname: str) -> bool:
        """Check if hostname resolves to internal/private IP"""
        try:
            # Try to resolve the hostname to an IP address
            ip = socket.gethostbyname(hostname)
            
            # Check if it's a private IP
            return (
                ipaddress.ip_address(ip).is_private or
                ipaddress.ip_address(ip).is_loopback or
                ipaddress.ip_address(ip).is_link_local
            )
        except (socket.gaierror, ValueError):
            # If hostname can't be resolved, assume it's safe (external)
            return False
    
    def _is_safe_url(self, url: str) -> bool:
        """Validate URL for safety against SSRF"""
        parsed = urlparse(url)
        
        # Only allow http and https schemes
        if parsed.scheme not in ['http', 'https']:
            return False
            
        # Prevent access to internal networks
        if self._is_internal_hostname(parsed.netloc):
            return False
            
        return True
            
    def _validate_file_extension(self, url: str) -> str:
        """Validate and return the file extension"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        # Check file extension
        extension = os.path.splitext(path)[1]
        if extension not in self.supported_extensions:
            raise ValueError(f"Unsupported file extension: {extension}")
            
        return extension
    
    async def _download_document_async(self, url: str) -> bytes:
        """Download document from URL asynchronously with proper URL handling"""
        try:
            # Better URL normalization
            parsed_url = urlparse(url)
            path = parsed_url.path
            
            # Handle local file paths
            if parsed_url.scheme == 'file' or not parsed_url.scheme:
                # For local files, read directly instead of using requests
                if os.path.exists(path):
                    async with aiofiles.open(path, 'rb') as f:
                        return await f.read()
                else:
                    raise ValueError(f"Local file not found: {path}")
            
            # Validate URL safety to prevent SSRF
            if not self._is_safe_url(url):
                raise ValueError(f"URL validation failed: {url} - Could be attempting to access internal resources")
                
            # Validate file extension before download
            extension = self._validate_file_extension(url)
            
            # Handle URL encoding properly - normalize the URL once
            url = unquote(url)
            
            # Use aiohttp for async HTTP requests instead of requests with run_in_executor
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=30, allow_redirects=True) as response:
                    if response.status != 200:
                        raise ValueError(f"Failed to download document: HTTP {response.status}")
                    
                    # Validate content type
                    content_type = response.headers.get('content-type', '').lower().split(';')[0]
                    if content_type not in self.allowed_mime_types:
                        raise ValueError(f"Unsupported content type: {content_type}")
                    
                    # If content type is application/octet-stream, rely on extension validation
                    if content_type != 'application/octet-stream' and self.allowed_mime_types[content_type] != extension:
                        raise ValueError(f"Content type {content_type} doesn't match extension {extension}")
                    
                    return await response.read()
                    
        except asyncio.TimeoutError:
            raise TimeoutError(f"Download timed out for URL: {url}")
        except Exception as e:
            print(f"Download error: {str(e)}")
            raise ValueError(f"Failed to download document: {str(e)}")
    
    async def _process_document_with_type(self, content: bytes, file_extension: str) -> List[Dict[str, Any]]:
        """Process document with appropriate handler based on type"""
        loop = asyncio.get_event_loop()
        process_func = self.supported_extensions[file_extension]
        return await loop.run_in_executor(None, process_func, content)
    
    def _log_memory_usage(self, stage: str):
        """Log current memory usage"""
        process = psutil.Process(os.getpid())
        mem_info = process.memory_info()
        memory_mb = mem_info.rss / 1024 / 1024
        available_mb = self._get_available_memory_mb()
        print(f"Memory usage at {stage}: {memory_mb:.2f} MB (Available: {available_mb:.2f} MB)")
    
    def _get_available_memory_mb(self) -> float:
        """Get available system memory in MB"""
        # Get memory stats from psutil
        mem = psutil.virtual_memory()
        return mem.available / 1024 / 1024
    
    async def _apply_chunking(self, text_chunks: List[Dict[str, Any]], document_url: str, doc_id: str) -> List[Dict[str, Any]]:
        """Apply semantic chunking to document chunks with memory efficiency"""
        # Add a guard for empty chunks
        if not text_chunks:
            return []
        
        self._log_memory_usage("Before chunking")
        print(f"🧩 Chunking {len(text_chunks)} text segments")
        
        # Calculate optimal batch size based on available memory
        available_memory = self._get_available_memory_mb()
        # Adaptive batch size: smaller batches when memory is constrained
        adaptive_batch_size = max(5, min(20, int(available_memory / 50)))
        print(f"Using adaptive batch size: {adaptive_batch_size} (based on {available_memory:.0f}MB available memory)")
        
        # Process chunks in batches to reduce memory pressure
        result_chunks = []
        total_batches = (len(text_chunks) + adaptive_batch_size - 1) // adaptive_batch_size
        
        for i in range(0, len(text_chunks), adaptive_batch_size):
            # Check memory before processing each batch
            if self._get_available_memory_mb() < 100:  # If less than 100MB available
                print("⚠️ Low memory detected, reducing batch size")
                # Process remaining chunks one by one
                for j in range(i, len(text_chunks)):
                    try:
                        single_item = [text_chunks[j]]
                        single_result = semantic_chunking(
                            single_item, 
                            chunk_size=self.settings.CHUNK_SIZE, 
                            chunk_overlap=self.settings.CHUNK_OVERLAP,
                            metadata={"document_url": document_url, "doc_id": doc_id}
                        )
                        result_chunks.extend(single_result)
                        
                        # Clear memory after each item
                        del single_item, single_result
                        gc.collect()
                        
                        print(f"Processed item {j+1}/{len(text_chunks)} individually")
                    except Exception as e:
                        print(f"Failed to process chunk {j}: {str(e)}")
                
                # Break out of the main loop
                break
            
            # Get current batch
            end_idx = min(i + adaptive_batch_size, len(text_chunks))
            batch = text_chunks[i:end_idx]
            print(f"Chunking batch {i//adaptive_batch_size + 1}/{total_batches}")
            
            try:
                # Process this batch with a limit on total chunks per batch
                batch_result = semantic_chunking(
                    batch, 
                    chunk_size=self.settings.CHUNK_SIZE, 
                    chunk_overlap=self.settings.CHUNK_OVERLAP,
                    metadata={"document_url": document_url, "doc_id": doc_id}
                )
                
                # Append results in smaller groups to avoid memory spikes
                chunk_group_size = 100  # Add chunks in groups of 100
                for j in range(0, len(batch_result), chunk_group_size):
                    result_chunks.extend(batch_result[j:j+chunk_group_size])
                
                # Explicitly clean up to reduce memory pressure
                del batch, batch_result
                gc.collect()
                
            except MemoryError:
                print("⚠️ Memory error during chunking, falling back to one-by-one processing")
                # If we hit memory issues, process remaining items one by one
                for j in range(i, len(text_chunks)):
                    try:
                        single_item = [text_chunks[j]]
                        single_result = semantic_chunking(
                            single_item, 
                            chunk_size=self.settings.CHUNK_SIZE, 
                            chunk_overlap=self.settings.CHUNK_OVERLAP,
                            metadata={"document_url": document_url, "doc_id": doc_id}
                        )
                        result_chunks.extend(single_result)
                        
                        # Force garbage collection after each item
                        del single_item, single_result
                        gc.collect()
                    except Exception as e:
                        print(f"Failed to process chunk {j}: {str(e)}")
                
                # Break out of the main loop
                break
            except Exception as e:
                print(f"Error in batch {i//adaptive_batch_size + 1}: {str(e)}")
                # Continue with next batch
            
            # Clean up batch data and force garbage collection
            batch = None
            batch_result = None
            gc.collect()
            
            # Log memory usage after each batch
            self._log_memory_usage(f"After chunking batch {i//adaptive_batch_size + 1}")
        
        # Clear the original chunks to free memory
        text_chunks = None
        gc.collect()
        
        self._log_memory_usage("After chunking complete")
        print(f"✅ Chunking complete: {len(result_chunks)} chunks generated")
        
        return result_chunks
    
    def _process_pdf(self, content: bytes) -> List[Dict[str, Any]]:
        """Process PDF document with memory and page limits"""
        text_chunks = []
        temp_path = None
        
        try:
            self._log_memory_usage("PDF processing start")
            
            # Create a temporary file to save memory
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(content)
                temp_path = temp_file.name
            
            # Clear content from memory as we now have it in a file
            content = None
            
            try:
                # Open the PDF from file path instead of keeping it in memory
                pdf_reader = PyPDF2.PdfReader(temp_path)
                
                # Limit the number of pages to process
                num_pages = len(pdf_reader.pages)
                if num_pages > self.max_pdf_pages:
                    print(f"Warning: PDF has {num_pages} pages, limiting to {self.max_pdf_pages}")
                    num_pages = self.max_pdf_pages
                
                # Adaptively set batch size based on available memory
                available_mb = self._get_available_memory_mb()
                adaptive_batch = max(1, min(5, int(available_mb / 100)))
                print(f"Using adaptive PDF batch size: {adaptive_batch} pages per batch")
                
                # Process pages in smaller batches
                for batch_start in range(0, num_pages, adaptive_batch):
                    batch_end = min(batch_start + adaptive_batch, num_pages)
                    print(f"Processing PDF pages {batch_start+1}-{batch_end}")
                    
                    batch_chunks = []
                    for page_num in range(batch_start, batch_end):
                        try:
                            page = pdf_reader.pages[page_num]
                            text = page.extract_text() or ""
                            
                            # Set a reasonable limit for page text size
                            max_page_len = 50000  # 50K chars max per page
                            if len(text) > max_page_len:
                                print(f"⚠️ Truncating oversized page {page_num+1} ({len(text)} chars)")
                                text = text[:max_page_len]
                            
                            if text.strip():  # Only add non-empty pages
                                batch_chunks.append({
                                    "text": text,
                                    "metadata": {
                                        "page": page_num + 1, 
                                        "type": "pdf"
                                    }
                                })
                            
                            # Free memory immediately
                            page = None
                            text = None
                            
                        except Exception as e:
                            print(f"Error extracting text from page {page_num+1}: {str(e)}")
                    
                    # Add the batch to the result
                    text_chunks.extend(batch_chunks)
                    
                    # Clean up memory after each batch
                    del batch_chunks
                    gc.collect()
                    self._log_memory_usage(f"After processing PDF pages {batch_start+1}-{batch_end}")
                
                # Clear the reader to free memory
                pdf_reader = None
                gc.collect()
            
            finally:
                # Always clean up the temp file
                if temp_path and os.path.exists(temp_path):
                    try:
                        os.unlink(temp_path)
                    except Exception as e:
                        print(f"Warning: Could not delete temp file {temp_path}: {e}")
                    
            self._log_memory_usage("PDF processing complete")
            return text_chunks
            
        except Exception as e:
            # Ensure temp file is cleaned up even on error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
            
            self._log_memory_usage("PDF processing error")
            print(f"Error processing PDF: {str(e)}")
            raise ValueError(f"Error processing PDF: {str(e)}")
    
    def _process_docx(self, content: bytes) -> List[Dict[str, Any]]:
        """Process DOCX document with memory management"""
        try:
            docx_file = io.BytesIO(content)
            text = docx2txt.process(docx_file)
            
            # Clear the content and file object to free memory
            content = None
            docx_file.close()
            docx_file = None
            
            # Basic paragraph splitting - this could be enhanced with more sophisticated splitting
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            # Clear the original text to free memory
            text = None
            
            # Limit number of paragraphs if too large
            max_paragraphs = 1000
            if len(paragraphs) > max_paragraphs:
                print(f"Warning: DOCX has {len(paragraphs)} paragraphs, limiting to {max_paragraphs}")
                paragraphs = paragraphs[:max_paragraphs]
            
            result = [
                {
                    "text": paragraph,
                    "metadata": {
                        "paragraph": i + 1,
                        "type": "docx"
                    }
                }
                for i, paragraph in enumerate(paragraphs)
            ]
            
            # Clear paragraphs to free memory
            paragraphs = None
            
            return result
        
        except Exception as e:
            print(f"Error processing DOCX: {str(e)}")
            raise ValueError(f"Error processing DOCX: {str(e)}")
    
    def _process_email(self, content: bytes) -> List[Dict[str, Any]]:
        """Process email document with memory management"""
        try:
            email_message = BytesParser(policy=policy.default).parse(io.BytesIO(content))
            
            # Clear content to free memory
            content = None
            
            # Extract headers
            headers = {
                "From": str(email_message.get("From", "")),
                "To": str(email_message.get("To", "")),
                "Subject": str(email_message.get("Subject", "")),
                "Date": str(email_message.get("Date", ""))
            }
            
            chunks = []
            
            # Add headers as a chunk
            header_text = "\n".join([f"{k}: {v}" for k, v in headers.items()])
            chunks.append({
                "text": header_text,
                "metadata": {
                    "section": "header",
                    "type": "email"
                }
            })
            
            # Process body with size limits
            max_part_size = 100000  # 100K chars max per part
            
            if email_message.is_multipart():
                for part_idx, part in enumerate(email_message.iter_parts()):
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            part_content = part.get_content()
                            
                            # Limit part size to prevent memory issues
                            if len(part_content) > max_part_size:
                                print(f"⚠️ Truncating large email part {part_idx} ({len(part_content)} chars)")
                                part_content = part_content[:max_part_size]
                            
                            chunks.append({
                                "text": part_content,
                                "metadata": {
                                    "section": "body",
                                    "content_type": content_type,
                                    "type": "email",
                                    "part_index": part_idx
                                }
                            })
                        except Exception as e:
                            print(f"Error processing email part {part_idx}: {str(e)}")
            else:
                try:
                    body_content = email_message.get_content()
                    
                    # Limit body size
                    if len(body_content) > max_part_size:
                        print(f"⚠️ Truncating large email body ({len(body_content)} chars)")
                        body_content = body_content[:max_part_size]
                    
                    chunks.append({
                        "text": body_content,
                        "metadata": {
                            "section": "body",
                            "content_type": email_message.get_content_type(),
                            "type": "email"
                        }
                    })
                except Exception as e:
                    print(f"Error processing email body: {str(e)}")
            
            # Clear email message to free memory
            email_message = None
            
            return chunks
        
        except Exception as e:
            print(f"Error processing email: {str(e)}")
            raise ValueError(f"Error processing email: {str(e)}")
    
    def _get_extension(self, url: str) -> str:
        """Extract file extension from URL safely"""
        parsed = urlparse(url)
        path = parsed.path
        
        # Handle URLs with query parameters
        path = path.split('?')[0]
        
        # Get extension
        ext = os.path.splitext(path)[1].lower()
        
        # If no extension, try to determine from content-type or filename
        if not ext:
            # Try to get from the filename part of the URL
            filename = os.path.basename(path)
            if '.' in filename:
                ext = '.' + filename.split('.')[-1].lower()
        
        return ext
    
    def _generate_doc_id(self, url: str) -> str:
        """Generate a unique document ID based on URL"""
        # Create a hash that's safe for use in filenames and paths
        doc_id = hashlib.md5(url.encode()).hexdigest()
        
        # Additional validation to ensure no path traversal is possible
        if not all(c.isalnum() for c in doc_id):
            raise ValueError("Generated document ID contains invalid characters")
            
        return doc_id