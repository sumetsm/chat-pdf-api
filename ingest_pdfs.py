"""
PDF Ingestion Script for Chat With PDF System
Usage: python ingest_pdfs.py
"""

import os
import sys
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm

# PDF processing
import fitz  # PyMuPDF
from pypdf import PdfReader

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFIngester:
    def __init__(self, pdfs_dir: str = "./pdfs", db_dir: str = "./chroma_db"):
        self.pdfs_dir = Path(pdfs_dir)
        self.db_dir = Path(db_dir)
        
        # Setup embedding model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        self.embed_model = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        
        # Setup text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Create directories if they don't exist
        self.pdfs_dir.mkdir(exist_ok=True)
        self.db_dir.mkdir(exist_ok=True)
    
    def extract_text_pymupdf(self, pdf_path: Path) -> str:
        """Extract text using PyMuPDF (better for complex layouts)"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_text_pypdf(self, pdf_path: Path) -> str:
        """Extract text using pypdf (fallback method)"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            return text
        except Exception as e:
            logger.error(f"pypdf extraction failed for {pdf_path}: {e}")
            return ""
    
    def extract_pdf_content(self, pdf_path: Path) -> str:
        """Extract text from PDF with fallback methods"""
        logger.info(f"Extracting content from: {pdf_path.name}")
        
        # Try PyMuPDF first
        text = self.extract_text_pymupdf(pdf_path)
        
        # Fallback to pypdf if PyMuPDF fails
        if not text.strip():
            logger.warning(f"PyMuPDF failed for {pdf_path.name}, trying pypdf")
            text = self.extract_text_pypdf(pdf_path)
        
        if not text.strip():
            logger.error(f"Failed to extract text from {pdf_path.name}")
            return ""
        
        logger.info(f"Extracted {len(text)} characters from {pdf_path.name}")
        return text
    
    def process_pdf(self, pdf_path: Path) -> List[Document]:
        """Process a single PDF into document chunks"""
        text = self.extract_pdf_content(pdf_path)
        
        if not text:
            return []
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create documents with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": str(pdf_path),
                    "filename": pdf_path.name,
                    "chunk_id": i,
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        logger.info(f"Created {len(documents)} chunks from {pdf_path.name}")
        return documents
    
    def ingest_pdfs(self):
        """Main ingestion function"""
        # Find all PDF files
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.pdfs_dir}")
            logger.info("Please place PDF files in the ./pdfs directory")
            return
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        # Process all PDFs
        all_documents = []
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            documents = self.process_pdf(pdf_path)
            all_documents.extend(documents)
        
        if not all_documents:
            logger.error("No documents were successfully processed")
            return
        
        logger.info(f"Total documents created: {len(all_documents)}")
        
        # Create/load vector store
        try:
            vectorstore = Chroma(
                collection_name="pdf_docs",
                embedding_function=self.embed_model,
                persist_directory=str(self.db_dir)
            )
            
            # Add documents in batches
            batch_size = 50
            for i in tqdm(range(0, len(all_documents), batch_size), desc="Adding to vector store"):
                batch = all_documents[i:i + batch_size]
                vectorstore.add_documents(batch)
            
            # Persist the vector store
            vectorstore.persist()
            logger.info("✅ All documents successfully ingested and persisted")
            
            # Print summary
            collection = vectorstore._collection
            logger.info(f"Vector store summary:")
            logger.info(f"  - Collection name: {collection.name}")
            logger.info(f"  - Total documents: {collection.count()}")
            logger.info(f"  - Database location: {self.db_dir}")
            
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise
    
    def verify_ingestion(self):
        """Verify that documents were properly ingested"""
        try:
            vectorstore = Chroma(
                collection_name="pdf_docs",
                embedding_function=self.embed_model,
                persist_directory=str(self.db_dir)
            )
            
            # Test search
            test_query = "what"
            results = vectorstore.similarity_search(test_query, k=3)
            
            logger.info(f"Verification test:")
            logger.info(f"  - Query: '{test_query}'")
            logger.info(f"  - Results found: {len(results)}")
            
            for i, doc in enumerate(results):
                logger.info(f"  - Result {i+1}: {doc.metadata.get('filename', 'Unknown')} "
                          f"(chunk {doc.metadata.get('chunk_id', 'N/A')})")
            
            return len(results) > 0
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return False

def main():
    """Main function"""
    logger.info("Starting PDF ingestion process...")
    
    # Initialize ingester
    ingester = PDFIngester()
    
    # Check if PDFs directory exists and has files
    if not ingester.pdfs_dir.exists():
        logger.error(f"PDFs directory does not exist: {ingester.pdfs_dir}")
        logger.info("Please create ./pdfs directory and add your PDF files")
        sys.exit(1)
    
    # Run ingestion
    ingester.ingest_pdfs()
    
    # Verify ingestion
    if ingester.verify_ingestion():
        logger.info("✅ PDF ingestion completed successfully!")
    else:
        logger.error("❌ PDF ingestion verification failed")
        sys.exit(1)

if __name__ == "__main__":
    main()