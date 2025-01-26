import os
import asyncio
import logging
from typing import List 
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_text_splitters import Language
from langchain_pinecone import PineconeVectorStore
from crawl4ai.async_dispatcher import MemoryAdaptiveDispatcher
from langchain.chains.retrieval import create_retrieval_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains.combine_documents import create_stuff_documents_chain
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlResult, CrawlerRunConfig, CacheMode
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from pydantic import SecretStr
# Load environment variables from .env file
load_dotenv()

# Configure logging
# logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("scraper")

# Environment variables
COHERE_API_KEY_ENV = os.getenv("COHERE_API_KEY")
OPENAI_API_KEY_ENV = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY= os.getenv("PINECONE_API_KEY")

if not COHERE_API_KEY_ENV or not OPENAI_API_KEY_ENV or not PINECONE_API_KEY:
    raise ValueError("Missing required environment variables")

COHERE_API_KEY = SecretStr(COHERE_API_KEY_ENV)  # Wrap in SecretStr
OPENAI_API_KEY = SecretStr(OPENAI_API_KEY_ENV)  # Wrap in SecretStr
# Wrap in SecretStr


async def cleanup_pinecone():
    """Clean up any existing Pinecone indexes at startup"""
    try:
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        indexes = pinecone.list_indexes()
        logger.info(f"Indexes: {indexes}")
        
        # Check if scraper index exists in the list of indexes
        existing_index = next((idx for idx in indexes.indexes if idx.name == "scraper"), None)
        if existing_index:
            logger.info("Found existing 'scraper' index, cleaning up...")
            try:
                # Get the index
                index = pinecone.Index("scraper")

                
                # Delete all vectors in the namespace first
                logger.info("Deleting all vectors in the index...")
                index.delete(delete_all=True)
                
                # Then delete the index
                logger.info("Deleting the index...")
                pinecone.delete_index("scraper")
                logger.info("Successfully deleted existing index")
            except Exception as e:
                logger.warning(f"Error during cleanup: {str(e)}")
                # Wait a bit and try again
                await asyncio.sleep(2)
                try:
                    pinecone.delete_index("scraper")
                    logger.info("Successfully deleted index on retry")
                except Exception as retry_e:
                    logger.error(f"Failed to delete index on retry: {str(retry_e)}")
                    raise
    except Exception as e:
        logger.error(f"Error during Pinecone cleanup: {str(e)}")
        raise

async def crawl_and_process(url: str, max_depth: int = 3, max_pages: int = 30) -> List[Document]:
    """
    Crawl website recursively and return list of LangChain Documents using parallel processing
    Args:
        url: Starting URL to crawl
        max_depth: Maximum depth for recursive crawling (default: 3)
        max_pages: Maximum number of pages to crawl (default: 30)
    """
    try:
        browser_conf = BrowserConfig(
            headless=True,
            java_script_enabled=True,
            verbose=False,
            extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"]
        )
        
        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_for_images=False,
            stream=False
        )

        # Initialize memory-adaptive dispatcher for efficient parallel crawling
        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,  # Pause if memory exceeds 70%
            check_interval=1.0,
            max_session_permit=10  # Maximum concurrent tasks
        )

        documents = []
        visited_urls = set([url])
        urls_to_crawl = [url]
        
        async with AsyncWebCrawler(config=browser_conf) as crawler:
            while urls_to_crawl and len(visited_urls) < max_pages:
                # Take next batch of URLs to process
                current_batch = urls_to_crawl[:max_pages - len(visited_urls)]
                urls_to_crawl = []  # Clear the list as we'll add new URLs from results
                
                # Process batch using arun_many without streaming
                results = await crawler.arun_many(
                    urls=current_batch,
                    config=run_conf,
                    dispatcher=dispatcher
                )
                
                # Process results - results is a List[CrawlResult] when not streaming
                for result in results:  # type: ignore
                    if not result.success:
                        logger.warning(f"Failed to crawl {result.url}: {result.error_message}")
                        continue

                    if result.markdown and isinstance(result.markdown, str):
                        # Create document from the crawled content
                        documents.append(
                            Document(
                                page_content=result.markdown,
                                metadata={"source": result.url}
                            )
                        )
                        
                        # Extract new links if we haven't reached max depth
                        current_depth = len(result.url.split('/')) - len(url.split('/'))
                        if current_depth < max_depth and result.html and isinstance(result.html, str):
                            from bs4 import BeautifulSoup
                            soup = BeautifulSoup(result.html, 'html.parser')
                            new_links = [
                                a.get('href') for a in soup.find_all('a', href=True)
                                if a.get('href').startswith('http')
                            ]
                            
                            # Filter for same domain and unvisited links
                            same_domain_links = [
                                link for link in new_links
                                if is_same_domain(link, url) and link not in visited_urls
                            ]
                            
                            # Add new links to process
                            visited_urls.update(same_domain_links)
                            urls_to_crawl.extend(same_domain_links)
                            
                            if same_domain_links:
                                logger.info(f"Found {len(same_domain_links)} new links at depth {current_depth}")
            
            logger.info(f"Created {len(documents)} documents from {len(visited_urls)} crawled pages")
            return documents
            
    except Exception as e:
        logger.error(f"Error during crawling: {str(e)}")
        raise

def extract_links_from_markdown(markdown: str) -> List[str]:
    """Extract links from markdown content"""
    # Simple regex to find markdown links
    import re
    links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', markdown)
    return [link[1] for link in links]

def is_same_domain(url1: str, url2: str) -> bool:
    """Check if two URLs are from the same domain"""
    from urllib.parse import urlparse
    return urlparse(url1).netloc == urlparse(url2).netloc

async def setup_rag_pipeline(docs):
    try:
        
        # Initialize Pinecone
        pinecone = Pinecone(api_key=PINECONE_API_KEY)
        
        # Create new index
        logger.info("Creating new Pinecone index...")
        try:
            index = pinecone.create_index(
                name="scraper",
                dimension=1024,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region="us-east-1"
                )
            )
        except Exception as e:
            if "ALREADY_EXISTS" in str(e):
                logger.info("Index already exists, using existing index")
                index = pinecone.Index("scraper")
            else:
                raise

        # Initialize components
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,
            streaming=True,
            api_key=OPENAI_API_KEY
        )
        
        embeddings = CohereEmbeddings(
            model="embed-multilingual-v3.0",
            cohere_api_key=COHERE_API_KEY,
            client=None,
            async_client=None

        )
        # Create vector store
        vector_store = PineconeVectorStore.from_documents(
            docs,
            embedding=embeddings,
            index_name="scraper",
            namespace="anthropic"
        )

        # Create prompts
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", "Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\n\nContext: {context}"),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])

        # Create retriever chain
        retriever = create_history_aware_retriever(
            llm=llm,
            retriever=vector_store.as_retriever(),
            prompt=contextualize_q_prompt
        )

        # Create QA chain
        combine_docs_chain = create_stuff_documents_chain(
            llm=llm,
            prompt=qa_prompt
        )

        # Create final chain
        chain = create_retrieval_chain(
            retriever=retriever,
            combine_docs_chain=combine_docs_chain
        )

        return chain, pinecone
    except Exception as e:
        logger.error(f"Error setting up RAG pipeline: {str(e)}")
        raise

async def main():
    try:
        # Clean up any existing indexes first
        await cleanup_pinecone()
        
        # Crawl the documentation
        logger.info("Starting document crawling...")
        documents = await crawl_and_process("https://docs.anthropic.com/en/docs/initial-setup")
        
        # Split content
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.MARKDOWN,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Split all documents
        split_docs = []
        for doc in documents:
            splits = splitter.split_text(doc.page_content)
            split_docs.extend([
                Document(
                    page_content=split,
                    metadata=doc.metadata
                ) for split in splits
            ])
            
        logger.info(f"Split content into {len(split_docs)} chunks")

        # Setup RAG pipeline
        chain, pinecone = await setup_rag_pipeline(split_docs)
        
        # CLI interface
        print("\n=== Chat initialized. You can start asking questions. Type 'exit' to quit. ===\n")
        
        chat_history = []
        while True:
            try:
                question = input("Enter your question: ").strip()
                
                if question.lower() == 'exit':
                    logger.info("Cleaning up...")
                    pinecone.delete_index("scraper")
                    logger.info("Goodbye!")
                    break
                
                response = await chain.ainvoke({
                    "chat_history": chat_history,
                    "input": question
                })
                
                print(f"\nAnswer: {response['answer']}\n")
                chat_history.extend([("human", question), ("assistant", response['answer'])])
                
            except KeyboardInterrupt:
                logger.info("\nReceived interrupt signal. Cleaning up...")
                pinecone.delete_index("scraper")
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("An error occurred. Please try again.")

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        raise
    finally:
        try:
            pinecone.delete_index("scraper")
        except:
            pass

def run():
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    run()


