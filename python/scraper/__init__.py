from collections import defaultdict, deque
import datetime
import os
import asyncio
import logging
from typing import List, Optional
from urllib.parse import urljoin, urlparse 
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
logging.basicConfig(level=logging.INFO)
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

async def crawl_and_process(url: str, max_depth: int = 4, max_pages: int =21) -> List[Document]:
    """Strict depth-first crawling with full error handling"""
    try:
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        base_url = f"{parsed_url.scheme}://{domain}"
        
        browser_conf = BrowserConfig(
            headless=True,
            java_script_enabled=True,
            verbose=False,
        
            extra_args=["--disable-gpu", "--no-sandbox"]
        )

        run_conf = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            wait_for_images=False,
            stream=True
        )

        dispatcher = MemoryAdaptiveDispatcher(
            memory_threshold_percent=70.0,
            max_session_permit=50
        )

        documents = []
        visited = set()
        depth_map = defaultdict(list)
        current_depth = 0

        # Initialize with normalized start URL
        start_url = normalize_url(url, base_url, domain)
        if not start_url:
            raise ValueError("Invalid starting URL")
            
        depth_map[0].append(start_url)
        visited.add(start_url)

        async with AsyncWebCrawler(config=browser_conf) as crawler:
            while current_depth <= max_depth and len(documents) < max_pages:
                # Get all URLs for current depth
                current_urls = depth_map[current_depth]
                if not current_urls:
                    current_depth += 1
                    continue

                # Process all URLs at this depth in one arun_many call
                logger.info(f"Processing depth {current_depth} with {len(current_urls)} URLs")
                results = await crawler.arun_many(
                    urls=current_urls,
                    config=run_conf,
                    dispatcher=dispatcher
                )

                # Process results with retries
                async for result in results: # type: ignore
                    if not result.success:
                        logger.error(f"Failed to crawl {result.url}: {result.error_message}")
                        await handle_retry(result.url, depth_map, current_depth, max_depth, visited)
                        continue

                    # Add document if successful
                    if result.markdown:
                        documents.append(Document(
                            page_content=result.markdown,
                            metadata={
                                "source": result.url,
                                "depth": current_depth,
                                "crawled_at": datetime.datetime.now().isoformat()
                            }
                        ))

                    # Process links for next depth
                    if current_depth < max_depth and result.html:
                        await process_links(
                            result.html, 
                            current_depth + 1,
                            depth_map,
                            base_url,
                            domain,
                            visited,
                            max_pages
                        )

                current_depth += 1
        logger.info(f"Crawled {len(documents)} documents from {len(visited)} pages")

        logger.info(f"visited: {visited}")

        return documents

    except Exception as e:
        logger.error(f"Crawling failed: {str(e)}")
        raise


async def process_links(html: str, next_depth: int, depth_map: defaultdict, 
                        base_url: str, domain: str, visited: set, max_pages: int):
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    links = []
    
    for a in soup.find_all('a', href=True):
        if len(visited) >= max_pages:
            break
        raw_link = a['href']
        normalized = normalize_url(raw_link, base_url, domain)
        if normalized and normalized not in visited:
            visited.add(normalized)
            links.append(normalized)
            logger.debug(f"Found new link: {normalized}")
    
    if links and len(visited) < max_pages:
        depth_map[next_depth].extend(links)

async def handle_retry(url: str, depth_map: defaultdict, current_depth: int, 
                      max_depth: int, visited: set, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        logger.warning(f"Retry {attempt}/{max_retries} for {url}")
        try:
            async with AsyncWebCrawler() as crawler:
                result = await crawler.arun(url)
                if result.success:
                    logger.info(f"Retry succeeded for {url}")
                    return
        except Exception as e:
            logger.error(f"Retry {attempt} failed: {str(e)}")
    
    logger.error(f"Permanent failure for {url}")
    # Do not remove from visited to prevent reprocessing
async def extract_and_filter_links(html: str, base_url: str, domain: str, visited: set, max_pages: int) -> List[str]:
    """Fast link extraction with domain filtering and URL normalization"""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, 'lxml')
    links = []
    
    for a in soup.find_all('a', href=True):
        if len(visited) >= max_pages:
            break
            
        raw_link = a['href']
        normalized = normalize_url(raw_link, base_url, domain)
        
        if normalized and normalized not in visited:
            visited.add(normalized)
            links.append(normalized)
            
            if len(visited) >= max_pages:
                break
    
    return links
def normalize_url(link: str, base_url: str, domain: str) -> Optional[str]:
    parsed = urlparse(link)
    if not parsed.netloc:
        link = urljoin(base_url, link)
        parsed = urlparse(link)
    
    # Allow subdomains by checking if the domain is a suffix
    if not parsed.netloc.endswith(domain):
        return None
    if parsed.scheme not in ('http', 'https'):
        return None
    
    path = parsed.path.rstrip('/') or '/'
    return parsed._replace(
        scheme=parsed.scheme.lower(),
        netloc=parsed.netloc.lower(),
        path=path,
        query=parsed.query.strip(),
        fragment=''
    ).geturl()
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
                question = await asyncio.to_thread(input) 
                question = question.strip()  
                
                if question.lower() == 'exit':
                    logger.info("Cleaning up...")
                    pinecone.delete_index("scraper")
                    logger.info("Goodbye!")
                    break
                full_answer = ""
                
                async for chunk in chain.astream({
                    "chat_history": chat_history,
                    "input": question
                }):
                    if chunk.get('answer') is not None:
                        print(chunk['answer'], end="", flush=True)    
                        full_answer += chunk['answer']
                
                print()
                chat_history.extend([("human", question), ("assistant", full_answer)])
                
            except KeyboardInterrupt:
                logger.info("\nReceived interrupt signal. Cleaning up...")
                pinecone.delete_index("scraper")
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}")
                print("An error occurred. Please try again.")

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


# how to get started  with anthropic api