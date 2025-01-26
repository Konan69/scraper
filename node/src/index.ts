import { ChatOpenAI } from "@langchain/openai";
import FirecrawlApp from "@mendable/firecrawl-js";
import { PineconeStore } from "@langchain/pinecone";
import type { Document } from "@langchain/core/documents";
import { CohereEmbeddings } from "@langchain/cohere";
import { Pinecone } from "@pinecone-database/pinecone";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { FireCrawlLoader } from "@langchain/community/document_loaders/web/firecrawl";
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { createHistoryAwareRetriever } from "langchain/chains/history_aware_retriever";
import { createRetrievalChain } from "langchain/chains/retrieval";
import {
  ChatPromptTemplate,
  MessagesPlaceholder,
} from "@langchain/core/prompts";
import * as readline from "readline";

import "dotenv/config";
import { stdout } from "process";

// Environment variables validation
const requiredEnvVars = [
  "FIRECRAWL_API_KEY",
  "OPENAI_API_KEY",
  "COHERE_API_KEY",
  "PINECONE_API_KEY",
] as const;

for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`${envVar} is not set in environment variables`);
  }
}

const firecrawlApiKey = process.env.FIRECRAWL_API_KEY!;
const openaiApiKey = process.env.OPENAI_API_KEY!;
const cohereApiKey = process.env.COHERE_API_KEY!;
const pineconeApiKey = process.env.PINECONE_API_KEY!;

const llm = new ChatOpenAI({
  model: "gpt-4",
  temperature: 0,
  streaming: true,
  openAIApiKey: openaiApiKey,
});

const embeddings = new CohereEmbeddings({
  model: "embed-multilingual-v3.0",
  apiKey: cohereApiKey,
});

const pinecone = new Pinecone({
  apiKey: pineconeApiKey,
});

const loader = new FireCrawlLoader({
  url: "https://docs.anthropic.com/en/docs/initial-setup",
  apiKey: firecrawlApiKey,
  mode: "crawl",
  params: {
    maxDepth: 3,
    limit: 30,
    scrapeOptions: {
      formats: ["markdown"],
    },
  },
});

const app = new FirecrawlApp({ apiKey: firecrawlApiKey });

async function cleanupPineconeIndex(indexName: string) {
  try {
    const indexes = await pinecone.listIndexes();
    console.log("Current indexes:", indexes);

    // Check if index exists in the list of indexes
    const existingIndex = indexes.indexes?.find(
      (idx: any) => idx.name === indexName
    );
    if (existingIndex) {
      console.log(`Found existing '${indexName}' index, cleaning up...`);
      try {
        // Get the index
        const index = pinecone.index(indexName);

        // Delete all vectors in the namespace first
        console.log("Deleting all vectors in the index...");
        await index.deleteAll();

        // Then delete the index
        console.log("Deleting the index...");
        await pinecone.deleteIndex(indexName);
        console.log("Successfully deleted existing index");
      } catch (error) {
        console.warn("Error during cleanup, retrying after delay...", error);
        // Wait a bit and try again
        await new Promise((resolve) => setTimeout(resolve, 5000));
        try {
          await pinecone.deleteIndex(indexName);
          console.log("Successfully deleted index on retry");
        } catch (retryError) {
          console.error("Failed to delete index on retry:", retryError);
          throw retryError;
        }
      }
    } else {
      console.log(`No existing index named ${indexName} found`);
    }
  } catch (error) {
    console.error("Error during Pinecone cleanup:", error);
    throw error;
  }
}

async function loadDocuments() {
  try {
    console.log("Loading documents from FireCrawl...");
    const docs = await loader.load();
    console.log(`Loaded ${docs.length} documents`);

    const mdSplitter = RecursiveCharacterTextSplitter.fromLanguage("markdown", {
      chunkSize: 1000,
      chunkOverlap: 200,
    });
    console.log(docs[0]);
    const mdDocs = await mdSplitter.splitDocuments(docs);
    console.log(`Split into ${mdDocs.length} chunks`);
    return mdDocs;
  } catch (error) {
    console.error("Error loading documents:", error);
    throw error;
  }
}

async function setupVectorStore(docs: any[]) {
  const indexName = "scraper";
  try {
    // Create new index
    console.log("Creating Pinecone index...");
    try {
      const index = await pinecone.createIndex({
        name: indexName,
        dimension: 1024,
        metric: "cosine",
        spec: {
          serverless: {
            cloud: "aws",
            region: "us-east-1",
          },
        },
      });

      if (!index) {
        throw new Error("Failed to create Pinecone index");
      }
    } catch (error: any) {
      if (error.message?.includes("ALREADY_EXISTS")) {
        console.log("Index already exists, using existing index");
      } else {
        throw error;
      }
    }

    const pineconeIndex = pinecone.index(indexName);
    console.log("Storing documents in vector store...");
    const vectorStore = await PineconeStore.fromDocuments(docs, embeddings, {
      pineconeIndex,
      namespace: "anthropic",
    });

    return { vectorStore, indexName };
  } catch (error) {
    console.error("Error setting up vector store:", error);
    throw error;
  }
}

async function createChain(vectorStore: PineconeStore) {
  try {
    // Create the contextualize question prompt
    const contextualizeQPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.`,
      ],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
    ]);

    // Create history-aware retriever
    const historyAwareRetriever = await createHistoryAwareRetriever({
      llm,
      retriever: vectorStore.asRetriever(),
      rephrasePrompt: contextualizeQPrompt,
    });

    // Create the QA prompt
    const qaPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        `Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

Context: {context}`,
      ],
      new MessagesPlaceholder("chat_history"),
      ["human", "{input}"],
    ]);

    // Create the question answering chain
    const questionAnswerChain = await createStuffDocumentsChain({
      llm,
      prompt: qaPrompt,
    });

    // Create the final RAG chain
    return createRetrievalChain({
      retriever: historyAwareRetriever,
      combineDocsChain: questionAnswerChain,
    });
  } catch (error) {
    console.error("Error creating chain:", error);
    throw error;
  }
}

function createReadlineInterface() {
  return readline.createInterface({
    input: process.stdin,
    output: process.stdout,
  });
}

async function handleUserInput(
  rl: readline.Interface,
  chain: any,
  chatHistory: (HumanMessage | AIMessage)[],
  indexName: string
): Promise<boolean> {
  return new Promise((resolve) => {
    rl.question(
      'Enter your question (or type "exit" to quit): ',
      async (input) => {
        try {
          if (input.toLowerCase() === "exit") {
            console.log("Cleaning up...");
            await cleanupPineconeIndex(indexName);
            rl.close();
            resolve(true);
            return;
          }

          const result = await chain.stream({
            chat_history: chatHistory,
            input,
          });

          let fullResponse = "";
          await result.pipeTo(
            new WritableStream({
              write(chunk) {
                const response = chunk.answer;
                stdout.write(response);
                fullResponse += response;
              },
            })
          );

          stdout.write("\n"); // Add newline after response
          chatHistory.push(new HumanMessage(input));
          chatHistory.push(new AIMessage(fullResponse));
          resolve(false);
        } catch (error) {
          console.error("Error processing question:", error);
          resolve(false);
        }
      }
    );
  });
}

async function main() {
  let indexName = "scraper";
  try {
    // Initial cleanup of any existing index
    await cleanupPineconeIndex(indexName);

    const docs = await loadDocuments();
    const { vectorStore, indexName: newIndexName } = await setupVectorStore(
      docs
    );
    indexName = newIndexName;

    const chain = await createChain(vectorStore);
    console.log(
      "\n=== Chat initialized. You can start asking questions. ===\n"
    );

    const rl = createReadlineInterface();
    const chatHistory: (HumanMessage | AIMessage)[] = [];

    let shouldExit = false;
    while (!shouldExit) {
      shouldExit = await handleUserInput(rl, chain, chatHistory, indexName);
    }
  } catch (error) {
    console.error("Fatal error:", error);
  } finally {
    await cleanupPineconeIndex(indexName);
    process.exit(0);
  }
}

// Handle process termination
process.on("SIGINT", async () => {
  console.log("\nReceived interrupt signal");
  await cleanupPineconeIndex("scraper");
  process.exit(0);
});

process.on("unhandledRejection", async (error) => {
  console.error("Unhandled promise rejection:", error);
  await cleanupPineconeIndex("scraper");
  process.exit(1);
});

console.log("Starting application...");
main().catch(async (error) => {
  console.error("Fatal error:", error);
  await cleanupPineconeIndex("scraper");
  process.exit(1);
});
