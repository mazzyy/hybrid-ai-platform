"""
E-T-A RAG Prototype - LLM Router

This module is responsible for analyzing the incoming user query and dynamically
routing it to the most appropriate backend:
- 'rag': Needs to access internal documentation/knowledge base.
- 'confidential': Contains sensitive information that shouldn't leave the network.
- 'complex': Requires advanced reasoning, coding, etc. (Azure OpenAI).
- 'simple': Basic queries that can be answered efficiently locally (Ollama).
"""

import json
import ollama
from openai import AzureOpenAI
from config.settings import OLLAMA_MODEL

class QueryRouter:
    def __init__(self):
        self.model = OLLAMA_MODEL
        
    def classify_query(self, query: str) -> str:
        """
        Submits the query to the local LLM and asks it to classify it into 
        one of four categories:
        - rag
        - confidential
        - complex
        - simple
        """
        prompt = f"""You are a query classifier router system. Your job is to classify the user's query into exactly ONE of these four categories:

1. 'rag': The query asks for internal company data, statistics, reports, policies, technical manuals, specific technical facts, data from a datasource, or anything related to the company's internal knowledge base (e.g. "What is the thermal rating of CB-2410?").
2. 'confidential': The query contains sensitive information like passwords, secrets, internal IPs, PII, or explicitly mentions confidential aspects.
3. 'complex': The query requires advanced reasoning, solving math problems, writing code, generating scripts, writing a long essay, or asks for a continuous list of facts (e.g., "Write a numbered list of facts about the universe").
4. 'simple': Basic conversational chat, greetings (e.g. "hi", "hello", "how are you"), or generic easy questions that do not fit into the other categories.

Respond with exactly ONE word: 'rag', 'confidential', 'complex', or 'simple'. Do not provide any conversational text, explanations, or quotes.

User Query: {query}
Category:"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.0, "num_predict": 10},
            )
            category = response["message"]["content"].strip().lower()
            
            # Clean up the response just in case the LLM was chatty
            if "rag" in category: return "rag"
            if "confidential" in category: return "confidential"
            if "complex" in category: return "complex"
            return "simple"
            
        except Exception as e:
            print(f"Warning: Router classification failed, defaulting to 'simple'. Error: {e}")
            return "simple"
