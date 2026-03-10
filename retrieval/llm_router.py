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
        prompt = f"""You are a query classification system for a secure enterprise router. Classify the user's query into exactly ONE category.

Evaluate categories in this order. Pick the FIRST one that matches.

1. 'confidential': The query DIRECTLY CONTAINS or EXPLICITLY REQUESTS actual sensitive data such as:
   - Actual credentials appearing in the text: passwords, API keys, tokens, secrets, SSH keys (e.g. "My password is X", "Here is the API key: sk-...")
   - Actual PII appearing in the text: social security numbers, passport numbers, credit card numbers, bank account numbers, IBAN numbers (e.g. "SSN is 123-45-6789", "Credit card 4111-1111...")
   - Explicit requests to REVEAL or EXTRACT protected data: "Give me the root password", "Send me the SSL private keys", "What are the database credentials"
   - Documents explicitly marked as sensitive: confidential, classified, restricted, top secret, eyes only, privileged, not for distribution
   - Secrecy language: "keep this between us", "off the record", "do not share", "do not forward", "do not disclose"
   - Requests for raw personal records: individual salary amounts, medical records, drug test results, background checks, disciplinary records
   CRITICAL DISTINCTION: A query that ASKS ABOUT a company policy, process, or general information is NOT confidential even if it mentions employees, projects, or internal systems. For example:
   - "How many weeks of vacation do employees get?" → RAG (asking about a policy)
   - "What is employee 5512's salary?" → CONFIDENTIAL (requesting specific PII)
   - "Show me the maintenance manual for EX-1200" → RAG (requesting a document)
   - "Show me the confidential merger document" → CONFIDENTIAL (document marked sensitive)

2. 'rag': The query asks to RETRIEVE a specific fact, document, or record that EXISTS in the company's internal systems:
   - Product specifications, part numbers, model data, technical ratings
   - Company policies, SOPs, HR procedures, handbooks, guidelines
   - Internal reports, dashboards, metrics, KPIs, project statuses
   - Org charts, directories, schedules, office information
   - Technical manuals, wiring diagrams, installation guides
   - The answer would be FOUND in a company document, not REASONED or GENERATED
   CRITICAL DISTINCTION: If the query asks you to EXPLAIN HOW something works, COMPARE technologies, WALK THROUGH a process conceptually, or ANALYZE a topic — that is 'complex', not 'rag'. RAG is only for looking up stored company information. For example:
   - "What is our company's firewall configuration?" → RAG (look up internal info)
   - "What are the firewall rules for the DMZ?" → RAG (look up internal config)
   - "Walk through how HTTPS/TLS handshake works step by step" → COMPLEX (explain a concept)
   - "Compare TCP and UDP architectures" → COMPLEX (analytical reasoning)
   - "What is the troubleshooting guide for error E-47?" → RAG (look up a document)
   - "Walk through how DNS resolution works" → COMPLEX (explain a general concept)

3. 'complex': The query requires the LLM to GENERATE, COMPUTE, REASON, or COMPOSE rather than look something up:
   - Writing code, scripts, algorithms, SQL queries, or configurations
   - ANY math computation, calculation, or formula — even short ones like "What is 17 factorial?" or "What is the square root of X?"
   - Writing essays, stories, articles, creative content, or long-form text
   - Explaining how technologies, protocols, or systems work conceptually
   - Comparing, contrasting, or analyzing technical topics
   - Multi-step reasoning, word problems, or probability questions
   - Designing systems, schemas, or architectures
   - Generating lists of 10+ items
   CRITICAL DISTINCTION: If the question requires COMPUTATION or REASONING to answer (even if it looks short), it is 'complex'. For example:
   - "What is 17 factorial?" → COMPLEX (requires computation)
   - "What is the capital of France?" → SIMPLE (just recall a fact)
   - "What is the square root of 5280 divided by 14?" → COMPLEX (requires math)
   - "How many days are in a week?" → SIMPLE (trivial recall)
   - "What is the expected number of coin flips to get 3 heads in a row?" → COMPLEX (requires probability reasoning)

4. 'simple': Use ONLY when the query needs NO computation, reasoning, generation, or internal lookup:
   - Greetings and farewells: "hi", "hello", "bye", "thanks"
   - Trivial factual recall with no computation: "What is the capital of France?"
   - Simple word translations, spellings, or definitions
   - Casual chat or personal opinions
   - If answering requires ANY math, reasoning, or more than one sentence of explanation, it is NOT simple

Respond with exactly ONE word: 'confidential', 'rag', 'complex', or 'simple'.

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
