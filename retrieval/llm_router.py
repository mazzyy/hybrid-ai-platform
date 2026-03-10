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

IMPORTANT: Evaluate categories in this exact order. Pick the FIRST one that matches.

1. 'confidential': SELECT THIS FIRST if ANY of these apply:
   - Query contains or requests sensitive data: passwords, API keys, tokens, secrets, credentials, encryption keys, certificates, SSH keys
   - Query contains or requests PII: social security numbers, passport numbers, credit card numbers, bank accounts, IBAN, dates of birth, salary figures, medical records, drug test results, biometric data, home addresses, personal phone numbers
   - Query contains or requests security-sensitive info: IP addresses, firewall rules, network configurations, SNMP strings, admin consoles, access codes, vulnerability reports
   - Query references or contains confidential documents: anything marked confidential, classified, restricted, privileged, internal only, eyes only, top secret, not for distribution
   - Query uses secrecy language: "keep this between us", "don't share", "off the record", "hush", "under wraps", "do not forward", "for your eyes only", "do not disclose"
   - Query reveals employee personal details: individual salaries, performance reviews, disciplinary records, medical leave, disability status, background checks
   - When in doubt between 'confidential' and any other category, ALWAYS choose 'confidential'

2. 'rag': The query asks to retrieve or look up information from the company's internal knowledge base. This includes:
   - Product specifications, part numbers, model data (e.g. "What is the thermal rating of CB-2410?")
   - Company policies, SOPs, HR procedures, employee handbooks
   - Internal reports, dashboards, metrics, KPIs
   - Internal tools, systems, org charts, contact directories
   - Technical manuals, installation guides, troubleshooting docs
   - The query is asking to FIND or LOOK UP existing information, not to CREATE new content
   NOTE: If the query also involves sensitive data, credentials, or PII, classify as 'confidential' instead.

3. 'complex': The query asks to CREATE, GENERATE, ANALYZE, or COMPUTE something substantial:
   - Writing code, scripts, or algorithms
   - Solving math problems, equations, or proofs
   - Writing essays, stories, articles, or long-form content (more than a paragraph)
   - Detailed technical explanations or multi-step analysis
   - Generating lists of 10+ items
   - Designing systems, architectures, or schemas

4. 'simple': ONLY use this if none of the above apply:
   - Greetings: "hi", "hello", "good morning", "hey"
   - Farewells: "bye", "thanks", "see you"
   - Basic factual questions with short answers: "What is the capital of France?"
   - Simple conversions, translations, or definitions
   - Casual chat or opinions: "What's your favorite color?"

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
