# anura_system_info.py

class SystemInformation:
    def __init__(self):
        self.system_docs = [
            {
                "title": "About Anura Aficionado",
                "content": """Anura Aficionado (A2) is a specialized assistant that leverages advanced natural language processing to help users navigate and understand the Anura schema within Optilogic's Cosmic Frog platform. As the foundational framework powering Cosmic Frog's optimization, simulation, and risk assessment capabilities, Anura Aficionado eliminates traditional barriers to schema understanding by providing immediate, authoritative guidance for supply chain modelers, developers, and analysts.

Core Capabilities:

1. Schema Understanding
   - Table Details: Provides descriptions, purposes, and complete column listings for all tables
   - Column Details: Explains the usage, data types, and validation rules for individual columns
   - Relationships: Maps connections between tables and their dependencies
   - Technical Requirements: Details primary keys, required fields, and default values

2. System Integration
   - Engine Awareness: Identifies which tables and columns are used by different Cosmic Frog engines
   - Template Models: Provides information about various cosmic frog template models
   - General help: Assists with general purpose questions about Anura"""
            }
        ]

    def get_docs(self):
        """Convert system information into Document objects for RAG system."""
        from configurable_rag import Document
        
        return [
            Document(
                content=doc["content"],
                metadata={
                    "title": doc["title"],
                    "type": "system_info"
                }
            )
            for doc in self.system_docs
        ]