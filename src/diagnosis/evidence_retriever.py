import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
import faiss
from src.retrieval.vector_store import VectorStore

class EvidenceRetriever:
    """Retrieves evidence for individual claims."""
    
    def __init__(self, vector_store: VectorStore, top_k: int = 5):
        self.vector_store = vector_store
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.top_k = top_k
    
    def retrieve_evidence_for_claims(self, claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retrieve evidence for each claim."""
        enhanced_claims = []
        
        for claim in claims:
            evidence = self._retrieve_claim_evidence(claim['text'])
            claim['evidence'] = evidence
            claim['evidence_score'] = self._compute_evidence_quality(evidence)
            enhanced_claims.append(claim)
        
        return enhanced_claims
    
    def _retrieve_claim_evidence(self, claim_text: str) -> List[Dict[str, Any]]:
        """Retrieve evidence documents for a specific claim."""
        # Expand claim into search queries
        queries = self._generate_search_queries(claim_text)
        
        all_evidence = []
        seen_texts = set()
        
        for query in queries:
            results = self.vector_store.search(query, k=self.top_k)
            
            for result in results:
                # Avoid duplicates
                if result['content'] not in seen_texts:
                    evidence_item = {
                        'content': result['content'],
                        'source': result.get('source', 'unknown'),
                        'score': result['score'],
                        'query': query,
                        'relevance': self._compute_relevance(claim_text, result['content'])
                    }
                    all_evidence.append(evidence_item)
                    seen_texts.add(result['content'])
        
        # Sort by relevance and return top evidence
        all_evidence.sort(key=lambda x: x['relevance'], reverse=True)
        return all_evidence[:self.top_k]
    
    def _generate_search_queries(self, claim: str) -> List[str]:
        """Generate multiple search queries for a claim."""
        queries = [claim]  # Original claim
        
        # Extract key entities/phrases
        key_phrases = self._extract_key_phrases(claim)
        for phrase in key_phrases:
            if len(phrase) > 3:  # Skip very short phrases
                queries.append(phrase)
        
        # Generate question form
        question = self._claim_to_question(claim)
        if question:
            queries.append(question)
        
        return queries[:3]  # Limit to 3 queries per claim
    
    def _extract_key_phrases(self, claim: str) -> List[str]:
        """Extract key phrases from claim for search."""
        # Simple approach: extract noun phrases and named entities
        words = claim.split()
        phrases = []
        
        # Extract potential noun phrases (consecutive title case words)
        current_phrase = []
        for word in words:
            if word[0].isupper() or word.lower() in ['the', 'of', 'in', 'at', 'on']:
                current_phrase.append(word)
            else:
                if len(current_phrase) > 1:
                    phrases.append(' '.join(current_phrase))
                current_phrase = []
        
        if len(current_phrase) > 1:
            phrases.append(' '.join(current_phrase))
        
        return phrases
    
    def _claim_to_question(self, claim: str) -> Optional[str]:
        """Convert claim to question form for search."""
        claim = claim.strip()
        
        # Simple patterns for question generation
        if ' is ' in claim:
            parts = claim.split(' is ', 1)
            if len(parts) == 2:
                return f"What is {parts[0]}?"
        
        if ' was ' in claim:
            parts = claim.split(' was ', 1)
            if len(parts) == 2:
                return f"What was {parts[0]}?"
        
        if ' has ' in claim:
            parts = claim.split(' has ', 1)
            if len(parts) == 2:
                return f"What has {parts[0]}?"
        
        return None
    
    def _compute_relevance(self, claim: str, evidence: str) -> float:
        """Compute relevance score between claim and evidence."""
        claim_embedding = self.encoder.encode([claim])
        evidence_embedding = self.encoder.encode([evidence])
        
        # Cosine similarity
        similarity = np.dot(claim_embedding[0], evidence_embedding[0]) / (
            np.linalg.norm(claim_embedding[0]) * np.linalg.norm(evidence_embedding[0])
        )
        
        return float(similarity)
    
    def _compute_evidence_quality(self, evidence_list: List[Dict[str, Any]]) -> float:
        """Compute overall quality score for evidence."""
        if not evidence_list:
            return 0.0
        
        # Average relevance weighted by search score
        total_score = 0.0
        total_weight = 0.0
        
        for evidence in evidence_list:
            weight = evidence['score']
            score = evidence['relevance']
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0.0