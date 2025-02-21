# risk_rag_system/nodes/web_search_node.py

from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from loguru import logger
import aiohttp
import asyncio
from urllib.parse import quote_plus

from nodes.base_node import BaseNode, NodeInput, NodeOutput
from llm.mistral_router import MistralRouter

class SearchResult(BaseModel):
    """Structure for web search results"""
    title: str
    snippet: str
    url: str
    source: str
    relevance_score: float

class WebSearchInput(NodeInput):
    """Input structure for web search node"""
    query: str
    num_results: int = 5
    min_relevance: float = 0.6
    search_type: str = "general"  # general, news, academic
    time_range: Optional[str] = None  # recent, day, week, month, year

class WebSearchOutput(NodeOutput):
    """Output structure for web search node"""
    results: List[SearchResult]
    enhanced_query: str
    search_metadata: Dict[str, Any]

class WebSearchNode(BaseNode):
    """Node for web search fallback"""
    
    def _initialize_node(self) -> None:
        """Initialize web search components"""
        self.model = MistralRouter()
        self.search_apis = {
            "general": self._search_general,
            "news": self._search_news,
            "academic": self._search_academic
        }
        self.session = None
        self.api_keys = self._load_api_keys()
        logger.info("Initialized WebSearchNode")

    def _load_api_keys(self) -> Dict[str, str]:
        """Load API keys from environment/config"""
        # Implementation would load from secure config
        return {
            "serp_api": "YOUR_SERP_API_KEY",
            "google_api": "YOUR_GOOGLE_API_KEY",
            "semantic_scholar": "YOUR_S2_API_KEY"
        }

    async def process(self, input_data: WebSearchInput) -> WebSearchOutput:
        """Process web search request"""
        logger.info(f"Processing web search for query: {input_data.query}")
        
        try:
            # Initialize aiohttp session if needed
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Enhance query for better search results
            enhanced_query = await self._enhance_query(
                input_data.query,
                input_data.search_type
            )

            # Perform search based on type
            search_func = self.search_apis.get(
                input_data.search_type,
                self._search_general
            )
            
            raw_results = await search_func(
                enhanced_query,
                input_data.num_results,
                input_data.time_range
            )

            # Score and filter results
            scored_results = await self._score_results(
                raw_results,
                input_data.query,
                input_data.min_relevance
            )

            # Calculate confidence based on result quality
            confidence_score = self._calculate_confidence(scored_results)

            return WebSearchOutput(
                content=scored_results,
                confidence_score=confidence_score,
                results=scored_results,
                enhanced_query=enhanced_query,
                search_metadata={
                    "original_query": input_data.query,
                    "search_type": input_data.search_type,
                    "time_range": input_data.time_range,
                    "num_results_found": len(scored_results)
                }
            )

        except Exception as e:
            logger.error(f"Error in web search process: {e}")
            raise

    async def _enhance_query(self, query: str, search_type: str) -> str:
        """Enhance query for better search results"""
        prompt = f"""Enhance this search query for {search_type} search:
        Query: {query}
        
        Requirements:
        1. Maintain original intent
        2. Add relevant context
        3. Use appropriate search operators
        4. Optimize for {search_type} results
        """
        
        enhanced = await self.model.get_completion(prompt)
        return enhanced.strip()

    async def _search_general(
        self,
        query: str,
        num_results: int,
        time_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform general web search"""
        params = {
            "q": quote_plus(query),
            "num": num_results,
            "key": self.api_keys["google_api"]
        }
        
        if time_range:
            params["dateRestrict"] = time_range
            
        async with self.session.get(
            "https://www.googleapis.com/customsearch/v1",
            params=params
        ) as response:
            data = await response.json()
            return data.get("items", [])

    async def _search_news(
        self,
        query: str,
        num_results: int,
        time_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform news search"""
        params = {
            "q": quote_plus(query),
            "category": "news",
            "num": num_results,
            "key": self.api_keys["serp_api"]
        }
        
        if time_range:
            params["time"] = time_range
            
        async with self.session.get(
            "https://serpapi.com/search",
            params=params
        ) as response:
            data = await response.json()
            return data.get("news_results", [])

    async def _search_academic(
        self,
        query: str,
        num_results: int,
        time_range: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Perform academic search"""
        headers = {"x-api-key": self.api_keys["semantic_scholar"]}
        params = {
            "query": query,
            "limit": num_results
        }
        
        if time_range:
            current_year = 2024  # This should be dynamic
            params["year"] = f"{current_year - 1}-{current_year}"
            
        async with self.session.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            headers=headers
        ) as response:
            data = await response.json()
            return data.get("data", [])

    async def _score_results(
        self,
        results: List[Dict[str, Any]],
        query: str,
        min_relevance: float
    ) -> List[SearchResult]:
        """Score and filter search results"""
        scored_results = []
        
        for result in results:
            # Construct result text based on source
            result_text = f"""
            Title: {result.get('title', '')}
            Snippet: {result.get('snippet', result.get('abstract', ''))}
            Source: {result.get('source', 'web')}
            """
            
            # Get relevance score from model
            prompt = f"""Rate the relevance of this result:
            Query: {query}
            Result: {result_text}
            
            Return a score between 0 and 1.
            Consider:
            1. Topic relevance
            2. Information value
            3. Source credibility
            4. Time relevance
            """
            
            score = float(await self.model.get_rating(prompt))
            
            if score >= min_relevance:
                scored_results.append(SearchResult(
                    title=result.get('title', ''),
                    snippet=result.get('snippet', result.get('abstract', '')),
                    url=result.get('link', result.get('url', '')),
                    source=result.get('source', 'web'),
                    relevance_score=score
                ))
                
        return sorted(
            scored_results,
            key=lambda x: x.relevance_score,
            reverse=True
        )

    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate confidence score based on results"""
        if not results:
            return 0.0
            
        # Consider top 3 results
        top_scores = [r.relevance_score for r in results[:3]]
        avg_score = sum(top_scores) / len(top_scores)
        
        # Adjust based on number of results
        score_adjustment = min(len(results) / 5, 1.0)  # Max adjustment at 5 results
        
        return avg_score * score_adjustment

    async def route(self, output: WebSearchOutput) -> Optional[BaseNode]:
        """Route based on search results"""
        if not output.results:
            return self.next_nodes.get('no_results')
        if output.confidence_score < 0.5:
            return self.next_nodes.get('low_confidence')
        return self.next_nodes.get('default')

    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the node"""
        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "search_types": list(self.search_apis.keys()),
            "has_active_session": self.session is not None
        }