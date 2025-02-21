# risk_rag_system/nodes/generator_node.py

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel
from loguru import logger

from nodes.base_node import BaseNode, NodeInput, NodeOutput
from llm.mistral_router import MistralRouter

class ContentPlan(BaseModel):
    """Structure for content generation plan"""
    sections: List[Dict[str, Any]]
    style_guide: Dict[str, Any]
    citations: List[Dict[str, Any]]
    constraints: Dict[str, Any]

class GeneratorInput(NodeInput):
    """Input structure for generator node"""
    query: str
    validated_content: List[Dict[str, Any]]
    target_length: Optional[int] = None
    style: Optional[str] = None
    format: Optional[str] = None
    constraints: Dict[str, Any] = {}

class GeneratorOutput(NodeOutput):
    """Output structure for generator node"""
    generated_content: str
    content_plan: ContentPlan
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]

class GeneratorNode(BaseNode):
    """Node responsible for content generation"""
    
    def _initialize_node(self) -> None:
        """Initialize generator components"""
        self.model = MistralRouter()
        self.min_confidence_threshold = 0.7
        self.style_templates = self._load_style_templates()
        logger.info("Initialized GeneratorNode")

    def _load_style_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load content style templates"""
        return {
            "academic": {
                "tone": "formal",
                "structure": "research",
                "citation_style": "apa"
            },
            "technical": {
                "tone": "precise",
                "structure": "documentation",
                "citation_style": "inline"
            },
            "conversational": {
                "tone": "informal",
                "structure": "dialogue",
                "citation_style": "narrative"
            }
        }

    async def process(self, input_data: GeneratorInput) -> GeneratorOutput:
        """Process generation request"""
        logger.info(f"Generating content for query: {input_data.query}")
        
        try:
            # Create content plan
            content_plan = await self._create_content_plan(
                query=input_data.query,
                validated_content=input_data.validated_content,
                style=input_data.style,
                constraints=input_data.constraints
            )
            
            # Extract and format citations
            citations = await self._prepare_citations(
                validated_content=input_data.validated_content,
                style=content_plan.style_guide["citation_style"]
            )
            
            # Generate content sections
            sections = []
            for section in content_plan.sections:
                section_content = await self._generate_section(
                    section=section,
                    validated_content=input_data.validated_content,
                    style_guide=content_plan.style_guide
                )
                sections.append(section_content)
            
            # Combine sections
            final_content = self._combine_sections(
                sections,
                style=content_plan.style_guide["structure"]
            )
            
            # Validate output
            confidence_score = await self._validate_output(
                generated_content=final_content,
                original_query=input_data.query,
                validated_content=input_data.validated_content
            )

            return GeneratorOutput(
                content=final_content,
                confidence_score=confidence_score,
                generated_content=final_content,
                content_plan=content_plan,
                citations=citations,
                metadata={
                    "sections_generated": len(sections),
                    "total_length": len(final_content),
                    "style_used": input_data.style
                }
            )

        except Exception as e:
            logger.error(f"Error in generation process: {e}")
            raise

    async def _create_content_plan(
        self,
        query: str,
        validated_content: List[Dict[str, Any]],
        style: Optional[str],
        constraints: Dict[str, Any]
    ) -> ContentPlan:
        """Create plan for content generation"""
        # Prepare style guide
        style_guide = self.style_templates.get(
            style,
            self.style_templates["conversational"]
        ).copy()
        
        # Create planning prompt
        prompt = f"""Create a content plan for this query:
        Query: {query}
        Style: {style_guide['tone']}
        Available Content: {len(validated_content)} sources
        
        Plan should include:
        1. Main sections to cover
        2. Key points per section
        3. Content flow and structure
        4. Citation placement
        
        Additional constraints:
        {constraints}
        """
        
        # Get plan from model
        plan_data = await self.model.get_structured_output(prompt)
        
        return ContentPlan(
            sections=plan_data["sections"],
            style_guide=style_guide,
            citations=[],  # Will be filled later
            constraints=constraints
        )

    async def _prepare_citations(
        self,
        validated_content: List[Dict[str, Any]],
        citation_style: str
    ) -> List[Dict[str, Any]]:
        """Prepare citations in specified style"""
        citations = []
        
        for content in validated_content:
            citation = {
                "source": content.get("source", "Unknown"),
                "reference_id": f"ref_{len(citations) + 1}",
                "metadata": content.get("metadata", {})
            }
            
            # Format citation based on style
            if citation_style == "apa":
                citation["formatted"] = self._format_apa_citation(content)
            elif citation_style == "inline":
                citation["formatted"] = self._format_inline_citation(content)
            else:  # narrative
                citation["formatted"] = self._format_narrative_citation(content)
                
            citations.append(citation)
            
        return citations

    async def _generate_section(
        self,
        section: Dict[str, Any],
        validated_content: List[Dict[str, Any]],
        style_guide: Dict[str, Any]
    ) -> str:
        """Generate individual content section"""
        # Filter relevant content for section
        relevant_content = self._filter_relevant_content(
            section["topic"],
            validated_content
        )
        
        # Create section prompt
        prompt = f"""Generate content for this section:
        Topic: {section['topic']}
        Key Points: {section['key_points']}
        Style: {style_guide['tone']}
        
        Use this content:
        {relevant_content}
        
        Requirements:
        1. Maintain specified style
        2. Include relevant citations
        3. Focus on key points
        4. Ensure factual accuracy
        """
        
        return await self.model.get_completion(prompt)

    def _filter_relevant_content(
        self,
        topic: str,
        content: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Filter content relevant to section topic"""
        relevant = []
        
        for item in content:
            # Calculate relevance score
            relevance = self._calculate_relevance(topic, item["content"])
            if relevance > 0.5:  # Threshold for relevance
                relevant.append({
                    **item,
                    "relevance_score": relevance
                })
                
        return sorted(
            relevant,
            key=lambda x: x["relevance_score"],
            reverse=True
        )

    def _calculate_relevance(self, topic: str, content: str) -> float:
        """Calculate content relevance to topic"""
        # Simple relevance calculation
        topic_words = set(topic.lower().split())
        content_words = set(content.lower().split())
        
        overlap = len(topic_words.intersection(content_words))
        return overlap / len(topic_words)

    def _combine_sections(
        self,
        sections: List[str],
        style: str
    ) -> str:
        """Combine generated sections"""
        if style == "research":
            return "\n\n".join(sections)
        elif style == "documentation":
            return "\n\n---\n\n".join(sections)
        else:  # dialogue
            return "\n".join(sections)

    async def _validate_output(
        self,
        generated_content: str,
        original_query: str,
        validated_content: List[Dict[str, Any]]
    ) -> float:
        """Validate generated output"""
        prompt = f"""Validate this generated content:
        Query: {original_query}
        Content: {generated_content}
        
        Rate from 0 to 1 based on:
        1. Query relevance
        2. Factual accuracy
        3. Content quality
        4. Source utilization
        """
        
        return float(await self.model.get_rating(prompt))

    async def route(self, output: GeneratorOutput) -> Optional[BaseNode]:
        """Route based on generation results"""
        if output.confidence_score < self.min_confidence_threshold:
            return self.next_nodes.get('low_confidence')
        return self.next_nodes.get('default')

    def _format_apa_citation(self, content: Dict[str, Any]) -> str:
        """Format citation in APA style"""
        metadata = content.get("metadata", {})
        
        if "author" in metadata and "year" in metadata:
            return f"{metadata['author']} ({metadata['year']})"
        return f"({content.get('source', 'Unknown Source')})"

    def _format_inline_citation(self, content: Dict[str, Any]) -> str:
        """Format citation for inline use"""
        return f"[{content.get('source', 'Unknown')}]"

    def _format_narrative_citation(self, content: Dict[str, Any]) -> str:
        """Format citation for narrative flow"""
        return f"according to {content.get('source', 'the source')}"

    def get_state(self) -> Dict[str, Any]:
        """Get current state of the node"""
        return {
            "available_styles": list(self.style_templates.keys()),
            "confidence_threshold": self.min_confidence_threshold
        }