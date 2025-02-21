# risk_rag_system/validation/rules/validation_rules.py

from typing import List, Dict, Any, Optional, Callable
from pydantic import BaseModel
from loguru import logger
import re
from datetime import datetime

class RuleConfig(BaseModel):
    """Configuration for validation rules"""
    name: str
    description: str
    severity: str  # "error", "warning", "info"
    enabled: bool = True
    parameters: Dict[str, Any] = {}

class ValidationResult(BaseModel):
    """Result of validation check"""
    rule_name: str
    passed: bool
    severity: str
    message: str
    location: Optional[str] = None
    context: Dict[str, Any] = {}

class ValidationRules:
    """Manages content validation rules"""
    
    def __init__(self):
        self.rules: Dict[str, Callable] = {}
        self.rule_configs: Dict[str, RuleConfig] = {}
        self._register_default_rules()
        logger.info("Initialized ValidationRules")

    def _register_default_rules(self) -> None:
        """Register default validation rules"""
        # Content quality rules
        self.register_rule(
            name="minimum_length",
            func=self._check_minimum_length,
            config=RuleConfig(
                name="Minimum Length Check",
                description="Ensures content meets minimum length requirements",
                severity="error",
                parameters={"min_length": 100}
            )
        )
        
        self.register_rule(
            name="maximum_length",
            func=self._check_maximum_length,
            config=RuleConfig(
                name="Maximum Length Check",
                description="Ensures content doesn't exceed maximum length",
                severity="error",
                parameters={"max_length": 10000}
            )
        )
        
        # Source validation rules
        self.register_rule(
            name="source_citation",
            func=self._check_source_citations,
            config=RuleConfig(
                name="Source Citation Check",
                description="Verifies proper source citations",
                severity="warning",
                parameters={"min_citations": 1}
            )
        )
        
        self.register_rule(
            name="source_recency",
            func=self._check_source_recency,
            config=RuleConfig(
                name="Source Recency Check",
                description="Checks if sources are recent enough",
                severity="warning",
                parameters={"max_age_days": 365}
            )
        )
        
        # Content quality rules
        self.register_rule(
            name="readability_score",
            func=self._check_readability,
            config=RuleConfig(
                name="Readability Check",
                description="Checks content readability level",
                severity="warning",
                parameters={"max_grade_level": 12}
            )
        )
        
        # Factual consistency rules
        self.register_rule(
            name="fact_consistency",
            func=self._check_fact_consistency,
            config=RuleConfig(
                name="Fact Consistency Check",
                description="Verifies consistency of facts across content",
                severity="error",
                parameters={"similarity_threshold": 0.8}
            )
        )
        
        # Risk-specific rules
        self.register_rule(
            name="risk_statement_clarity",
            func=self._check_risk_statements,
            config=RuleConfig(
                name="Risk Statement Clarity",
                description="Ensures risk statements are clear and specific",
                severity="error",
                parameters={"min_risk_context": 50}
            )
        )

    def register_rule(
        self,
        name: str,
        func: Callable,
        config: RuleConfig
    ) -> None:
        """Register a new validation rule"""
        self.rules[name] = func
        self.rule_configs[name] = config
        logger.info(f"Registered validation rule: {name}")

    async def validate_content(
        self,
        content: str,
        metadata: Dict[str, Any]
    ) -> List[ValidationResult]:
        """Run all enabled validation rules on content"""
        results = []
        
        for name, rule_func in self.rules.items():
            config = self.rule_configs[name]
            if config.enabled:
                try:
                    result = await rule_func(content, metadata, config.parameters)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error running validation rule {name}: {e}")
                    results.append(ValidationResult(
                        rule_name=name,
                        passed=False,
                        severity="error",
                        message=f"Rule execution failed: {str(e)}"
                    ))
                    
        return results

    async def _check_minimum_length(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check if content meets minimum length requirement"""
        min_length = parameters["min_length"]
        content_length = len(content.split())
        
        return ValidationResult(
            rule_name="minimum_length",
            passed=content_length >= min_length,
            severity="error",
            message=f"Content length ({content_length} words) below minimum ({min_length})",
            context={"actual_length": content_length}
        )

    async def _check_maximum_length(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check if content exceeds maximum length"""
        max_length = parameters["max_length"]
        content_length = len(content.split())
        
        return ValidationResult(
            rule_name="maximum_length",
            passed=content_length <= max_length,
            severity="error",
            message=f"Content length ({content_length} words) exceeds maximum ({max_length})",
            context={"actual_length": content_length}
        )

    async def _check_source_citations(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check for proper source citations"""
        min_citations = parameters["min_citations"]
        citations = re.findall(r'\[\d+\]|\(\w+,\s*\d{4}\)', content)
        
        return ValidationResult(
            rule_name="source_citation",
            passed=len(citations) >= min_citations,
            severity="warning",
            message=f"Found {len(citations)} citations, minimum required is {min_citations}",
            context={"citations_found": citations}
        )

    async def _check_source_recency(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check if sources are recent enough"""
        max_age_days = parameters["max_age_days"]
        source_dates = metadata.get("source_dates", [])
        
        if not source_dates:
            return ValidationResult(
                rule_name="source_recency",
                passed=False,
                severity="warning",
                message="No source dates found to verify recency",
                context={}
            )
        
        current_date = datetime.now()
        outdated_sources = []
        
        for date_str in source_dates:
            try:
                source_date = datetime.strptime(date_str, "%Y-%m-%d")
                age_days = (current_date - source_date).days
                if age_days > max_age_days:
                    outdated_sources.append(date_str)
            except ValueError:
                continue
        
        return ValidationResult(
            rule_name="source_recency",
            passed=len(outdated_sources) == 0,
            severity="warning",
            message=f"Found {len(outdated_sources)} outdated sources",
            context={"outdated_sources": outdated_sources}
        )

    async def _check_readability(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check content readability level"""
        max_grade_level = parameters["max_grade_level"]
        grade_level = self._calculate_grade_level(content)
        
        return ValidationResult(
            rule_name="readability_score",
            passed=grade_level <= max_grade_level,
            severity="warning",
            message=f"Content grade level {grade_level:.1f} exceeds maximum {max_grade_level}",
            context={"grade_level": grade_level}
        )

    async def _check_fact_consistency(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check for factual consistency"""
        similarity_threshold = parameters["similarity_threshold"]
        
        # Extract factual statements
        facts = self._extract_facts(content)
        inconsistencies = self._find_inconsistencies(facts, similarity_threshold)
        
        return ValidationResult(
            rule_name="fact_consistency",
            passed=len(inconsistencies) == 0,
            severity="error",
            message=f"Found {len(inconsistencies)} potential factual inconsistencies",
            context={"inconsistencies": inconsistencies}
        )

    async def _check_risk_statements(
        self,
        content: str,
        metadata: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> ValidationResult:
        """Check clarity of risk statements"""
        min_risk_context = parameters["min_risk_context"]
        risk_statements = self._extract_risk_statements(content)
        unclear_statements = []
        
        for statement in risk_statements:
            context_length = len(statement.split())
            if context_length < min_risk_context:
                unclear_statements.append(statement)
        
        return ValidationResult(
            rule_name="risk_statement_clarity",
            passed=len(unclear_statements) == 0,
            severity="error",
            message=f"Found {len(unclear_statements)} unclear risk statements",
            context={"unclear_statements": unclear_statements}
        )

    def _calculate_grade_level(self, text: str) -> float:
        """Calculate text grade level using Flesch-Kincaid"""
        words = len(text.split())
        sentences = len(re.split(r'[.!?]+', text))
        syllables = self._count_syllables(text)
        
        if sentences == 0 or words == 0:
            return 0.0
            
        return 0.39 * (words / sentences) + 11.8 * (syllables / words) - 15.59

    def _count_syllables(self, text: str) -> int:
        """Count syllables in text"""
        text = text.lower()
        count = 0
        vowels = "aeiouy"
        on_vowel = False
        
        for char in text:
            is_vowel = char in vowels
            if is_vowel and not on_vowel:
                count += 1
            on_vowel = is_vowel
            
        return count

    def _extract_facts(self, content: str) -> List[str]:
        """Extract factual statements from content"""
        # Simple fact extraction based on common patterns
        fact_patterns = [
            r'is [^.!?]*\.',
            r'are [^.!?]*\.',
            r'was [^.!?]*\.',
            r'were [^.!?]*\.'
        ]
        
        facts = []
        for pattern in fact_patterns:
            facts.extend(re.findall(pattern, content))
        
        return facts

    def _find_inconsistencies(
        self,
        facts: List[str],
        threshold: float
    ) -> List[Dict[str, Any]]:
        """Find potentially inconsistent facts"""
        inconsistencies = []
        
        for i, fact1 in enumerate(facts):
            for fact2 in facts[i+1:]:
                similarity = self._calculate_similarity(fact1, fact2)
                if similarity > threshold and self._are_contradicting(fact1, fact2):
                    inconsistencies.append({
                        "fact1": fact1,
                        "fact2": fact2,
                        "similarity": similarity
                    })
                    
        return inconsistencies

    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _are_contradicting(self, fact1: str, fact2: str) -> bool:
        """Check if facts are potentially contradicting"""
        # Simple contradiction check based on negation
        negation_words = {"not", "no", "never", "cannot", "isn't", "aren't", "wasn't", "weren't"}
        
        has_negation1 = any(word in fact1.lower().split() for word in negation_words)
        has_negation2 = any(word in fact2.lower().split() for word in negation_words)
        
        return has_negation1 != has_negation2

    def _extract_risk_statements(self, content: str) -> List[str]:
        """Extract risk-related statements"""
        risk_patterns = [
            r'risk[s]? [^.!?]*[.!?]',
            r'hazard[s]? [^.!?]*[.!?]',
            r'danger[s]? [^.!?]*[.!?]',
            r'warning[s]? [^.!?]*[.!?]',
            r'threat[s]? [^.!?]*[.!?]'
        ]
        
        statements = []
        for pattern in risk_patterns:
            statements.extend(re.findall(pattern, content, re.IGNORECASE))
            
        return statements

    def get_state(self) -> Dict[str, Any]:
        """Get current state of validation rules"""
        return {
            "active_rules": [
                {
                    "name": name,
                    "config": config.dict()
                }
                for name, config in self.rule_configs.items()
                if config.enabled
            ]
        }