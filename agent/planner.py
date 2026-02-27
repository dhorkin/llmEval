"""Agent planner with LLM integration for tool orchestration."""

from __future__ import annotations

import json
import time
from datetime import datetime
from typing import Any

from anthropic import Anthropic
from openai import OpenAI

from config.settings import get_settings
from decision_engine.rules import DecisionEngine
from models.schemas import (
    AgentResponse,
    BookAnalysisReport,
    MealRecommendationReport,
    NEOReport,
    PoetryAnalysisReport,
    ReportType,
    ToolCall,
)
from tools.book_tool import BookTool
from tools.nasa_tool import NASATool
from tools.nutrition_tool import NutritionTool
from tools.poetry_tool import PoetryTool


class AgentPlanner:
    """LLM-powered agent for orchestrating tools and generating reports."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.decision_engine = DecisionEngine()

        self.book_tool = BookTool()
        self.nasa_tool = NASATool()
        self.poetry_tool = PoetryTool()
        self.nutrition_tool = NutritionTool()

        self._openai_client: OpenAI | None = None
        self._anthropic_client: Anthropic | None = None

    @property
    def openai_client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    @property
    def anthropic_client(self) -> Anthropic:
        """Lazy initialization of Anthropic client."""
        if self._anthropic_client is None:
            self._anthropic_client = Anthropic(api_key=self.settings.anthropic_api_key)
        return self._anthropic_client

    async def run(self, query: str) -> AgentResponse:
        """
        Process a user query through the agent pipeline.

        Steps:
        1. Parse user goal and extract intent
        2. Determine tool requirements
        3. Execute tools
        4. Normalize outputs with decision engine
        5. Generate structured output with LLM
        """
        start_time = time.time()
        tool_calls: list[ToolCall] = []

        intent = self.decision_engine.detect_intent(query)
        tools_needed = self.decision_engine.determine_tools_needed(intent)

        report: ReportType

        if intent == "book_analysis":
            report, tool_calls = await self._handle_book_query(query)
        elif intent == "neo_report":
            report, tool_calls = await self._handle_neo_query(query)
        elif intent == "poetry_analysis":
            report, tool_calls = await self._handle_poetry_query(query)
        elif intent == "meal_recommendation":
            report, tool_calls = await self._handle_nutrition_query(query)
        else:
            report = await self._handle_unknown_query(query)

        reasoning = self._generate_reasoning(query, intent, tools_needed, tool_calls)
        total_latency = (time.time() - start_time) * 1000

        return AgentResponse(
            query=query,
            intent=intent,
            tool_calls=tool_calls,
            report=report,
            reasoning=reasoning,
            timestamp=datetime.utcnow(),
            total_latency_ms=total_latency,
        )

    async def _handle_book_query(
        self, query: str
    ) -> tuple[BookAnalysisReport, list[ToolCall]]:
        """Handle book-related queries."""
        tool_calls: list[ToolCall] = []

        author = self._extract_author_from_query(query)
        title = self._extract_title_from_query(query)
        year_before = self.decision_engine._extract_year_before(query)
        year_after = self.decision_engine._extract_year_after(query)

        params: dict[str, Any] = {"author": author}
        if title:
            params["title"] = title
        if year_before:
            params["published_before"] = year_before
        if year_after:
            params["published_after"] = year_after

        tool_call = ToolCall(
            tool_name="book_search",
            parameters=params,
        )

        try:
            start = time.time()
            result = await self.book_tool.execute(**params)
            tool_call.latency_ms = (time.time() - start) * 1000
            tool_call.result = result.model_dump()
            tool_call.success = True
        except Exception as e:
            tool_call.success = False
            tool_call.error_message = str(e)
            result = None

        tool_calls.append(tool_call)

        if result and result.books:
            theme_summary, common_themes = await self._analyze_book_themes(
                query, result.books
            )
        else:
            theme_summary = "No books found to analyze."
            common_themes = []

        if result:
            report = self.decision_engine.process_book_results(
                query, result, theme_summary, common_themes
            )
        else:
            from models.schemas import BookSearchResult
            empty_result = BookSearchResult(
                query_author=author,
                query_title=title,
                books=[],
                total_found=0,
                no_results=True,
            )
            report = self.decision_engine.process_book_results(
                query, empty_result, "Failed to retrieve book data.", []
            )

        return report, tool_calls

    async def _handle_neo_query(
        self, query: str
    ) -> tuple[NEOReport, list[ToolCall]]:
        """Handle NASA NEO queries."""
        tool_calls: list[ToolCall] = []

        date_range = self.decision_engine.parse_date_intent(query)
        if date_range:
            start_date, end_date = date_range
        else:
            today = datetime.now()
            start_date = today.strftime("%Y-%m-%d")
            end_date = (today + __import__("datetime").timedelta(days=7)).strftime("%Y-%m-%d")

        is_valid, error = self.decision_engine.validate_date(start_date)
        if not is_valid:
            report = NEOReport(
                type="NEO_REPORT",
                query=query,
                date_range=f"{start_date} to {end_date}",
                objects_count=0,
                hazardous_count=0,
                risk_level="normal",
                risk_assessment=f"Invalid date: {error}",
                near_earth_objects=[],
            )
            return report, tool_calls

        params = {"start_date": start_date, "end_date": end_date}
        tool_call = ToolCall(tool_name="nasa_neo", parameters=params)

        try:
            start = time.time()
            result = await self.nasa_tool.execute(**params)
            tool_call.latency_ms = (time.time() - start) * 1000
            tool_call.result = result.model_dump()
            tool_call.success = True
        except Exception as e:
            tool_call.success = False
            tool_call.error_message = str(e)
            result = None

        tool_calls.append(tool_call)

        if result:
            report = self.decision_engine.process_neo_results(query, result)
        else:
            report = NEOReport(
                type="NEO_REPORT",
                query=query,
                date_range=f"{start_date} to {end_date}",
                objects_count=0,
                hazardous_count=0,
                risk_level="normal",
                risk_assessment="Failed to retrieve NEO data.",
                near_earth_objects=[],
            )

        return report, tool_calls

    async def _handle_poetry_query(
        self, query: str
    ) -> tuple[PoetryAnalysisReport, list[ToolCall]]:
        """Handle poetry-related queries."""
        tool_calls: list[ToolCall] = []

        author = self._extract_poet_from_query(query)
        title = self._extract_poem_title_from_query(query)
        linecount = 14 if self.decision_engine._mentions_sonnet(query) else None

        params: dict[str, Any] = {}
        if author:
            params["author"] = author
        if title:
            params["title"] = title
        if linecount:
            params["linecount"] = linecount

        if not params:
            params["author"] = "Shakespeare"

        tool_call = ToolCall(tool_name="poetry_search", parameters=params)

        try:
            start = time.time()
            result = await self.poetry_tool.execute(**params)
            tool_call.latency_ms = (time.time() - start) * 1000
            tool_call.result = result.model_dump()
            tool_call.success = True
        except Exception as e:
            tool_call.success = False
            tool_call.error_message = str(e)
            result = None

        tool_calls.append(tool_call)

        if result and result.poems:
            analysis, literary_devices = await self._analyze_poem(query, result.poems[0])
        else:
            analysis = "No poems found to analyze."
            literary_devices = []

        if result:
            report = self.decision_engine.process_poetry_results(
                query, result, analysis, literary_devices
            )
        else:
            from models.schemas import PoetryResult
            empty_result = PoetryResult(
                query_author=params.get("author"),
                query_title=params.get("title"),
                poems=[],
                total_found=0,
                no_results=True,
            )
            report = self.decision_engine.process_poetry_results(
                query, empty_result, "Failed to retrieve poetry data.", []
            )

        return report, tool_calls

    async def _handle_nutrition_query(
        self, query: str
    ) -> tuple[MealRecommendationReport, list[ToolCall]]:
        """Handle nutrition/meal queries."""
        tool_calls: list[ToolCall] = []

        diet_type, restrictions = self.decision_engine.extract_diet_and_restrictions(query)

        if not diet_type:
            diet_type = "balanced"

        params: dict[str, Any] = {
            "diet_type": diet_type,
            "restrictions": restrictions,
            "meal_count": 3,
        }

        tool_call = ToolCall(tool_name="nutrition_meal_recommendation", parameters=params)

        try:
            start = time.time()
            result = await self.nutrition_tool.execute(**params)
            tool_call.latency_ms = (time.time() - start) * 1000
            tool_call.result = result.model_dump()
            tool_call.success = True
        except Exception as e:
            tool_call.success = False
            tool_call.error_message = str(e)
            result = None

        tool_calls.append(tool_call)

        if result:
            report = self.decision_engine.process_nutrition_results(query, result)
        else:
            report = MealRecommendationReport(
                type="MEAL_RECOMMENDATION",
                query=query,
                diet_type=diet_type,
                restrictions=restrictions,
                recommendations=[],
                nutritional_summary="Failed to retrieve meal recommendations.",
                conflict_detected=False,
            )

        return report, tool_calls

    async def _handle_unknown_query(self, query: str) -> BookAnalysisReport:
        """Handle queries with unknown intent."""
        return BookAnalysisReport(
            type="BOOK_ANALYSIS",
            query=query,
            books_found=[],
            theme_summary="Unable to determine intent from query. Please specify if you want to search for books, check NEO data, find poetry, or get meal recommendations.",
            common_themes=[],
            no_results=True,
        )

    def _extract_author_from_query(self, query: str) -> str:
        """Extract author name from query."""
        patterns = [
            r"(?:by|written by|author)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+books",
            r"books\s+(?:by|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        ]
        import re
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return "Unknown"

    def _extract_title_from_query(self, query: str) -> str | None:
        """Extract book title from query."""
        import re
        match = re.search(r'["\']([^"\']+)["\']', query)
        if match:
            return match.group(1)
        return None

    def _extract_poet_from_query(self, query: str) -> str | None:
        """Extract poet name from query."""
        import re
        patterns = [
            r"(?:by|poem by|sonnet by|poetry by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'s\s+(?:poem|sonnet|poetry)",
        ]
        for pattern in patterns:
            match = re.search(pattern, query)
            if match:
                return match.group(1)
        return None

    def _extract_poem_title_from_query(self, query: str) -> str | None:
        """Extract poem title from query."""
        import re
        match = re.search(r'["\']([^"\']+)["\']', query)
        if match:
            return match.group(1)
        return None

    async def _analyze_book_themes(
        self, query: str, books: list[Any]
    ) -> tuple[str, list[str]]:
        """Use LLM to analyze common themes in books."""
        book_info = "\n".join(
            f"- {b.title} ({b.year}): {', '.join(b.subjects[:5])}"
            for b in books[:5]
        )

        prompt = f"""Analyze the following books and identify their common themes.

Query: {query}

Books found:
{book_info}

Provide:
1. A brief summary of common themes (2-3 sentences)
2. A list of 3-5 key themes

Format your response as JSON:
{{"theme_summary": "...", "common_themes": ["theme1", "theme2", ...]}}"""

        try:
            response = await self._call_llm(prompt)
            data = self._parse_llm_json_response(response, "theme analysis LLM")
            return data.get("theme_summary", ""), data.get("common_themes", [])
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                print("[Agent] Theme analysis: LLM rate limited, using fallback")
            elif "Empty response" in error_msg or "HTML error" in error_msg:
                print(f"[Agent] Theme analysis: {error_msg}, using fallback")
            else:
                print(f"[Agent] Theme analysis failed ({type(e).__name__}): {error_msg}")
            all_subjects: list[str] = []
            for b in books:
                all_subjects.extend(b.subjects[:3])
            unique_subjects = list(set(all_subjects))[:5]
            return f"Found {len(books)} books with themes including: {', '.join(unique_subjects[:3])}", unique_subjects

    async def _analyze_poem(
        self, query: str, poem: Any
    ) -> tuple[str, list[str]]:
        """Use LLM to analyze a poem."""
        focus_on_quatrain = self.decision_engine._mentions_quatrain(query)
        lines_to_analyze = poem.lines[:4] if focus_on_quatrain else poem.lines[:8]
        lines_text = "\n".join(lines_to_analyze)

        prompt = f"""Analyze the following poem excerpt.

Query: {query}

Poem: "{poem.title}" by {poem.author}

Lines:
{lines_text}

Provide:
1. A literary analysis focusing on {'the first quatrain (first 4 lines)' if focus_on_quatrain else 'the poem'}
2. Identify key literary devices (metaphor, imagery, symbolism, etc.)

Format your response as JSON:
{{"analysis": "...", "literary_devices": ["device1: explanation", "device2: explanation", ...]}}"""

        try:
            response = await self._call_llm(prompt)
            data = self._parse_llm_json_response(response, "poem analysis LLM")
            return data.get("analysis", ""), data.get("literary_devices", [])
        except Exception as e:
            error_msg = str(e)
            fallback_result = f"Poem '{poem.title}' by {poem.author} ({poem.linecount} lines)."
            
            if "429" in error_msg or "quota" in error_msg.lower() or "rate limit" in error_msg.lower():
                print("[Agent] Poem analysis: LLM rate limited")
                return fallback_result, ["Analysis skipped: API rate limited"]
            elif "Empty response" in error_msg:
                print("[Agent] Poem analysis: Empty LLM response (possible rate limit)")
                return fallback_result, ["Analysis skipped: Empty API response"]
            elif "HTML error" in error_msg:
                print("[Agent] Poem analysis: Received HTML error page (likely rate limited)")
                return fallback_result, ["Analysis skipped: API returned error page"]
            else:
                print(f"[Agent] Poem analysis failed ({type(e).__name__}): {error_msg}")
                return fallback_result, [f"Analysis failed: {type(e).__name__}"]

    def _parse_llm_json_response(self, response: str, context: str = "LLM") -> dict[str, Any]:
        """Parse JSON from LLM response with diagnostic error handling.
        
        Args:
            response: Raw LLM response string
            context: Description of what operation this is for (for error messages)
            
        Returns:
            Parsed JSON dict
            
        Raises:
            ValueError: With descriptive message about what went wrong
        """
        if not response or not response.strip():
            raise ValueError(f"Empty response from {context} (possible rate limit or API issue)")
        
        cleaned = response.strip()
        
        if cleaned.startswith("<!") or cleaned.startswith("<html"):
            raise ValueError(f"Received HTML error page from {context} (likely rate limited)")
        
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            json_lines = []
            in_block = False
            for line in lines:
                if line.startswith("```") and not in_block:
                    in_block = True
                    continue
                elif line.startswith("```") and in_block:
                    break
                elif in_block:
                    json_lines.append(line)
            cleaned = "\n".join(json_lines).strip()
        
        try:
            result: dict[str, Any] = json.loads(cleaned)
            return result
        except json.JSONDecodeError as e:
            preview = response[:100] + "..." if len(response) > 100 else response
            raise ValueError(f"Invalid JSON from {context}: {e}. Response preview: {preview!r}")

    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.settings.llm_provider == "openai":
            return await self._call_openai(prompt)
        else:
            return await self._call_anthropic(prompt)

    async def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API."""
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.max_tokens,
        )
        content = response.choices[0].message.content
        return content or ""

    async def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API."""
        response = self.anthropic_client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=self.settings.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.content and len(response.content) > 0:
            block = response.content[0]
            if hasattr(block, "text"):
                return str(block.text)
        return ""

    def _generate_reasoning(
        self,
        query: str,
        intent: str,
        tools_needed: list[str],
        tool_calls: list[ToolCall],
    ) -> str:
        """Generate reasoning explanation for the agent's actions."""
        successful_calls = [tc for tc in tool_calls if tc.success]
        failed_calls = [tc for tc in tool_calls if not tc.success]

        reasoning_parts = [
            f"Query: {query}",
            f"Detected intent: {intent}",
            f"Tools required: {', '.join(tools_needed) if tools_needed else 'none'}",
        ]

        if successful_calls:
            reasoning_parts.append(
                f"Successfully executed: {', '.join(tc.tool_name for tc in successful_calls)}"
            )

        if failed_calls:
            reasoning_parts.append(
                f"Failed calls: {', '.join(f'{tc.tool_name}: {tc.error_message}' for tc in failed_calls)}"
            )

        return " | ".join(reasoning_parts)

    async def close(self) -> None:
        """Close all tool connections."""
        await self.book_tool.close()
        await self.nasa_tool.close()
        await self.poetry_tool.close()
        await self.nutrition_tool.close()
