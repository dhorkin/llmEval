# LLM Evaluation Agent

An LLM-powered agent that uses external APIs to answer human-centric queries, with dual evaluation using **DeepEval** and **Arize Phoenix**.

## Overview

This project demonstrates:
1. **Tool-using LLM Agent** - Orchestrates 4 public APIs based on user queries
2. **Structured Outputs** - All responses follow strict Pydantic schemas
3. **Dual Evaluation** - Independent quality assessment using both DeepEval and Phoenix
4. **Drift Detection** - Monitors evaluation scores over time

## Features

### Supported APIs

| Tool | API | Example Query |
|------|-----|---------------|
| **BookTool** | Open Library | "Find all books by George Orwell published before 1950" |
| **NASATool** | NASA NEO | "Check for Near Earth Objects this weekend" |
| **PoetryTool** | PoetryDB | "Find a sonnet by Shakespeare and explain the metaphor" |
| **NutritionTool** | LogMeal | "Recommend Mediterranean meals without dairy" |

### Evaluation Frameworks

- **DeepEval**: Answer relevancy, faithfulness, custom tool correctness metrics
- **Phoenix**: Hallucination detection, QA correctness, relevance evaluation

## Requirements

- **Python 3.9 - 3.13** (Python 3.14+ not yet supported by evaluation frameworks)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd llmEval

# Create virtual environment (ensure you're using Python 3.13 or earlier)
python3.13 -m venv venv  # or: py -3.13 -m venv venv (Windows)
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Verify Python version
python --version  # Should show 3.9.x - 3.13.x

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Copy the sample environment file:
```bash
cp sample.env .env
```

2. Edit `.env` with your API keys:
```bash
# Required for LLM
OPENAI_API_KEY=sk-your-key-here
# or
ANTHROPIC_API_KEY=sk-ant-your-key-here

# Required for tools
NASA_API_KEY=your-nasa-key        # Get free at https://api.nasa.gov/
LOGMEAL_API_KEY=your-logmeal-key  # Get at https://logmeal.es/

# Optional: Rate limiting for evaluation (default 0.1 = 10s between requests)
EVAL_RATE_LIMIT_INITIAL_RPS=0.1
```

## Usage

### Run a Single Query

```bash
python main.py query "Find all books by George Orwell published before 1950"
```

### Run Evaluation

```bash
# Run both DeepEval and Phoenix
python main.py evaluate --framework both

# Run only DeepEval
python main.py evaluate --framework deepeval

# Run only Phoenix
python main.py evaluate --framework phoenix
```

### Run Full Pipeline

```bash
python main.py pipeline
```

This runs the complete flow:
1. Generate test inputs
2. Run agent on each test case
3. Evaluate with both frameworks
4. Generate comparison report
5. Log results

### View Reports

```bash
# View last evaluation report
python main.py report

# Check for drift
python main.py drift-check
```

## Evaluation Metrics Explained

### DeepEval Metrics

| Metric | What It Measures | Threshold |
|--------|------------------|-----------|
| **Answer Relevancy** | Is the output relevant to the query? | 0.7 |
| **Faithfulness** | Does output only contain info from context? | 0.8 |
| **Tool Correctness** | Were the correct tools called? | 1.0 |
| **Schema Validation** | Does output match expected schema? | 0.8 |

### Phoenix Metrics

| Metric | What It Measures | Threshold |
|--------|------------------|-----------|
| **Hallucination** | Does output contain fabricated info? | 0.8 |
| **QA Correctness** | Is the answer factually correct? | 0.7 |
| **Relevance** | Is output relevant to the input? | 0.7 |

## Drift Detection

### What is Drift?

Model drift occurs when evaluation scores gradually decrease over time without code changes. In this agent, drift manifests as:

1. **Tool Selection Drift**: The LLM starts calling incorrect tools for queries it previously handled correctly (e.g., calling PoetryTool for book queries)

2. **Hallucination Drift**: Faithfulness scores decrease as the model begins adding information not present in API responses

3. **Schema Drift**: Output structure becomes inconsistent, failing Pydantic validation more frequently

4. **Relevance Drift**: Answers become less focused on the original query

### Detection Method

The system tracks a rolling 7-day average of each metric. Alerts trigger when any metric drops >10% from the baseline established during the initial evaluation run. Use `python main.py drift-check` to analyze historical scores.

## Project Structure

```
llmEval/
├── agent/               # LLM planner and orchestration
├── decision_engine/     # Deterministic decision rules
├── tools/               # API wrappers (Book, NASA, Poetry, Nutrition)
├── evaluation/          # DeepEval and Phoenix runners
├── models/              # Pydantic schemas
├── tests/               # Test suites
├── logs/                # Evaluation results
└── main.py              # CLI entry point
```

## Running Tests

```bash
# Run all tests
pytest

# Run functional tests only
pytest tests/functional/

# Run schema validation tests
pytest tests/schema/

# Run DeepEval tests (requires API keys)
deepeval test run tests/deepeval/

# Type checking
mypy .
```

## CI/CD

The project includes GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. Runs mypy type checking
2. Executes pytest tests
3. Runs DeepEval evaluation (on push to main)

### CI Fail Conditions

- mypy type checking fails
- Faithfulness score < 0.8
- Schema validation fails
- Tool mismatch detected

## Known Failure Case

**Query**: "Find all books about quantum physics by Einstein published after 2020"

- **Expected**: Empty result (Einstein died 1955)
- **Failure Mode**: LLM may hallucinate fictional books
- **Detection**: Faithfulness metric catches via fact-checking
- **Resolution**: Decision engine cross-checks temporal constraints

## Edge Case

**Query**: "Check NEO data for February 30th"

- **Issue**: Invalid date that doesn't exist
- **Expected**: Validation error with helpful message
- **Test**: Ensures input validation before API calls

## License

MIT
