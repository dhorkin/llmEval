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


| Tool              | API          | Example Query                                           |
| ----------------- | ------------ | ------------------------------------------------------- |
| **BookTool**      | Open Library | "Find all books by George Orwell published before 1950" |
| **NASATool**      | NASA NEO     | "Check for Near Earth Objects this weekend"             |
| **PoetryTool**    | PoetryDB     | "Find a sonnet by Shakespeare and explain the metaphor" |
| **NutritionTool** | LogMeal      | "Recommend Mediterranean meals without dairy"           |


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

### Dependency Management

This project uses [pip-tools](https://pip-tools.readthedocs.io/) for reproducible builds.


| File               | Purpose                                         |
| ------------------ | ----------------------------------------------- |
| `requirements.in`  | Direct dependencies with flexible versions      |
| `requirements.txt` | Auto-generated lock file (do not edit manually) |


**Adding or updating dependencies:**

```bash
# Add a new dependency: edit requirements.in, then:
pip-compile requirements.in

# Update all dependencies to latest compatible versions:
pip-compile --upgrade requirements.in

# Update a single package:
pip-compile --upgrade-package <package-name> requirements.in

# Sync your environment to match the lock file:
pip-sync
```

## Configuration

1. Copy the sample environment file:

```bash
cp sample.env .env
```

1. Edit `.env` with your API keys:

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

# Optional: Phoenix evaluation scoring method (default: categorical)
PHOENIX_EVALUATION_METHOD=categorical  # Options: categorical, discrete, continuous

# Optional: Minimum agreement threshold to pass (default: 0.8)
MINIMUM_AGREEMENT_PASS_THRESHOLD=0.8  # Range: 0.0 - 1.0

# Optional: Minimum individual metric score to pass (default: 0.7)
MINIMUM_METRIC_PASS_THRESHOLD=0.7  # Range: 0.0 - 1.0
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


| Metric                | What It Measures                            | Threshold |
| --------------------- | ------------------------------------------- | --------- |
| **Answer Relevancy**  | Is the output relevant to the query?        | 0.7       |
| **Faithfulness**      | Does output only contain info from context? | 0.8       |
| **Tool Correctness**  | Were the correct tools called?              | 1.0       |
| **Schema Validation** | Does output match expected schema?          | 0.8       |


### Phoenix Metrics


| Metric             | What It Measures                     | Threshold |
| ------------------ | ------------------------------------ | --------- |
| **Hallucination**  | Does output contain fabricated info? | 0.8       |
| **QA Correctness** | Is the answer factually correct?     | 0.7       |
| **Relevance**      | Is output relevant to the input?     | 0.7       |


### Score Normalization

**DeepEval Hallucination**: DeepEval's hallucination metric uses an inverted scale where 1.0 = high hallucination (bad) and 0.0 = no hallucination (good). This project normalizes it to match the "higher is better" convention used by other metrics. A displayed score of 0.0 means the response was highly hallucinatory, while 1.0 means no hallucination detected.

> **Note**: When debugging, the reason text will show both the normalized score and the original DeepEval score, e.g., "score is 0.00 (inverted from 1.00)".

### Phoenix Evaluation Methods

Control how Phoenix scores responses using `PHOENIX_EVALUATION_METHOD`:


| Method          | Score Range               | Description                                                                                                                                                                   |
| --------------- | ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **categorical** | 0.0 or 1.0                | Built-in evaluators return text labels (e.g., "factual"/"hallucinated", "correct"/"incorrect", "relevant"/"unrelated") which are mapped to binary 0.0 or 1.0 scores (default) |
| **discrete**    | 0.0, 0.25, 0.5, 0.75, 1.0 | 5-point scale (excellent/good/fair/poor/very_poor)                                                                                                                            |
| **continuous**  | 0.00 - 1.00               | Numeric scores with 2 decimal precision                                                                                                                                       |


**When to use each method:**

- **categorical**: Fast evaluations, CI/CD pipelines where pass/fail is sufficient
- **discrete**: More nuanced feedback while maintaining consistency across runs
- **continuous**: Fine-grained scoring for detailed analysis and drift detection

### Pass/Fail Thresholds

A test case **passes** only if both conditions are met:

1. Framework agreement is at or above `MINIMUM_AGREEMENT_PASS_THRESHOLD`
2. All individual metrics are at or above `MINIMUM_METRIC_PASS_THRESHOLD`

#### Agreement Threshold

The `MINIMUM_AGREEMENT_PASS_THRESHOLD` setting controls how closely DeepEval and Phoenix must agree.

**How agreement is calculated (varies by evaluation method):**


| Method                  | Agreement Calculation                                                    |
| ----------------------- | ------------------------------------------------------------------------ |
| **continuous/discrete** | Average similarity: `1.0 -                                               |
| **categorical**         | Percentage of metrics where both agree on pass/fail (score ≥ 0.5 = pass) |


**Continuous/Discrete Example:**

- Relevance: DeepEval=0.92, Phoenix=1.00 → similarity = 1.0 - 0.08 = 0.92
- Correctness: DeepEval=0.92, Phoenix=0.75 → similarity = 1.0 - 0.17 = 0.83
- Average agreement = (0.92 + 0.83) / 2 = **87.5%**

**Categorical Example:**

- Relevance: DeepEval=0.85 (pass), Phoenix=1.0 (pass) → **AGREE**
- Hallucination: DeepEval=0.70 (pass), Phoenix=0.0 (fail) → **DIFFER**
- Agreement = 1/2 = **50%**


| Threshold         | Use Case                                                                |
| ----------------- | ----------------------------------------------------------------------- |
| **0.9+**          | High confidence required; strict consistency between evaluators         |
| **0.8** (default) | Balanced; allows minor variations while catching major disagreements    |
| **0.7**           | More lenient; useful when evaluators have known methodology differences |


#### Metric Threshold

The `MINIMUM_METRIC_PASS_THRESHOLD` setting controls the minimum acceptable score for any individual metric from either framework.

**Behavior by evaluation method:**


| Method          | Score Range               | With Threshold 0.7                       |
| --------------- | ------------------------- | ---------------------------------------- |
| **categorical** | 0.0 or 1.0                | 0.0 fails, 1.0 passes (no middle ground) |
| **discrete**    | 0.0, 0.25, 0.5, 0.75, 1.0 | 0.75 and 1.0 pass; others fail           |
| **continuous**  | 0.00 - 1.00               | Any score ≥ 0.70 passes                  |


**Example:**

- If Phoenix hallucination = 0.60 and threshold = 0.7, the test **fails** (0.60 < 0.70)
- If DeepEval relevance = 0.85 and threshold = 0.7, the metric **passes** (0.85 ≥ 0.70)


| Threshold         | Use Case                                      |
| ----------------- | --------------------------------------------- |
| **0.8+**          | High quality bar; all metrics must score well |
| **0.7** (default) | Balanced; catches clearly poor responses      |
| **0.5**           | Lenient; only fails on very low scores        |


> **Note:** In categorical mode, thresholds between 0.0 and 1.0 effectively act as binary (only 0.0 fails, only 1.0 passes). Use discrete or continuous mode for more granular threshold control.

**Why separate thresholds?**

- **High agreement, low metric**: Frameworks agree the response is poor → should fail
- **Low agreement, high metrics**: Frameworks disagree but both give decent scores → may indicate evaluation inconsistency
- **Both thresholds**: Ensures responses are both high-quality AND consistently evaluated

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

# Run regression tests (decision engine snapshot testing)
pytest tests/regression/

# Run regression tests and update snapshots if intentionally changed
pytest tests/regression/ --snapshot-update

# Run DeepEval tests (requires API keys)
deepeval test run tests/deepeval/

# Type checking
mypy .
```

### Regression Tests

The `tests/regression/` directory contains snapshot tests for the deterministic decision engine. These tests:

- **Do not require API keys** - They use mock data, no external calls
- **Run fast** - Pure Python logic, no network latency
- **Catch unintended changes** - Snapshots detect if decision engine output structure changes

Test coverage includes:

- **Book**: Unknown author handling, year filtering (before/after), temporal impossibility (Einstein after 2020)
- **NEO**: Invalid date validation (Feb 30th), hazardous asteroid risk assessment, closest approach detection
- **Nutrition**: Conflicting diet restrictions (vegan + beef), nutritional summary generation
- **Poetry**: Sonnet filtering (14 lines), quatrain extraction, literary analysis detection

Run these before submitting PRs to ensure decision engine logic hasn't regressed.

## CI/CD

The project includes GitHub Actions workflow (`.github/workflows/ci.yml`) that:

1. Runs mypy type checking
2. Executes pytest tests (functional, schema, and regression)
3. Runs DeepEval evaluation (on push to main)
4. Runs full evaluation pipeline and verifies pass rate

### CI Fail Conditions

- mypy type checking fails
- Any pytest test fails (including regression snapshot mismatches)
- Any metric < `MINIMUM_METRIC_PASS_THRESHOLD`
- Agreement < `MINIMUM_AGREEMENT_PASS_THRESHOLD`
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