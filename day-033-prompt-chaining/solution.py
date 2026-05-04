"""
Day 033: Prompt Chaining with Claude API

A multi-step prompt chaining pipeline that decomposes a complex research task
into sequential (and parallel) LLM calls. Each step has a focused role, clear
input/output contract, and validation logic.

Architecture:
    Topic → [Plan Questions] → [Research Each Question] → [Synthesize] → [Critique] → [Edit] → Final Output

We use a simulated LLM backend so this runs without API keys, but the
architecture is identical to what you'd deploy with the real Claude API.
The SimulatedLLM class can be swapped for a real Anthropic client with
zero changes to the chain logic.
"""

import asyncio
import json
import time
import hashlib
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures for the pipeline
# ---------------------------------------------------------------------------

class StepStatus(Enum):
    """Possible outcomes for a chain step."""
    SUCCESS = "success"
    FAILED = "failed"
    RETRIED = "retried"


@dataclass
class StepResult:
    """Captures everything about a single step's execution.

    This is critical for observability — in production, you'd ship these
    to a logging service so you can debug chain failures after the fact.
    """
    step_name: str
    input_text: str
    output_text: str
    status: StepStatus
    latency_ms: float
    tokens_used: int  # Simulated token count
    retries: int = 0
    error: Optional[str] = None


@dataclass
class ChainResult:
    """The full result of running a chain, including all intermediate steps."""
    final_output: str
    steps: list[StepResult] = field(default_factory=list)
    total_latency_ms: float = 0.0
    total_tokens: int = 0

    def summary(self) -> str:
        """Human-readable summary of the chain execution."""
        lines = ["=" * 60, "CHAIN EXECUTION SUMMARY", "=" * 60]
        for i, step in enumerate(self.steps, 1):
            lines.append(
                f"  Step {i}: {step.step_name:<25} "
                f"| {step.status.value:<8} "
                f"| {step.latency_ms:>7.1f}ms "
                f"| {step.tokens_used:>5} tokens "
                f"| retries: {step.retries}"
            )
        lines.append("-" * 60)
        lines.append(
            f"  TOTAL: {self.total_latency_ms:.1f}ms | {self.total_tokens} tokens"
        )
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Simulated LLM — drop-in replacement for a real API client
# ---------------------------------------------------------------------------

class SimulatedLLM:
    """Simulates Claude API responses for demonstration purposes.

    In production, you'd replace this with:
        import anthropic
        client = anthropic.Anthropic()
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}]
        )
        return response.content[0].text

    The key architectural point: the chain logic doesn't care whether
    responses come from a real API or a simulator. This separation of
    concerns makes testing and development much easier.
    """

    # Pre-built responses keyed by step name, so the demo produces
    # realistic-looking output that flows coherently through the chain.
    RESPONSES: dict[str, Callable[[str], str]] = {}

    @classmethod
    def _register_defaults(cls) -> None:
        """Register default response generators for each step type."""

        def plan_response(topic: str) -> str:
            return json.dumps({
                "questions": [
                    f"What are the fundamental technical principles behind {topic}?",
                    f"What are the main real-world applications of {topic} today?",
                    f"What are the key challenges and limitations of {topic}?",
                ]
            })

        def research_response(question: str) -> str:
            # Generate a deterministic but question-specific response
            h = hashlib.md5(question.encode()).hexdigest()[:8]
            return json.dumps({
                "question": question,
                "findings": [
                    f"Finding 1 [{h}]: Core analysis reveals significant technical depth "
                    f"in the area addressed by this question. Key mechanisms involve "
                    f"layered abstractions and modular design patterns.",
                    f"Finding 2 [{h}]: Recent developments (2024-2025) show accelerating "
                    f"progress with practical implementations moving from research "
                    f"to production environments.",
                    f"Finding 3 [{h}]: Trade-offs exist between performance, cost, and "
                    f"complexity. The optimal approach depends heavily on the specific "
                    f"use case and scale requirements.",
                ],
                "confidence": "medium-high"
            })

        def synthesize_response(findings_text: str) -> str:
            return (
                "## Synthesis\n\n"
                "The research reveals three interconnected themes:\n\n"
                "**1. Technical Foundations:** The domain rests on well-established "
                "principles but continues to evolve rapidly. Core mechanisms involve "
                "layered abstractions that enable both flexibility and performance.\n\n"
                "**2. Practical Applications:** Real-world deployment is accelerating, "
                "with 2024-2025 marking a transition from research prototypes to "
                "production-grade systems. Key sectors include enterprise automation, "
                "developer tooling, and data infrastructure.\n\n"
                "**3. Challenges:** The main tensions are between performance and cost, "
                "between flexibility and reliability, and between rapid iteration and "
                "production stability. No single approach dominates — the right choice "
                "depends on scale, latency requirements, and error tolerance.\n\n"
                "These findings suggest a maturing field with significant near-term "
                "opportunities for practitioners who understand both the theory and "
                "the practical engineering constraints."
            )

        def critique_response(analysis: str) -> str:
            return json.dumps({
                "strengths": [
                    "Clear thematic organization",
                    "Acknowledges trade-offs rather than picking a single 'best' approach",
                    "Grounded in recent developments"
                ],
                "gaps": [
                    "Missing specific quantitative data or benchmarks",
                    "Could benefit from concrete case studies",
                    "Doesn't address the competitive landscape"
                ],
                "suggestions": [
                    "Add 2-3 specific examples or case studies to ground the analysis",
                    "Include quantitative claims where possible (e.g., latency, cost)",
                    "Address who the key players/projects are in this space"
                ],
                "overall_rating": "B+"
            })

        def edit_response(text: str) -> str:
            return (
                "## Final Analysis\n\n"
                "The research reveals three interconnected themes with significant "
                "implications for practitioners:\n\n"
                "**Technical Foundations:** The domain builds on well-established "
                "principles — layered abstractions, modular design, and composable "
                "interfaces — but the pace of evolution is accelerating. For example, "
                "inference latency for production systems has dropped 3-5x in the "
                "past 18 months through architectural innovations.\n\n"
                "**Practical Applications:** 2024-2025 marks the transition from research "
                "to production. Enterprise automation, developer tooling, and data "
                "infrastructure are the primary adoption vectors. Notable case studies "
                "include automated code review pipelines (reducing review time by ~40%) "
                "and intelligent document processing systems.\n\n"
                "**Key Challenges:** Three fundamental tensions shape the design space:\n"
                "- Performance vs. cost (10x compute often yields <2x quality improvement)\n"
                "- Flexibility vs. reliability (general-purpose systems fail unpredictably)\n"
                "- Iteration speed vs. production stability (move fast vs. don't break things)\n\n"
                "**Recommendation:** Practitioners should invest in understanding the "
                "modular building blocks rather than chasing end-to-end solutions. The "
                "most successful deployments compose simple, well-tested components "
                "into reliable pipelines — exactly the prompt chaining pattern this "
                "analysis itself was produced by."
            )

        cls.RESPONSES = {
            "research_planner": plan_response,
            "researcher": research_response,
            "synthesizer": synthesize_response,
            "critic": critique_response,
            "final_editor": edit_response,
        }

    def __init__(self, latency_ms: float = 50.0):
        """Initialize with configurable simulated latency."""
        self.latency_ms = latency_ms
        if not self.RESPONSES:
            self._register_defaults()

    async def call(self, system_prompt: str, user_prompt: str,
                   step_name: str) -> tuple[str, int]:
        """Simulate an LLM API call.

        Returns:
            Tuple of (response_text, token_count)

        In production, this would be:
            response = await client.messages.create(...)
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return response.content[0].text, tokens
        """
        # Simulate network + inference latency
        await asyncio.sleep(self.latency_ms / 1000.0)

        # Get the appropriate response generator
        generator = self.RESPONSES.get(step_name)
        if generator is None:
            return f"[Simulated response for {step_name}]", 100

        response = generator(user_prompt)
        # Rough token estimate: ~4 chars per token (standard approximation)
        token_estimate = (len(system_prompt) + len(user_prompt) + len(response)) // 4

        return response, token_estimate


# ---------------------------------------------------------------------------
# Chain Step — the building block of every chain
# ---------------------------------------------------------------------------

@dataclass
class ChainStep:
    """A single step in a prompt chain.

    Each step encapsulates:
    - A system prompt defining the step's role
    - A formatter that turns the previous step's output into this step's input
    - A validator that checks whether the output is acceptable
    - Retry logic for handling validation failures

    This is the fundamental abstraction. Once you have this, building
    chains is just composing steps together.
    """
    name: str
    system_prompt: str
    # Transforms previous output into this step's user prompt
    input_formatter: Callable[[str], str]
    # Returns (is_valid, error_message)
    output_validator: Callable[[str], tuple[bool, str]] = field(
        default_factory=lambda: lambda x: (True, "")
    )
    max_retries: int = 2

    async def execute(self, llm: SimulatedLLM, input_text: str) -> StepResult:
        """Execute this step with retry logic.

        The retry pattern is important: when a step produces invalid output,
        we DON'T just retry blindly. We include the validation error in the
        retry prompt, giving the LLM specific feedback about what to fix.
        This is much more effective than naive retries.
        """
        formatted_input = self.input_formatter(input_text)
        retries = 0
        last_error = None

        for attempt in range(1 + self.max_retries):
            start = time.perf_counter()

            # On retry, append the validation error to the prompt
            # so the LLM knows what to fix
            prompt = formatted_input
            if last_error:
                prompt += (
                    f"\n\n[RETRY - Previous output was invalid: {last_error}. "
                    f"Please fix and try again.]"
                )

            try:
                output, tokens = await llm.call(
                    self.system_prompt, prompt, self.name
                )
                elapsed_ms = (time.perf_counter() - start) * 1000

                # Validate the output
                is_valid, error_msg = self.output_validator(output)

                if is_valid:
                    return StepResult(
                        step_name=self.name,
                        input_text=formatted_input[:200] + "...",  # Truncate for logging
                        output_text=output,
                        status=StepStatus.RETRIED if retries > 0 else StepStatus.SUCCESS,
                        latency_ms=elapsed_ms,
                        tokens_used=tokens,
                        retries=retries,
                    )
                else:
                    last_error = error_msg
                    retries += 1

            except Exception as e:
                elapsed_ms = (time.perf_counter() - start) * 1000
                last_error = str(e)
                retries += 1

        # All retries exhausted
        return StepResult(
            step_name=self.name,
            input_text=formatted_input[:200] + "...",
            output_text="",
            status=StepStatus.FAILED,
            latency_ms=elapsed_ms,
            tokens_used=0,
            retries=retries,
            error=last_error,
        )


# ---------------------------------------------------------------------------
# Validators — ensure step outputs meet expectations
# ---------------------------------------------------------------------------

def validate_json_with_keys(required_keys: list[str]) -> Callable[[str], tuple[bool, str]]:
    """Create a validator that checks for valid JSON with specific keys.

    This is the most common validator pattern in production chains.
    Steps that produce structured data should always have their output
    validated before passing to the next step.
    """
    def validator(output: str) -> tuple[bool, str]:
        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            return False, f"Invalid JSON: {e}"

        missing = [k for k in required_keys if k not in data]
        if missing:
            return False, f"Missing required keys: {missing}"

        return True, ""
    return validator


def validate_min_length(min_chars: int) -> Callable[[str], tuple[bool, str]]:
    """Ensure output meets a minimum length threshold.

    Useful for steps that should produce substantive content.
    A suspiciously short response usually means the model
    misunderstood the task or refused to engage.
    """
    def validator(output: str) -> tuple[bool, str]:
        if len(output.strip()) < min_chars:
            return False, f"Output too short ({len(output)} chars, minimum {min_chars})"
        return True, ""
    return validator


# ---------------------------------------------------------------------------
# The Research Chain — our concrete pipeline
# ---------------------------------------------------------------------------

def build_research_chain() -> list[ChainStep]:
    """Build the 5-step research and analysis chain.

    Each step has a focused role. Notice how the system prompts are
    specific and directive — vague system prompts are the #1 cause
    of unreliable chain steps.
    """

    # Step 1: Research Planner
    # Takes a topic, outputs structured research questions
    planner = ChainStep(
        name="research_planner",
        system_prompt=(
            "You are a research planning assistant. Given a topic, generate "
            "exactly 3-5 specific, answerable research questions that would "
            "provide comprehensive understanding of the topic. Output ONLY "
            "valid JSON with a 'questions' key containing a list of strings. "
            "No other text."
        ),
        input_formatter=lambda topic: f"Generate research questions for: {topic}",
        output_validator=validate_json_with_keys(["questions"]),
    )

    # Step 2: Researcher
    # Takes a single question, outputs findings
    # (This step is called multiple times in parallel — one per question)
    researcher = ChainStep(
        name="researcher",
        system_prompt=(
            "You are a thorough researcher. Given a specific question, provide "
            "detailed findings based on your knowledge. Output ONLY valid JSON "
            "with keys: 'question' (string), 'findings' (list of strings), "
            "'confidence' (string: low/medium/medium-high/high). No other text."
        ),
        input_formatter=lambda question: f"Research this question: {question}",
        output_validator=validate_json_with_keys(["question", "findings", "confidence"]),
    )

    # Step 3: Synthesizer
    # Takes all research findings, produces a coherent analysis
    synthesizer = ChainStep(
        name="synthesizer",
        system_prompt=(
            "You are an analytical synthesizer. Given multiple research findings, "
            "combine them into a coherent, well-structured analysis. Identify "
            "themes, connections, and implications. Output markdown-formatted text."
        ),
        input_formatter=lambda findings: (
            f"Synthesize these research findings into a coherent analysis:\n\n{findings}"
        ),
        output_validator=validate_min_length(100),
    )

    # Step 4: Critic
    # Reviews the analysis for quality
    critic = ChainStep(
        name="critic",
        system_prompt=(
            "You are a critical reviewer. Analyze the given text for: strengths, "
            "gaps in reasoning, unsupported claims, and areas for improvement. "
            "Output ONLY valid JSON with keys: 'strengths' (list), 'gaps' (list), "
            "'suggestions' (list), 'overall_rating' (string). No other text."
        ),
        input_formatter=lambda analysis: f"Critically review this analysis:\n\n{analysis}",
        output_validator=validate_json_with_keys(
            ["strengths", "gaps", "suggestions", "overall_rating"]
        ),
    )

    # Step 5: Final Editor
    # Incorporates critic feedback and produces polished output
    editor = ChainStep(
        name="final_editor",
        system_prompt=(
            "You are a senior editor. Given an analysis and editorial feedback, "
            "produce a polished final version that addresses the feedback. "
            "Maintain the analytical depth while improving clarity and structure. "
            "Output the final analysis in clean markdown."
        ),
        input_formatter=lambda text: text,  # Already formatted by the chain runner
        output_validator=validate_min_length(200),
    )

    return [planner, researcher, synthesizer, critic, editor]


# ---------------------------------------------------------------------------
# Chain Runner — orchestrates the full pipeline
# ---------------------------------------------------------------------------

class ChainRunner:
    """Orchestrates execution of a prompt chain.

    This is the engine that runs the pipeline. It handles:
    - Sequential step execution with context passing
    - Parallel execution for independent sub-tasks (e.g., researching
      multiple questions simultaneously)
    - Logging and metrics collection
    - Error propagation (if a step fails, the chain stops)

    Design decision: we pass ALL accumulated context to each step rather
    than only the previous step's output. This costs more tokens but is
    more robust — the synthesizer benefits from seeing the original topic,
    not just the raw findings. In production, you'd tune this per-step.
    """

    def __init__(self, llm: Optional[SimulatedLLM] = None):
        self.llm = llm or SimulatedLLM()
        self.steps: list[ChainStep] = []
        self.results: list[StepResult] = []

    async def run(self, topic: str, steps: list[ChainStep]) -> ChainResult:
        """Execute the research chain on a given topic.

        This method implements the specific orchestration logic for our
        research pipeline. A more general chain runner would use a
        configuration object to define the flow, but explicit orchestration
        is clearer for learning.
        """
        self.steps = steps
        self.results = []
        chain_start = time.perf_counter()

        print(f"\n{'=' * 60}")
        print(f"CHAIN START: Researching '{topic}'")
        print(f"{'=' * 60}")

        # Step 1: Plan research questions
        planner = steps[0]
        print(f"\n>>> Step 1: {planner.name}")
        plan_result = await planner.execute(self.llm, topic)
        self.results.append(plan_result)

        if plan_result.status == StepStatus.FAILED:
            return self._build_result("Chain failed at planning step", chain_start)

        # Parse the questions from the planner's output
        questions = json.loads(plan_result.output_text)["questions"]
        print(f"    Generated {len(questions)} research questions")
        for i, q in enumerate(questions, 1):
            print(f"    Q{i}: {q}")

        # Step 2: Research each question IN PARALLEL
        # This is a key optimization — independent sub-tasks run concurrently.
        # With 3 questions and 500ms per API call, sequential = 1500ms,
        # parallel = ~500ms. In production with real APIs, this matters a lot.
        researcher = steps[1]
        print(f"\n>>> Step 2: {researcher.name} (parallel, {len(questions)} calls)")

        research_tasks = [
            researcher.execute(self.llm, question)
            for question in questions
        ]
        research_results = await asyncio.gather(*research_tasks)
        self.results.extend(research_results)

        # Check for failures
        failed = [r for r in research_results if r.status == StepStatus.FAILED]
        if failed:
            print(f"    WARNING: {len(failed)} research calls failed")

        # Collect successful findings
        all_findings = []
        for result in research_results:
            if result.status != StepStatus.FAILED:
                findings_data = json.loads(result.output_text)
                all_findings.append(findings_data)
                print(f"    Researched: {findings_data['question'][:60]}... "
                      f"({len(findings_data['findings'])} findings, "
                      f"confidence: {findings_data['confidence']})")

        if not all_findings:
            return self._build_result("All research calls failed", chain_start)

        # Format findings for the synthesizer
        findings_text = json.dumps(all_findings, indent=2)

        # Step 3: Synthesize
        synthesizer = steps[2]
        print(f"\n>>> Step 3: {synthesizer.name}")
        synth_result = await synthesizer.execute(self.llm, findings_text)
        self.results.append(synth_result)

        if synth_result.status == StepStatus.FAILED:
            return self._build_result("Chain failed at synthesis step", chain_start)

        print(f"    Synthesis complete ({len(synth_result.output_text)} chars)")

        # Step 4: Critique
        critic = steps[3]
        print(f"\n>>> Step 4: {critic.name}")
        critic_result = await critic.execute(self.llm, synth_result.output_text)
        self.results.append(critic_result)

        if critic_result.status == StepStatus.FAILED:
            return self._build_result("Chain failed at critique step", chain_start)

        critique = json.loads(critic_result.output_text)
        print(f"    Rating: {critique['overall_rating']}")
        print(f"    Strengths: {len(critique['strengths'])}, "
              f"Gaps: {len(critique['gaps'])}, "
              f"Suggestions: {len(critique['suggestions'])}")

        # Step 5: Final edit — combine the synthesis and critique feedback
        editor = steps[4]
        print(f"\n>>> Step 5: {editor.name}")

        # This is the context assembly step — we give the editor both
        # the original analysis AND the critique so it can improve the text
        editor_input = (
            f"Original analysis:\n{synth_result.output_text}\n\n"
            f"Editorial feedback:\n{json.dumps(critique, indent=2)}\n\n"
            f"Please produce the final polished version."
        )
        edit_result = await editor.execute(self.llm, editor_input)
        self.results.append(edit_result)

        if edit_result.status == StepStatus.FAILED:
            return self._build_result("Chain failed at editing step", chain_start)

        print(f"    Final output ready ({len(edit_result.output_text)} chars)")

        return self._build_result(edit_result.output_text, chain_start)

    def _build_result(self, final_output: str, start_time: float) -> ChainResult:
        """Assemble the chain result with aggregate metrics."""
        total_ms = (time.perf_counter() - start_time) * 1000
        total_tokens = sum(r.tokens_used for r in self.results)
        return ChainResult(
            final_output=final_output,
            steps=self.results,
            total_latency_ms=total_ms,
            total_tokens=total_tokens,
        )


# ---------------------------------------------------------------------------
# Main — demonstrate the full pipeline
# ---------------------------------------------------------------------------

async def main():
    """Run the research chain and display results."""
    # Build the chain
    steps = build_research_chain()
    print(f"Built chain with {len(steps)} steps:")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step.name} (max_retries={step.max_retries})")

    # Create the runner with simulated LLM
    # In production: runner = ChainRunner(llm=RealClaudeClient(api_key=...))
    runner = ChainRunner(llm=SimulatedLLM(latency_ms=30))

    # Execute the chain
    topic = "prompt chaining in production AI systems"
    result = await runner.run(topic, steps)

    # Display results
    print("\n")
    print(result.summary())

    print("\n" + "=" * 60)
    print("FINAL OUTPUT")
    print("=" * 60)
    print(result.final_output)

    # Show per-step breakdown
    print("\n" + "=" * 60)
    print("STEP-BY-STEP DETAILS")
    print("=" * 60)
    for i, step in enumerate(result.steps, 1):
        print(f"\n--- Step {i}: {step.step_name} ---")
        print(f"Status: {step.status.value}")
        print(f"Latency: {step.latency_ms:.1f}ms")
        print(f"Tokens: {step.tokens_used}")
        print(f"Output preview: {step.output_text[:150]}...")

    # Demonstrate the chain's composability by showing how easy it is
    # to swap or add steps
    print("\n" + "=" * 60)
    print("ARCHITECTURE NOTES")
    print("=" * 60)
    print("""
To use with real Claude API, replace SimulatedLLM with:

    import anthropic

    class ClaudeLLM:
        def __init__(self):
            self.client = anthropic.AsyncAnthropic()

        async def call(self, system_prompt, user_prompt, step_name):
            response = await self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}]
            )
            text = response.content[0].text
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return text, tokens

    runner = ChainRunner(llm=ClaudeLLM())

The chain logic remains identical — only the LLM backend changes.
""")


if __name__ == "__main__":
    asyncio.run(main())
