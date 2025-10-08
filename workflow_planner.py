import os
import json
import datetime
from crewai import Agent, Task, Crew
import google.generativeai as genai
from crewai.llm import LLM
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set the Google API key
os.environ["GOOGLE_API_KEY"] = "AIzaSyBQ727UMPalA1PLVaje0seFTldOV6z8vB0"

# Configure the Google Generative AI SDK
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

class GeminiLLM(LLM):
    """Custom LLM class for Google Gemini integration with CrewAI"""

    def __init__(self, model="gemini-2.0-flash", temperature=0.7, **kwargs):
        super().__init__(model=model, **kwargs)
        self.model_name = model
        self.temperature = temperature
        self.gemini_model = genai.GenerativeModel(model)

    def call(self, messages, **kwargs):
        """Convert CrewAI messages to Gemini format and get response"""
        # Convert messages to Gemini format
        prompt = ""
        for msg in messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                # Handle string messages
                content = str(msg)
                role = "user"

            if role == "system":
                prompt = f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"

        # Generate response
        response = self.gemini_model.generate_content(
            prompt.strip(),
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=2000,
            )
        )

        return response.text.strip()

class ProblemUnderstandingAgent:
    """Layer 1: Problem Understanding Agent (Intent Extraction Layer)"""

    def __init__(self):
        self.gemini_llm = GeminiLLM(model="gemini-2.0-flash", temperature=0.3)  # Lower temperature for analytical output
        self.setup_agent()

    def setup_agent(self):
        """Initialize the problem understanding agent"""
        self.understanding_agent = Agent(
            role="Problem Understanding Agent",
            goal="Extract and structure the underlying intent from problem statements without providing solutions",
            backstory="""
You are an expert problem analyst trained to extract structure and intent from user problem statements
across business, operational, and technical domains. Your mission is to understand, not to solve.
You identify goals, entities, processes, pain points, and implicit or explicit constraints with
precision and neutrality. You never suggest tools or implementation methods.

When the user's input is ambiguous or incomplete, you explicitly surface those gaps as 'unknowns'
and generate clarifying questions to improve understanding. Your goal is to achieve a high-clarity,
structured representation of the problem that can serve as input for workflow planning or system design.

You adapt to the user's context and communication style — whether technical or non-technical —
and always respond in a neutral, concise, and analytical tone optimized for documentation.
""",
            llm=self.gemini_llm,
            verbose=False,  # Less verbose for clean output
            allow_delegation=False
        )

    def infer_fallback_questions(self, intent_json):
        """Generate smart fallback clarifying questions when model returns none."""
        fallback_questions = []

        # 1. Trigger Event - ALWAYS ask if missing
        if not intent_json.get("trigger_event") or "unspecified" in str(intent_json.get("trigger_event", "")).lower() or "unclear" in str(intent_json.get("trigger_event", "")).lower():
            fallback_questions.append(("trigger_event", "What should trigger or start this process? (e.g., when a file is uploaded, daily schedule, manual click)"))

        # 2. Input Sources - Critical for workflow design
        if not intent_json.get("inputs") or len(intent_json["inputs"]) == 0:
            fallback_questions.append(("inputs", "What does your process start with — files, messages, forms, or API data?"))

        # 3. Output Destination - Essential for workflow completion
        if not intent_json["entities"].get("destination") or "unspecified" in str(intent_json["entities"]["destination"]).lower():
            fallback_questions.append(("outputs", "Where should the final output or results be stored or sent?"))

        # 4. Success Criteria / Output Format
        if not intent_json.get("outputs") or len(intent_json["outputs"]) == 0:
            fallback_questions.append(("outputs", "What does a successful result look like — a file, a report, or a message?"))

        # 5. Current State - How it's done today (improved logic)
        if not intent_json.get("current_state") or any(x in str(intent_json.get("current_state", "")).lower() for x in ["unspecified", "unknown", "unclear", "not mentioned"]):
            fallback_questions.append(("current_state", "How is this process currently handled? (manually, semi-automated, or not at all?)"))

        # 6. Constraints - Design feasibility limits
        if not intent_json.get("constraints") or len(intent_json["constraints"]) == 0:
            fallback_questions.append(("constraints", "Are there any limitations or rules to follow — timing, tools, compliance, or cost?"))

        # 7. Pain Points - Why automate this
        if not intent_json.get("pain_points") or len(intent_json["pain_points"]) == 0:
            fallback_questions.append(("pain_points", "What's frustrating or slow about how you do this today?"))

        # 8. Not-Negotiables - Absolute requirements
        if not intent_json.get("not_negotiable") or len(intent_json["not_negotiable"]) == 0:
            fallback_questions.append(("not_negotiable", "Are there any parts of this process that must stay exactly the same?"))

        # 9. Volume/Frequency - Scale requirements (improved deterministic logic)
        if not intent_json.get("volume") and "volume" not in json.dumps(intent_json).lower():
            fallback_questions.append(("volume", "How often or how many times per month does this process happen?"))

        # 10. User Role/Actor - Who performs this (improved deterministic logic)
        if not intent_json.get("user_role") and not any(word in json.dumps(intent_json).lower() for word in ["user", "role", "actor", "person", "team", "department"]):
            fallback_questions.append(("user_role", "Who usually performs or initiates this process?"))

        # Prioritize questions by importance
        priority_order = [
            "trigger_event", "inputs", "outputs", "current_state",
            "pain_points", "constraints", "not_negotiable", "volume", "user_role"
        ]

        # Sort by priority and return top 6
        fallback_questions.sort(key=lambda q: next((i for i, p in enumerate(priority_order) if p == q[0]), 99))
        return [q[1] for q in fallback_questions[:6]]

    def generate_clarifying_questions(self, intent_json: dict) -> list[str]:
        """Generate smart clarifying questions from unknowns in the intent."""
        unknowns = intent_json.get("unknowns", [])
        if not unknowns:
            return []

        clarification_prompt = f"""
Based on the following unknowns about the user's problem:
{json.dumps(unknowns, indent=2)}

Generate 3 to 5 short, specific, user-friendly questions to clarify these unknowns.
Do NOT restate the unknowns verbatim. Each question should be actionable and easy to answer in one sentence.

Problem context: {intent_json.get("summary", "")}

Return them as a numbered list (no extra text).
Example format:
1. What specific data do you need to extract?
2. How often should this process run?
3. What format should the output be in?
"""

        try:
            # Use Gemini directly for question generation
            response = self.gemini_llm.call([{"role": "user", "content": clarification_prompt}])

            # Parse the numbered list into individual questions
            questions = []
            for line in str(response).split('\n'):
                line = line.strip()
                if line and (line[0].isdigit() or line.startswith('-')):
                    # Remove numbering and clean up
                    question = line.lstrip('0123456789.- ').strip()
                    if question and len(question) > 10:  # Filter out too short questions
                        questions.append(question)

            # Limit to 5 questions max
            return questions[:5] if questions else []

        except Exception as e:
            # Fallback questions based on common unknowns
            fallback_questions = []
            for unknown in unknowns[:3]:  # Limit to 3
                if "specific" in unknown.lower() or "what" in unknown.lower():
                    fallback_questions.append(f"Can you specify what {unknown.lower().replace('specific ', '').replace('what ', '')} you need?")
                elif "how" in unknown.lower():
                    fallback_questions.append(f"Can you describe {unknown.lower()}?")
                elif "frequency" in unknown.lower() or "often" in unknown.lower():
                    fallback_questions.append("How often should this process run?")
                else:
                    fallback_questions.append(f"Can you provide more details about: {unknown}")

            return fallback_questions

    def extract_intent(self, problem_statement: str) -> dict:
        """Extract structured intent from a problem statement with dynamic clarification"""

        understanding_prompt = f"""
You are the Problem Understanding Agent.

Your goal is to understand, not to solve.
Given the user's problem statement, extract and structure the underlying intent.

Problem Statement: "{problem_statement}"

Return only a structured JSON with these exact fields:
{{
  "goal": "Brief, clear statement of what the user wants to achieve",
  "problem_type": "workflow | data task | single action | unclear",
  "trigger_event": "What initiates this process (e.g., user action, file upload, schedule, system trigger)",
  "inputs": ["List of input sources or data types"],
  "outputs": ["List of desired outputs or destinations"],
  "volume": "How often or how many times this process occurs (if mentioned)",
  "user_role": "Who performs or initiates this process (if mentioned)",
  "success_criteria": "What defines successful completion of this process",
  "entities": {{
    "action": "Primary action being performed",
    "sources": "Where data/process comes from",
    "destination": "Where results go",
    "tools": "Any mentioned tools or systems",
    "data_types": "Types of data involved"
  }},
  "current_state": "How the process works now, if mentioned",
  "pain_points": ["List of challenges or frustrations mentioned"],
  "constraints": ["List of limitations or requirements mentioned"],
  "not_negotiable": ["List of absolute requirements that cannot be compromised"],
  "clarity_level": "high | medium | low",
  "unknowns": ["List of missing information needed for clarity"],
  "summary": "One-sentence summary of the core intent"
}}

Be sure to include 'trigger_event' even if it's inferred.
For example:
- "User uploads a file"
- "Daily at 6 PM"
- "When a form is submitted"
- "When an email arrives"
If no explicit event is found, infer the most likely one based on context.

Do NOT suggest tools, solutions, or roadmaps.
Keep tone neutral and analytical.
Return ONLY the JSON object, no additional text.

After forming the JSON, verify internally that:
- Each field is filled as completely as possible based on the input.
- 'current_state' reflects any described manual or existing process.
- 'constraints' include both explicit (mentioned) and implicit (logical) limitations.
- 'pain_points' include any friction points mentioned or implied.
- 'unknowns' only include questions that are genuinely unclear from the user's input.

Ensure all string values are specific and contextual — avoid placeholders like "unspecified" unless truly no data exists.
"""

        task = Task(
            description=understanding_prompt,
            agent=self.understanding_agent,
            expected_output="Structured JSON object containing intent analysis"
        )

        # Execute the understanding task
        crew = Crew(
            agents=[self.understanding_agent],
            tasks=[task],
            verbose=False
        )

        result = crew.kickoff()

        # Parse the JSON result
        try:
            import json
            # Clean the result to extract just the JSON
            result_str = str(result).strip()

            # Find JSON object in the result
            start_idx = result_str.find('{')
            end_idx = result_str.rfind('}') + 1

            if start_idx != -1 and end_idx != -1:
                json_str = result_str[start_idx:end_idx]
                intent_json = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")

        except Exception as e:
            # Fallback response
            intent_json = {
                "goal": f"Error processing: {problem_statement}",
                "problem_type": "unclear",
                "trigger_event": "Unclear — likely manual user initiation",
                "inputs": [],
                "outputs": [],
                "entities": {},
                "current_state": "Unknown",
                "pain_points": [],
                "constraints": [],
                "not_negotiable": [],
                "clarity_level": "low",
                "unknowns": [f"Processing error: {str(e)}"],
                "summary": f"Failed to analyze: {problem_statement}",
                "metadata": {
                    "version": "1.0",
                    "agent_type": "ProblemUnderstandingAgent",
                    "last_updated": datetime.datetime.now().isoformat(),
                    "essential_fields_verified": [],
                    "missing_fields_prompted": ["trigger_event", "inputs", "outputs", "constraints", "pain_points", "not_negotiable", "current_state", "volume", "user_role"],
                    "clarity_confidence": "low",
                    "layer_ready_for": "workflow_planner",
                    "needs_refinement": True
                }
            }

        # --- 🧠 Dead-Important Field Enforcement ---
        critical_fields = {
            "trigger_event": "What should trigger or start this process? (e.g., file upload, scheduled time, or user action?)",
            "inputs": "What does your process start with — files, forms, messages, or API data?",
            "outputs": "What does a successful result look like — a file, report, or message?",
            "current_state": "How is this process currently handled? (manually, semi-automated, or not at all?)",
            "pain_points": "What's frustrating or slow about how you do this today?",
            "constraints": "Are there any limitations or rules to follow — timing, tools, compliance, or cost?",
            "not_negotiable": "Are there any parts of this process that must stay exactly the same?",
            "user_role": "Who usually performs or initiates this process?",
            "volume": "How often or how many times per month does this process happen?"
        }

        # Detect missing or vague critical fields
        missing_critical_questions = []
        for field, question in critical_fields.items():
            value = intent_json.get(field)
            # If field missing, empty, or unclear → ask
            if not value or str(value).strip().lower() in ["", "unspecified", "unknown", "unclear", "not mentioned", "none"]:
                missing_critical_questions.append(question)

        # If Gemini said clarity is high but skipped criticals → downgrade and ask
        originally_high_clarity = intent_json.get("clarity_level", "").lower() == "high"
        if missing_critical_questions:
            if originally_high_clarity:
                intent_json["clarity_level"] = "medium"
            print("\n⚙️ Some critical details are missing — let's clarify those first.")

        # Auto trigger clarification - too important to leave to inference
        if not intent_json.get("trigger_event") or "unspecified" in str(intent_json.get("trigger_event", "")).lower() or "unclear" in str(intent_json.get("trigger_event", "")).lower():
            print("\n⚙️ I need one key clarification before continuing.")
            print("What should trigger or start this process?")
            print("Examples: when a file is uploaded, every morning at 9 AM, when a new email arrives, or when you click a button.")
            user_trigger = input("> ").strip()
            if user_trigger:
                intent_json["trigger_event"] = user_trigger
            else:
                intent_json["trigger_event"] = "Unclear — likely manual user initiation or file upload"

        # Layer 1B: Dynamic Clarification Loop
        unknowns = intent_json.get("unknowns", [])
        questions = self.generate_clarifying_questions(intent_json)

        # --- Merge all question sources ---
        if not questions and (not unknowns or len(unknowns) == 0):
            print("\n⚙️ No clarifying questions detected from Gemini — inferring fallback ones...")
            questions = self.infer_fallback_questions(intent_json)

        # Merge critical missing questions if any
        if missing_critical_questions:
            questions = list(dict.fromkeys(missing_critical_questions + questions))  # Deduplicate while preserving order

        # Step 4: Proceed only if we have *any* questions now
        if questions:
            print("\n🤔 I need a few clarifications to understand better:")
            user_answers = {}

            # Ask questions interactively
            for i, question in enumerate(questions, 1):
                print(f"\n{i}. {question}")
                answer = input("> ").strip()
                user_answers[question] = answer

            # Update the JSON with clarifications
            intent_json["clarifications"] = user_answers

            # Check if all critical fields are now populated - if so, upgrade clarity
            critical_fields_filled = all(
                intent_json.get(field) and str(intent_json.get(field)).strip().lower() not in ["", "unspecified", "unknown", "unclear", "not mentioned", "none"]
                for field in critical_fields.keys()
            )

            # Upgrade clarity back to high if originally high and now all criticals are filled
            if originally_high_clarity and critical_fields_filled:
                intent_json["clarity_level"] = "high"
                clarity_confidence = "high"  # Restore confidence since all criticals are now filled
            else:
                intent_json["clarity_level"] = "high"  # Still improved with clarifications

            # Clear unknowns since we've addressed them
            intent_json["unknowns"] = []

            # Refinement Step: Re-evaluate and update fields based on clarifications
            if user_answers:
                refinement_prompt = f"""
You are the Problem Understanding Agent.

Refine the following intent analysis using the clarifications provided.
Improve or complete fields like 'current_state', 'pain_points', 'constraints', 'not_negotiable', and 'summary'
based on the clarified understanding.

If constraints are not explicitly mentioned, infer implicit ones — such as:
- Legal or regulatory obligations
- Data privacy and security needs
- System or integration limits
- Accuracy or performance requirements

For 'not_negotiable' items, identify absolute requirements that cannot be compromised — such as:
- Mandatory legal compliance requirements
- Critical security or privacy mandates
- Fixed budget limitations
- Essential regulatory standards
- Non-negotiable business rules

Do not invent unrelated details. Keep tone analytical and factual.

Original Problem Statement:
{problem_statement}

Clarifications:
{json.dumps(user_answers, indent=2)}

Existing Intent JSON:
{json.dumps(intent_json, indent=2)}

Return ONLY the updated JSON object, no explanations.

Before finalizing the JSON, review whether the clarifications change the problem classification
(e.g., from 'data task' to 'workflow'). If so, update the 'problem_type' field accordingly.
Also ensure that inferred constraints or not_negotiable items align with real-world logic
(e.g., legal, performance, or privacy-related considerations).
"""

                try:
                    refined_response = self.gemini_llm.call([{"role": "user", "content": refinement_prompt}])
                    start_idx = refined_response.find('{')
                    end_idx = refined_response.rfind('}') + 1
                    if start_idx != -1 and end_idx != -1:
                        refined_json = json.loads(refined_response[start_idx:end_idx])
                        intent_json = refined_json  # overwrite with refined version
                except Exception as e:
                    print(f"⚠️ Refinement failed: {e}. Keeping original JSON.")

        else:
            print("\n✅ Gemini captured all context — no further clarifications needed.")

        # Add automatic inferred pain points if missing
        if not intent_json.get("pain_points") or len(intent_json["pain_points"]) < 2:
            intent_json["pain_points"] = list(set(intent_json.get("pain_points", []) + [
                "Manual effort leads to delays or errors.",
                "Lack of automation increases operational cost.",
                "No standardized process for handling inputs and outputs."
            ]))[:3]

        # Add metadata for versioning and traceability
        essential_fields = ["trigger_event", "inputs", "outputs", "constraints", "pain_points", "not_negotiable", "current_state", "volume", "user_role"]
        verified_fields = [field for field in essential_fields if intent_json.get(field)]
        missing_fields = [field for field in essential_fields if not intent_json.get(field)]

        # Determine clarity confidence (can be upgraded after clarifications)
        originally_high_clarity = intent_json.get("clarity_level", "").lower() == "high"
        clarity_confidence = "high" if intent_json.get("clarity_level") == "high" else "medium"
        if missing_critical_questions:
            clarity_confidence = "medium"

        intent_json["metadata"] = {
            "version": "1.0",
            "agent_type": "ProblemUnderstandingAgent",
            "last_updated": datetime.datetime.now().isoformat(),
            "essential_fields_verified": verified_fields,
            "missing_fields_prompted": missing_fields,
            "clarity_confidence": clarity_confidence,
            "layer_ready_for": "workflow_planner",
            "needs_refinement": False if intent_json.get("clarity_level") == "high" and not missing_critical_questions else True
        }

        return intent_json


def main():
    """Main function to run the Problem Understanding Agent (Layer 1)"""
    print("🧠 Welcome to the AI Problem Understanding Assistant!")
    print("This tool analyzes your problem statements and extracts structured intent.\n")

    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ GOOGLE_API_KEY environment variable not found!")
        print("Please set your Google API key:")
        print("export GOOGLE_API_KEY='your_key_here'")
        return

    try:
        # Step 1: Initial problem understanding (Layer 1)
        print("🧠 LAYER 1: PROBLEM UNDERSTANDING")
        print("-" * 40)

        initial_problem = input("Describe your problem or goal briefly:\n> ").strip()

        if not initial_problem:
            print("❌ No problem statement provided. Exiting.")
            return

        # Initialize understanding agent
        understanding_agent = ProblemUnderstandingAgent()

        print("\n🔍 Analyzing your problem statement...")
        intent_analysis = understanding_agent.extract_intent(initial_problem)

        # Display the structured understanding
        print("\n📊 PROBLEM ANALYSIS RESULTS:")
        print("=" * 50)
        import json
        print(json.dumps(intent_analysis, indent=2))
        print("=" * 50)

        # Display unknowns if any
        unknowns = intent_analysis.get("unknowns", [])
        if unknowns:
            print(f"\n📋 IDENTIFIED UNKNOWNS ({len(unknowns)}):")
            for i, unknown in enumerate(unknowns, 1):
                print(f"   {i}. {unknown}")

        print(f"\n🎯 CLARITY LEVEL: {intent_analysis.get('clarity_level', 'unknown').upper()}")

        # Layer 1 is complete - just understanding, no workflow planning
        print("\n💡 UNDERSTANDING COMPLETE")
        print("Layer 1 analysis finished. Your intent has been extracted and structured.")
        print("This understanding can be used for any subsequent analysis or planning.")

        # Option to save the intent analysis
        save_intent = input("\n💾 Would you like to save this intent analysis? (y/n): ").lower().strip()
        if save_intent == 'y':
            filename = input("Enter filename (default: intent_analysis.json): ").strip()
            if not filename:
                filename = "intent_analysis.json"

            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(intent_analysis, f, indent=2)

            print(f"✅ Intent analysis saved to {filename}")

        print("\n🙏 Thank you for using the Workflow Planning Assistant!")
        print("Your analysis is complete!")

    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        print("Please check your API key and internet connection, then try again.")

if __name__ == "__main__":
    main()
