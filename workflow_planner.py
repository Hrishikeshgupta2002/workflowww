import os
import json
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
You are an expert problem analyst trained to extract structure and intent from user problem statements.
Your mission is to understand, not to solve. You identify goals, entities, processes, pain points, and
implicit constraints with precision and neutrality. You never suggest tools or implementation methods.
Your focus is on structured comprehension and clarification — surfacing unknowns and enabling refinement
until a high-clarity representation of the problem is achieved.
""",
            llm=self.gemini_llm,
            verbose=False,  # Less verbose for clean output
            allow_delegation=False
        )

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
  "inputs": ["List of input sources or data types"],
  "outputs": ["List of desired outputs or destinations"],
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
  "clarity_level": "high | medium | low",
  "unknowns": ["List of missing information needed for clarity"],
  "summary": "One-sentence summary of the core intent"
}}

Do NOT suggest tools, solutions, or roadmaps.
Keep tone neutral and analytical.
Return ONLY the JSON object, no additional text.
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
                "inputs": [],
                "outputs": [],
                "entities": {},
                "current_state": "Unknown",
                "pain_points": [],
                "constraints": [],
                "clarity_level": "low",
                "unknowns": [f"Processing error: {str(e)}"],
                "summary": f"Failed to analyze: {problem_statement}"
            }

        # Layer 1B: Dynamic Clarification Loop
        unknowns = intent_json.get("unknowns", [])
        if unknowns and intent_json.get("clarity_level", "low") != "high":
            print("\n🤔 I need a few clarifications to understand better:")

            # Generate clarifying questions
            questions = self.generate_clarifying_questions(intent_json)

            if questions:
                user_answers = {}

                # Ask questions interactively
                for i, question in enumerate(questions, 1):
                    print(f"\n{i}. {question}")
                    answer = input("> ").strip()
                    user_answers[question] = answer

                # Update the JSON with clarifications
                intent_json["clarifications"] = user_answers
                intent_json["clarity_level"] = "high"  # Improved with clarifications

                # Clear unknowns since we've addressed them
                intent_json["unknowns"] = []

                # Refinement Step: Re-evaluate and update fields based on clarifications
                if user_answers:
                    refinement_prompt = f"""
You are the Problem Understanding Agent.

Refine the following intent analysis using the clarifications provided.
Improve or complete fields like 'current_state', 'pain_points', 'constraints', and 'summary'
based on the clarified understanding.

If constraints are not explicitly mentioned, infer implicit ones — such as:
- Legal or regulatory obligations
- Data privacy and security needs
- System or integration limits
- Accuracy or performance requirements

Do not invent unrelated details. Keep tone analytical and factual.

Original Problem Statement:
{problem_statement}

Clarifications:
{json.dumps(user_answers, indent=2)}

Existing Intent JSON:
{json.dumps(intent_json, indent=2)}

Return ONLY the updated JSON object, no explanations.
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
                print("No clarifying questions needed.")

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
