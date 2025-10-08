# Using CrewAI with Google Gemini (AI Studio)

A comprehensive guide to integrating Google Gemini models with CrewAI for building powerful AI agent workflows using the Google AI Studio API.

## 🎯 Overview

This guide demonstrates how to use **Google Gemini** models (via Google AI Studio API) with **CrewAI** to create multi-agent AI workflows. Unlike Vertex AI integrations, this approach uses the direct Google AI Studio REST API through the `google-generativeai` Python SDK.

## 📋 Prerequisites

- Python 3.10 or higher
- Google AI Studio API key
- Basic understanding of CrewAI concepts (Agents, Tasks, Crews)

## 🛠️ Installation

### 1. Install Required Packages

```bash
pip install crewai google-generativeai python-dotenv
```

### 2. Get Your Google AI Studio API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key (it will look like: `AIzaSy...`)

### 3. Set Up Environment Variables

Create a `.env` file in your project directory:

```bash
# .env
GOOGLE_API_KEY=your_api_key_here
```

Or set the environment variable directly:

**Windows (PowerShell):**
```powershell
$env:GOOGLE_API_KEY = "your_api_key_here"
```

**Windows (Command Prompt):**
```cmd
set GOOGLE_API_KEY=your_api_key_here
```

## 🔧 Core Implementation

### Custom Gemini LLM Class

Create a custom LLM class that integrates Gemini with CrewAI:

```python
import os
from crewai import Agent, Task, Crew
import google.generativeai as genai
from crewai.llm import LLM

# Configure Google AI SDK
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
```

## 🚀 Basic Usage

### Simple Single Agent Example

```python
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize Gemini LLM
gemini_llm = GeminiLLM(model="gemini-2.0-flash", temperature=0.7)

# Create an agent
researcher = Agent(
    role="Research Analyst",
    goal="Analyze and summarize information on given topics",
    backstory="You are an expert research analyst with years of experience in data analysis and information synthesis.",
    llm=gemini_llm,
    verbose=True
)

# Create a task
task = Task(
    description="Research and summarize the benefits of renewable energy",
    agent=researcher,
    expected_output="A comprehensive summary of renewable energy benefits"
)

# Create and run the crew
crew = Crew(
    agents=[researcher],
    tasks=[task],
    verbose=True
)

result = crew.kickoff()
print(result)
```

## 🏗️ Advanced Multi-Agent Workflow

### Complete Agent Team Setup

```python
class GeminiCrewAgent:
    def __init__(self):
        self.gemini_llm = GeminiLLM(model="gemini-2.0-flash", temperature=0.7)
        self.setup_agents()

    def setup_agents(self):
        """Initialize Crew AI agents with Gemini LLM"""

        # Research Analyst Agent
        self.research_analyst = Agent(
            role="Research Analyst",
            goal="Gather and analyze information to support content creation",
            backstory="""You are a meticulous research analyst who excels at finding
            relevant information, analyzing trends, and providing data-driven insights.""",
            llm=self.gemini_llm,
            verbose=True,
            allow_delegation=False
        )

        # Content Creator Agent
        self.content_creator = Agent(
            role="Content Creator",
            goal="Create engaging and informative content",
            backstory="""You are an experienced content creator who writes compelling
            articles and educational content.""",
            llm=self.gemini_llm,
            verbose=True,
            allow_delegation=False
        )

        # Editor Agent
        self.editor = Agent(
            role="Content Editor",
            goal="Review and polish content for clarity and accuracy",
            backstory="""You are a professional editor with a keen eye for detail.""",
            llm=self.gemini_llm,
            verbose=True,
            allow_delegation=False
        )

    def create_workflow(self, topic: str):
        """Create a complete content creation workflow"""

        # Research task
        research_task = Task(
            description=f"""Research key facts, statistics, and trends about {topic}.
            Provide data-driven insights that can support content creation.""",
            agent=self.research_analyst,
            expected_output="Research findings and key insights"
        )

        # Content creation task (depends on research)
        content_task = Task(
            description=f"""Create a comprehensive article about {topic}.
            Include introduction, key points, examples, and conclusion.
            Use the research provided to support your content.""",
            agent=self.content_creator,
            context=[research_task],  # Depends on research task
            expected_output="A well-structured article"
        )

        # Editing task (depends on content creation)
        editing_task = Task(
            description=f"""Review and edit the content about {topic}.
            Ensure it's well-structured, engaging, and error-free.""",
            agent=self.editor,
            context=[content_task],  # Depends on content task
            expected_output="Polished final content"
        )

        return [research_task, content_task, editing_task]

    def run_workflow(self, topic: str):
        """Execute the complete agent workflow"""
        tasks = self.create_workflow(topic)

        crew = Crew(
            agents=[self.research_analyst, self.content_creator, self.editor],
            tasks=tasks,
            verbose=True
        )

        return crew.kickoff()
```

### Usage Example

```python
# Initialize the agent system
agent_system = GeminiCrewAgent()

# Run a complete workflow
topic = "The Future of Artificial Intelligence in Healthcare"
result = agent_system.run_workflow(topic)

print("Final Result:")
print(result)
```

## 🎛️ Configuration Options

### Available Gemini Models

- `gemini-2.0-flash` (recommended - fast and capable)
- `gemini-2.0-flash-lite` (faster, less capable)
- `gemini-2.0-pro` (more capable, slower)
- `gemini-1.5-flash` (legacy model)
- `gemini-1.5-pro` (legacy model)

### LLM Parameters

```python
# Customize temperature and other settings
gemini_llm = GeminiLLM(
    model="gemini-2.0-flash",
    temperature=0.3,  # Lower = more focused, Higher = more creative
)
```

### Agent Configuration

```python
agent = Agent(
    role="Your Agent Role",
    goal="Specific goal for the agent",
    backstory="Detailed background story for the agent",
    llm=gemini_llm,
    verbose=True,  # Show detailed output
    allow_delegation=True,  # Allow agent to delegate tasks
    max_iterations=5  # Maximum thinking iterations
)
```

## 🔍 Testing Your Setup

### Test Script

Create a simple test to verify your setup:

```python
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def test_gemini_connection():
    """Test Gemini API connection"""
    try:
        genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Hello! Respond with just 'Gemini is working!'")
        print("✅ Gemini API test successful!")
        print(f"Response: {response.text}")
        return True
    except Exception as e:
        print(f"❌ Gemini API test failed: {e}")
        return False

def test_crewai_gemini():
    """Test CrewAI with Gemini integration"""
    try:
        from your_gemini_crew_file import GeminiLLM, Agent, Task, Crew

        gemini_llm = GeminiLLM()
        agent = Agent(
            role="Tester",
            goal="Verify integration works",
            backstory="You are a test agent.",
            llm=gemini_llm,
            verbose=False
        )

        task = Task(
            description="Say 'CrewAI + Gemini integration successful!'",
            agent=agent
        )

        crew = Crew(agents=[agent], tasks=[task])
        result = crew.kickoff()

        print("✅ CrewAI + Gemini integration test successful!")
        print(f"Result: {result}")
        return True
    except Exception as e:
        print(f"❌ CrewAI + Gemini integration test failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing Gemini + CrewAI Integration...\n")

    if test_gemini_connection():
        print("\n" + "="*50)
        test_crewai_gemini()
    else:
        print("Fix Gemini API connection before testing CrewAI integration.")
```

## 🚨 Troubleshooting

### Common Issues

#### 1. "models/gemini-2.0-flash is not found" Error

**Cause:** Using an incorrect model name or API version issue.

**Solution:**
```python
# Try alternative models
gemini_llm = GeminiLLM(model="gemini-1.5-flash")  # Fallback model
# or
gemini_llm = GeminiLLM(model="gemini-pro")  # Legacy model
```

#### 2. "LLM Provider NOT provided" Error

**Cause:** CrewAI expects a different LLM format.

**Solution:** Ensure you're using the custom `GeminiLLM` class, not passing raw parameters to CrewAI.

#### 3. API Key Issues

**Cause:** Invalid or missing API key.

**Solutions:**
- Verify your API key in Google AI Studio
- Check that `GOOGLE_API_KEY` environment variable is set
- Ensure the key has proper permissions

#### 4. Import Errors

**Cause:** Missing dependencies or incorrect imports.

**Solution:**
```bash
pip install --upgrade crewai google-generativeai
```

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Tips

### Model Selection
- Use `gemini-2.0-flash` for most applications (fast, capable)
- Use `gemini-2.0-pro` for complex reasoning tasks
- Use `gemini-2.0-flash-lite` for simple, fast responses

### Temperature Settings
- `temperature=0.1-0.3`: Factual, consistent responses
- `temperature=0.7`: Balanced creativity and consistency
- `temperature=1.0+`: More creative, varied responses

### Token Limits
- Gemini has a 2M token context window for newer models
- Adjust `max_output_tokens` based on your needs
- Monitor token usage to manage costs

## 🔒 Security Best Practices

1. **Never commit API keys** to version control
2. **Use environment variables** for sensitive data
3. **Rotate API keys** regularly
4. **Monitor API usage** in Google AI Studio dashboard
5. **Implement rate limiting** if needed

## 📈 Advanced Features

### Custom Message Formatting

Override the `call` method for custom message handling:

```python
class CustomGeminiLLM(GeminiLLM):
    def call(self, messages, **kwargs):
        # Custom message processing logic
        custom_prompt = self.format_messages_custom(messages)

        response = self.gemini_model.generate_content(
            custom_prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=self.temperature,
                max_output_tokens=2000,
            )
        )

        return self.post_process_response(response.text)

    def format_messages_custom(self, messages):
        # Your custom formatting logic
        pass

    def post_process_response(self, response):
        # Your custom response processing
        pass
```

### Streaming Responses

For real-time streaming:

```python
def call_with_streaming(self, messages, **kwargs):
    response = self.gemini_model.generate_content(
        self.format_messages(messages),
        stream=True  # Enable streaming
    )

    for chunk in response:
        yield chunk.text
```

## 🎯 Use Cases

### Content Creation Pipeline
- Research Agent → Content Writer → Editor → Publisher

### Data Analysis Workflow
- Data Collector → Analyst → Report Generator → Reviewer

### Customer Support System
- Query Classifier → Knowledge Base Searcher → Response Generator → Quality Checker

### Code Review System
- Code Analyzer → Bug Detector → Documentation Generator → Style Checker

## 📚 Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Google AI Studio](https://makersuite.google.com/)
- [Google Generative AI Python SDK](https://ai.google.dev/docs)
- [Gemini Model Documentation](https://ai.google.dev/models/gemini)

## 🤝 Contributing

Feel free to extend this implementation with:
- Additional Gemini model support
- Custom agent templates
- Integration with other tools
- Performance optimizations

---

**Happy building with CrewAI and Gemini! 🚀**
