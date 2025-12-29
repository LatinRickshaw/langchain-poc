import os
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_core.tools import Tool

# Disable tokenizer parallelism warnings (set to "true" for better performance)
os.environ["TOKENIZERS_PARALLELISM"] = "true"

# Initialize the LLM
llm = ChatAnthropic(
    model="claude-sonnet-4-20250514",
    anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
)

# === PART 1: Load and Process Your Skills Profile ===
def load_skills_profile():
    """Load your skills into a vector store for similarity matching"""
    loader = TextLoader("skills_profile.txt")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings()
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        collection_name="skills_profile"
    )
    
    return vectorstore

# === PART 2: Create Analysis Chain ===
def create_requirement_extractor():
    """Chain to extract key requirements from job description"""
    template = """
    Analyze this job description and extract:
    1. Must-have technical skills
    2. Nice-to-have skills
    3. Domain knowledge required
    4. Management/leadership requirements
    5. Day rate range (if mentioned)
    6. Contract duration

    Job Description:
    {job_description}

    Provide the analysis in a structured format.
    """

    prompt = PromptTemplate(
        input_variables=["job_description"],
        template=template
    )

    return prompt | llm

def create_match_analyzer():
    """Chain to analyze skill match percentage"""
    template = """
    Based on the job requirements and the candidate's skills, provide:

    Job Requirements:
    {requirements}

    Candidate Skills:
    {candidate_skills}

    Analysis:
    1. Match percentage (be realistic)
    2. Strong alignment areas
    3. Skills gaps
    4. Whether to apply (yes/no with reasoning)
    """

    prompt = PromptTemplate(
        input_variables=["requirements", "candidate_skills"],
        template=template
    )

    return prompt | llm

def create_tailoring_chain():
    """Chain to generate CV tailoring suggestions"""
    template = """
    Based on this match analysis, suggest how to tailor the CV:

    Match Analysis:
    {match_analysis}

    Provide:
    1. Which experience to emphasize first
    2. Key achievements to highlight
    3. Specific keywords to include
    4. 3-4 bullet points for a cover message
    """

    prompt = PromptTemplate(
        input_variables=["match_analysis"],
        template=template
    )

    return prompt | llm

# === PART 3: Create Agent with Tools ===
def create_contract_analyzer_agent(vectorstore):
    """Agent that can use multiple tools to analyze opportunities"""
    
    # Tool 1: Skill matcher using vector similarity
    def skill_matcher(query: str) -> str:
        """Searches your skills profile for relevant experience"""
        docs = vectorstore.similarity_search(query, k=3)
        return "\n".join([doc.page_content for doc in docs])
    
    # Tool 2: Rate calculator
    def rate_calculator(input_str: str) -> str:
        """Calculates daily/monthly rates and compares to market"""
        try:
            day_rate = float(input_str)
            monthly = day_rate * 20  # Approximate working days
            yearly = monthly * 12
            
            analysis = f"""
            Day Rate: £{day_rate}
            Monthly: £{monthly:,.0f}
            Yearly equivalent: £{yearly:,.0f}
            
            Market context:
            - Senior Frontend/Fullstack: £500-700/day
            - Data Engineering: £600-800/day
            - Technical Leadership: £700-900/day
            
            Assessment: {"Competitive" if day_rate >= 600 else "Below market" if day_rate < 500 else "Reasonable"}
            """
            return analysis
        except:
            return "Please provide a numeric day rate value"
    
    # Tool 3: Job description analyzer
    def analyze_job_description(job_text: str) -> str:
        """Extracts key information from job description"""
        extractor = create_requirement_extractor()
        result = extractor.invoke({"job_description": job_text})
        return result.content
    
    # Create tools
    tools = [
        Tool(
            name="SkillMatcher",
            func=skill_matcher,
            description="Searches your skills profile to find relevant experience. Input should be a skill or requirement from the job description."
        ),
        Tool(
            name="RateCalculator",
            func=rate_calculator,
            description="Calculates and analyzes day rates. Input should be a numeric day rate value."
        ),
        Tool(
            name="JobAnalyzer",
            func=analyze_job_description,
            description="Analyzes a job description to extract requirements. Input should be the full job description text."
        )
    ]
    
    # For simplicity, return a simple wrapper that uses the LLM directly with tools
    # Modern agents in LangChain are more complex - this is a simplified version
    class SimpleAgent:
        def __init__(self, tools, llm):
            self.tools = tools
            self.llm = llm
            self.tool_map = {tool.name: tool for tool in tools}

        def run(self, query: str) -> str:
            """Simple agent that can call tools based on the query"""
            # For this simplified version, we'll just use the LLM directly
            result = self.llm.invoke(query)
            return result.content

    return SimpleAgent(tools, llm)

# === PART 4: Main Application ===
def analyze_opportunity(job_description: str, vectorstore):
    """Full pipeline to analyze a contract opportunity"""
    
    print("\n" + "="*60)
    print("CONTRACT OPPORTUNITY ANALYZER")
    print("="*60)
    
    # Step 1: Extract requirements
    print("\n[1] Extracting requirements...")
    extractor = create_requirement_extractor()
    requirements_result = extractor.invoke({"job_description": job_description})
    requirements = requirements_result.content
    print(requirements)
    
    # Step 2: Find matching skills from your profile
    print("\n[2] Finding relevant skills in your profile...")
    relevant_skills = vectorstore.similarity_search(job_description, k=5)
    candidate_skills = "\n".join([doc.page_content for doc in relevant_skills])
    
    # Step 3: Analyze match
    print("\n[3] Analyzing match...")
    matcher = create_match_analyzer()
    match_result = matcher.invoke({
        "requirements": requirements,
        "candidate_skills": candidate_skills
    })
    match_analysis = match_result.content
    print(match_analysis)

    # Step 4: Generate tailoring suggestions
    print("\n[4] Generating CV tailoring suggestions...")
    tailoring = create_tailoring_chain()
    suggestions_result = tailoring.invoke({"match_analysis": match_analysis})
    suggestions = suggestions_result.content
    print(suggestions)
    
    return {
        "requirements": requirements,
        "match_analysis": match_analysis,
        "suggestions": suggestions
    }

# === PART 5: Interactive Agent Mode ===
def interactive_mode():
    """Chat with the agent about opportunities"""
    
    print("\n" + "="*60)
    print("INTERACTIVE CONTRACT ANALYZER")
    print("="*60)
    print("Ask questions about job opportunities, rates, or your skills")
    print("Type 'exit' to quit\n")
    
    vectorstore = load_skills_profile()
    agent = create_contract_analyzer_agent(vectorstore)
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['exit', 'quit']:
            break
        
        response = agent.run(user_input)
        print(f"\nAgent: {response}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    # Example usage
    sample_job = """
    Senior Data Engineer - 6 month contract - £650/day
    
    We're seeking a Data Engineer with strong Python and AWS experience.
    Must have: Python, Spark, Kafka, AWS (Lambda, S3)
    Nice to have: Energy sector experience, IoT protocols
    You'll be building real-time data pipelines for our EV charging platform.
    """
    
    vectorstore = load_skills_profile()
    
    # Run full analysis
    results = analyze_opportunity(sample_job, vectorstore)
    
    # Or enter interactive mode
    # interactive_mode()