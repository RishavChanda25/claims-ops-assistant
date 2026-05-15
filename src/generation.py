import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Ensure your GOOGLE_API_KEY is loaded from your .env file
load_dotenv()

def generate_answer(question: str, retrieved_chunks: list) -> str:
    """
    Takes the user query and retrieved context chunks (LangChain Document objects), 
    and generates a final answer using Gemini 2.5 Flash.
    """
    
    # 1. Initialize the LLM (Temperature 0 for strictly factual, non-creative answers)
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 2. Format the chunks to explicitly show the SOURCE filename to the LLM
    formatted_context = ""
    for i, doc in enumerate(retrieved_chunks):
        # Clean up the filepath to just get the filename
        source = doc.metadata.get('source', 'Unknown Document').split('/')[-1].split('\\')[-1]
        formatted_context += f"\n--- DOCUMENT SOURCE: {source} (Clause {i+1}) ---\n"
        formatted_context += doc.page_content + "\n"
    
    # 3. The Multi-Document System Prompt
    system_prompt = (
        "You are an expert, highly precise Insurance Claims Operations Assistant. "
        "You are analyzing multiple policy documents simultaneously to answer a user's question.\n\n"
        "CRITICAL RULES:\n"
        "1. SYNTHESIS: If different policies provide different coverages (e.g., Homeowners covers one item, Auto covers another), explicitly break down your answer by policy.\n"
        "2. EXCLUSIONS: Always mention specific exclusions or conditions if they apply to the scenario.\n"
        "3. CITATIONS: You MUST cite the exact source document name (provided in the context) when making a coverage determination.\n"
        "4. GROUNDING: If you cannot find the answer in the provided context, state: 'I cannot make a determination based on the provided policies.' Do not guess or use outside knowledge.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # 4. Build the LCEL Chain and Execute
    chain = prompt | llm 
    
    response = chain.invoke({
        "input": question, 
        "context": formatted_context
    })
    
    return response.content