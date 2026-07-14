import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

def generate_answer(question: str, retrieved_chunks: list, image_bytes: bytes = None) -> str:
    """
    Takes the user query, retrieved context chunks, and an optional image, 
    and generates a final answer using Gemini 2.5 Flash.
    """
    
    # 1. Initialize the LLM 
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    
    # 2. Format the chunks to explicitly show the SOURCE filename
    formatted_context = ""
    for i, doc in enumerate(retrieved_chunks):
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
        "4. GROUNDING: If you cannot find the answer in the provided context, state: 'I cannot make a determination based on the provided policies.' Do not guess or use outside knowledge.\n"
        "5. VISUAL ASSESSMENT: If an image is provided, first describe the damage you observe in the image. Then, use ONLY the provided text context to determine if that specific observed damage is covered.\n\n"
        f"Context:\n{formatted_context}"
    )
    
    # 4. Dynamically build the Human input
    if image_bytes:
        encoded_image = base64.b64encode(image_bytes).decode("utf-8")
        human_content = [
            {"type": "text", "text": question},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]
    else:
        human_content = question

    # 5. Execute using native Message classes
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_content)
    ]
    
    response = llm.invoke(messages)
    
    return response.content