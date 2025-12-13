import os
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.documents import Document


def check_question_relevance(question: str, api_key: str) -> tuple[bool, str]:
    """
    Guardrail function to check if a question is related to restaurant information.
    Returns (is_relevant, explanation)
    """
    guardrail_model = ChatMistralAI(
        model="ministral-3b-latest", 
        mistral_api_key=api_key,
        temperature=0
    )
    
    guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a question classifier for a restaurant review chatbot.
Your job is to determine if a user's question is related to restaurant information.

RELEVANT topics include:
- Food quality, taste, dishes, menu items
- Service quality, staff behavior
- Restaurant atmosphere, ambiance, decor
- Prices, value for money
- Location, hours, contact information
- Reservations, wait times
- Dietary options (vegetarian, vegan, allergies)
- Cleanliness, hygiene
- Parking, accessibility
- Overall experience and recommendations
- Comparing restaurants or cuisines
- Restaurant website or online ordering

NOT RELEVANT topics include:
- General cooking recipes (unless asking about restaurant's specific dishes)
- Unrelated topics (politics, sports, weather, entertainment)
- Personal advice unrelated to dining
- Technical support for other services
- Mathematical calculations
- Programming or coding questions
- Health/medical advice (unless related to food allergies/dietary restrictions)
- Ask for information about chatbot configuration, server, parameters, etc..

Respond with ONLY "RELEVANT" or "NOT_RELEVANT" followed by a brief reason.
Format: RELEVANT: [reason] or NOT_RELEVANT: [reason]"""),
        ("human", "{question}")
    ])
    
    chain = guardrail_prompt | guardrail_model
    response = chain.invoke({"question": question})
    
    response_text = response.content.strip()
    
    # Parse response
    if response_text.startswith("RELEVANT"):
        return True, response_text.split(":", 1)[1].strip() if ":" in response_text else "Question is relevant"
    else:
        return False, response_text.split(":", 1)[1].strip() if ":" in response_text else "Question is not related to restaurants"


def get_rag_chain(reviews_text, restaurant_info):
    """
    Creates a RAG chain for the given restaurant reviews.
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY environment variable not set")

    # 1. Create Documents from reviews
    if isinstance(reviews_text, str):
        texts = [reviews_text]
    else:
        texts = list(reviews_text)

    documents = [Document(page_content=text) for text in texts]

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(documents)

    # 3. Create Vector Store
    embeddings = MistralAIEmbeddings(model="mistral-embed", mistral_api_key=api_key)
    vectorstore = FAISS.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # 4. Create Chain
    model = ChatMistralAI(model="ministral-3b-latest", mistral_api_key=api_key)

    # Format restaurant_info
    clean_info = restaurant_info.replace("{", "(").replace("}", ")")

    system_template = f"""You are a helpful restaurant assistant.
Use the following pieces of retrieved context to answer the question.
The context includes reviews and details about the restaurant: {clean_info}.
If the user asks for a website link or menu, use the 'restaurant_url' from the details if available.
If you don't know the answer, just say that you don't know.
Keep the answer concise and friendly.

Context: {{context}}
"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(model, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


def chat_with_guardrail(question: str, rag_chain, api_key: str) -> dict:
    """
    Process a question through the guardrail before sending to RAG chain.
    Returns a dict with 'answer' and 'is_relevant' keys.
    """
    # Check relevance first
    is_relevant, explanation = check_question_relevance(question, api_key)
    
    if not is_relevant:
        return {
            "answer": f"I'm sorry, but I can only answer questions related to restaurant information, reviews, menus, and dining experiences. Your question appears to be about: {explanation}\n\nPlease ask me about the restaurant's food, service, atmosphere, prices, location, or any other dining-related topics!",
            "is_relevant": False,
            "reason": explanation
        }
    
    # If relevant, proceed with RAG chain
    response = rag_chain.invoke({"input": question})
    
    return {
        "answer": response["answer"],
        "is_relevant": True,
        "context": response.get("context", [])
    }
