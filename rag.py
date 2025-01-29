import os

import dotenv
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

dotenv.load_dotenv()

current_dir = os.path.dirname(os.path.realpath(__file__))
persistent_dir = os.path.join(current_dir, "db", "chroma_db")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    embedding_function=embeddings, persist_directory=persistent_dir,
)

retreiver = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.0},
)

while True:
    query = input("Query: ")

    if query.lower() == "exit":
        break

    relevant_docs = retreiver.invoke(query)

    system_template = ("""You are helpful assistant who answers the question in roman urdu.
Here are some documents which might help you to answer the human query
Relevant documents:
{documents}

Please provide exact answer only based on provided documents.
If the answer is not found in the document, respond with 'I'm not sure'

"""
)

    system_message_prompt = SystemMessagePromptTemplate.from_template(template=system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(template="""{query}""")
    prompt_template = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    model = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.7)
    model_chain = prompt_template | model

    result = model_chain.invoke({
        "documents": "\n\n".join([relevant_doc.page_content for relevant_doc in relevant_docs]),
        "query": query,
    })

    print(result.content)
