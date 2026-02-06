from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Input Schema for Research Tool
class ResearchToolInput(BaseModel):
    query: str  = Field(
        ...,
        description="Input of Research Tool"
    )

#Research Tool
class ResearchTool(BaseTool):
    name : str = "Research Tool"
    args_schema : Type[BaseModel] = ResearchToolInput
    description : str = (
        "Research Tool is integrated with a rich external knowldege source full of research papers"
    )

       
    def _run(self,query:str) -> str:
         #Defined the embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        #defined a PDF loader
        loader = PyPDFLoader('knowldege/survey_on_icl.pdf')
        #Loading documents from and external knowlege source
        #here ach document is a page
        #len(documents) = no of pages in research paper /external knowledge source
        documents = loader.load()

        #splitter to split documents into page
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap = 200,
            separators=["\n\n","\n",".","!","?"," ",""]
        )

        chunks = splitter.split_documents(documents)

        #defined the vector db
        vector_db = FAISS().from_documents(
            chunks,
            embeddings=embeddings
        )
       

        docs = vector_db.similarity_search(query,k=6)
        return "\n\n".join([d.page_content for d in docs])