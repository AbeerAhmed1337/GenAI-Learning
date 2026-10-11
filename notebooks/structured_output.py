import os 
from dotenv import load_dotenv 
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY is missing. Add it to your .env file or environment variables.")

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=groq_api_key,)

from pydantic import BaseModel, Field

class Person(BaseModel):
    name: str = Field(..., description = "The Person's Full Name")
    age: int = Field(..., description = "The Person's Age")
    email: str = Field(...,  description = "The Person's Email Address")
    person_role: str = Field(..., description = "The Person's Role in the Organization")
    company_name: str = Field(..., description = "The Person's Company Name")

structure_llm = llm.with_structured_output(Person, method="json_mode")


text = """
Extract the person's details from the text below.
Return only a valid JSON object with these keys: name, age, email, person_role, company_name.

I am Mohammad Abeer Ahmed Siddiqui, I am 23 years old, I work as a AI Engineer at Groq Inc. My email is siddiqui.abeer456@gmail.com and I am excited to be a part of this amazing organization.
"""

res = structure_llm.invoke(text)
print(f"name='{res.name}'")
print(f"age={res.age}")
print(f"email='{res.email}'")
print(f"person_role='{res.person_role}'")
print(f"company_name='{res.company_name}'")


from typing import List

class Movies(BaseModel):
    movie: str = Field(..., description = "The movie title")
    year: int = Field(..., description = "The year of release of the movies")
    protagonist: str = Field(..., description="The main character of that movie")

class AllMovies(BaseModel):
    movies: List[Movies]


movie_output = llm.with_structured_output(AllMovies, method="json_mode")


response = movie_output.invoke("""
Suggest 3 science-fiction movies.

Return only valid JSON in this format:
{"movies": [{"movie": "...", "year": 2000, "protagonist": "..."}]}
""")

for movie in response.movies:
    print(f"movie='{movie.movie}'")
    print(f"year={movie.year}")
    print(f"protagonist='{movie.protagonist}'")
    print()