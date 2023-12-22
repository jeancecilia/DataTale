from langchain.prompts import PromptTemplate

topic_prompt = PromptTemplate.from_template(
    """{dummy}Give me a single, specific topic to write an informative, engaging blog about.
This blog topic must be relevant and appealing to many people so that many readers will want to read about it.
The specific topic can be from a wide range of broader topics like physics, science, engineering, lifestyle, health, learning, teaching, history, technology, cryptocurrency, art, music, sport, business, economics, travel, entertainment, gaming, food, etc.
Only give me the specific topic name after this prompt and nothing else. The topic is:""",
)

keyword_prompt = PromptTemplate.from_template(
    "Give me a list of 5 keywords and a table-of-content that for using in blog about {title}",
)

content_prompt = PromptTemplate.from_template(
    """
write a blog article with long pagragrpahs.
- Each section should have a clear and catchy heading that summarizes its main point.
Your goal is to deliver a unified and eloquent blog post about '{topic}', one that maintains a singular thread of thought throughout, engaging the reader with its depth and cohesiveness. Launch into your narrative without preambles:"
""",
)
