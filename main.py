import os
import argparse
import logging
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains import LLMChain, SimpleSequentialChain
from prompt_templates import keyword_prompt, content_prompt
from write import to_markdown, md2hugo
from langchain.agents.openai_assistant import OpenAIAssistantRunnable
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.utilities.dataforseo_api_search import DataForSeoAPIWrapper
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import StrOutputParser
from langchain.schema import SystemMessage

os.environ["DATAFORSEO_LOGIN"] = "danurahul17@gmail.com"
os.environ["DATAFORSEO_PASSWORD"] = "8fa39b9c069ca970"

wrapper = DataForSeoAPIWrapper()
os.environ["OPENAI_API_KEY"] = "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"
os.environ[
    "SERPAPI_API_KEY"
] = "2c28be3a45d3a3927a9b9f6c59f31c323c7f7cac43c0b9112888e9c2b2eb626b"
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)

Title = "EU agreement on AI law"
Seo_keywords = "EU agreement on AI law"
live_search = True
language = "english"
article_lenght = 1000
Faq = False
Generate_lists = False
tone = "professional"


def get_blog_chain():
    # chain it all together
    # search chain
    Seo = DataForSeoAPIWrapper(
        top_count=3,
        json_result_types=["organic", "knowledge_graph", "answer_box"],
        json_result_fields=["title", "description", "type", "text"],
    )
    Search = SerpAPIWrapper()
    tools = [
        Tool(
            name="Search",
            func=Search.run,
            description="useful for when you need to answer questions about current events. You should ask targeted questions",
        )
    ]

    llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
    agent_executor = initialize_agent(
        tools,
        llm,
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs={
            "system_message": SystemMessage(
                content="""You are a world class resercher, who can do detailed research on any topic and produce facts based results; you do not make things up, you will try as har as possible to gather facts and data to back up the research.

please make sure you complete the objective above with the followin rules:
1/ you should do enough research to gather as much information as possible about the objective
2/ if there are url of relvent links & articles, you will scrap it to gather more information
3/ after scraping & search, you should think "is there any new things i should search and scraping based on the data i collected to increase research quality?" if answer is yes, continue; But don't do this more than 3 iterations
4/ you should not make things up, you should only write facts and data that you have gathered
5/ in the final output, you should include all refrence data & links to back up your research; you should include all refrence data & links to back up your research
6/ Do not use G2, or linkedin, they are mostly outdated data."""
            )
        },
    )

    table_content_chain = (
        [keyword_prompt + f"Tone: {tone}" if tone else keyword_prompt][0]
        | llm
        | StrOutputParser()
    )
    content_chain = (
        [
            content_prompt
            + f"- The length of the blog post should be roughly {article_lenght} words:"
            + "- Bullet points and Lists are included where appropriate:"
            if Generate_lists
            else content_prompt
            + f"- The length of the blog post should be roughly {article_lenght} words:"
        ][0]
        | llm
        | StrOutputParser()
    )
    chain = {"topic": table_content_chain} | RunnablePassthrough.assign(
        content=content_chain
    )
    return chain, agent_executor


if __name__ == "__main__":
    logging.info("Parsing CLI args")
    parser = argparse.ArgumentParser(
        description="A create a blog post as a Markdown file with ecrivai"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./content",
        help="The path to the output directory",
    )
    args = parser.parse_args()

    chain, agent = get_blog_chain()
    logging.info("Generating Title and blog (can take some time)...")
    if live_search == True:
        research_data = agent.invoke({"input": Title + "\n" + Seo_keywords})
        blog_text = research_data["output"]
        blog_text = chain.invoke(
            {
                "title": research_data["output"]
                + "Generate Frequently Asked Questions Section at the end of article"
                if Faq
                else research_data["output"]
            }
        )
    else:
        blog_text = chain.invoke(
            {
                "title": Title
                + "\n"
                + Seo_keywords
                + "Generate Frequently Asked Questions Section at the end of article"
                if Faq
                else Title + "\n" + Seo_keywords
            }
        )
    logging.info("Blog content finished")

    out_dir = args.out_dir
    table_dir = ".table-content"
    logging.info(f"Writing table-of-contents to Markdown file at: {table_dir}")
    md_file_name = to_markdown(blog_text["topic"], out_dir=table_dir)
    logging.info(f"Formatting file header for Hugo")
    blof_file_path = os.path.join(table_dir, md_file_name)
    md2hugo(blof_file_path, blof_file_path)
    logging.info(f"Done")

    logging.info(f"Writing blog to Markdown file at: {out_dir}")
    md_file_name = to_markdown(blog_text["content"], out_dir=out_dir)
    logging.info(f"Formatting file header for Hugo")
    blof_file_path = os.path.join(out_dir, md_file_name)
    md2hugo(blof_file_path, blof_file_path)
    logging.info(f"Done")
