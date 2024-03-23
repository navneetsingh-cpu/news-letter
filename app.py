import os
import streamlit as st
import openai
import json
import requests
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.utilities import GoogleSerperAPIWrapper

OPENAI_API_KEY = ""


#  1. Function to make a SERP request and get a list of relevant articles
def search_serp(query):
    search = GoogleSerperAPIWrapper(k=5, type="search")
    response_json = search.results(query)

    print(f"Response=====>, {response_json}")

    return response_json


# 2. Function to choose the best articles using an LLM and return their URLs
def pick_best_articles_urls(response_json, query):

    # Convert response JSON to string
    response_str = json.dumps(response_json)

    # Create LLM to choose best articles
    llm = ChatOpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)
    template = """ 
      You are a world class journalist, researcher, tech, Software Engineer, Developer and a online course creator
      , you are amazing at finding the most interesting and relevant, useful articles in certain topics.
      
      QUERY RESPONSE:{response_str}
      
      Above is the list of search results for the query {query}.
      
      Please choose the best 3 articles from the list and return ONLY an array of the urls.  
      Do not include anything else -
      return ONLY an array of the urls. 
      Also make sure the articles are recent and not too old.
      If the file, or URL is invalid, show www.google.com.
    """
    prompt_template = PromptTemplate(
        input_variables=["response_str", "query"], template=template
    )
    article_chooser_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    urls = article_chooser_chain.run(response_str=response_str, query=query)
    print("******************************")
    print(urls)
    print("******************************")
    # Convert string URLs to a list
    url_list = json.loads(urls)

    return url_list


# 3. Function to get content for each article from URLs and make summaries
def extract_content_from_urls(urls):
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    docs = text_splitter.split_documents(data)
    db = FAISS.from_documents(docs, embeddings)

    return db


# 4. Function to summarize the articles
def summarizer(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([d.page_content for d in docs])

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    template = """
       {docs}
        As a world class journalist, researcher, article, newsletter and blog writer, 
        you will summarize the text above in order to create a 
        newsletter around {query}.
        This newsletter will be sent as an email.  The format is going to be like
        Tim Ferriss' "5-Bullet Friday" newsletter.
        
        Please follow all of the following guidelines:
        1/ Make sure the content is engaging, informative with good data
        2/ Make sure the conent is not too long, it should be the size of a nice newsletter bullet point and summary
        3/ The content should address the {query} topic very well
        4/ The content needs to be good and informative
        5/ The content needs to be written in a way that is easy to read, digest and understand
        6/ The content needs to give the audience actinable advice & insights including resouces and links if necessary
        
        SUMMARY:
    """

    prompt_template = PromptTemplate(
        input_variables=["docs", "query"], template=template
    )
    summarizer_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    response = summarizer_chain.run(docs=docs_page_content, query=query)

    return response.replace("\n", "")


# 5. Function to turn summarization into a newsletter
def generate_newsletter(summaries, query):
    summaries_str = str(summaries)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
    template = """
      Template for generating a newsletter...
      {summaries_str} and {query} are placeholders for actual data.
    """
    prompt_template = PromptTemplate(
        input_variables=["summaries_str", "query"], template=template
    )
    news_letter_chain = LLMChain(llm=llm, prompt=prompt_template, verbose=True)
    news_letter = news_letter_chain.predict(summaries_str=summaries_str, query=query)

    return news_letter


# Define the main function
def main(query, openai_api_key):

    # Check if the user has entered a query
    if query:
        # Print the query to the console
        print(query)

        # Display the query in the app
        st.write(query)

        # Show a spinner while generating the newsletter
        with st.spinner(f"Generating newsletter for {query}"):

            # Search the web for articles related to the query
            search_results = search_serp(query=query)

            # Extract the best URLs from the search results
            urls = pick_best_articles_urls(response_json=search_results, query=query)

            # Extract content from the URLs
            data = extract_content_from_urls(urls)

            # Generate summaries for the extracted content
            summaries = summarizer(data, query)

            # Generate the newsletter thread using the summaries
            newsletter_thread = generate_newsletter(summaries, query)

            # Display search results in an expander
            with st.expander("Search Results"):
                st.info(search_results)

            # Display best URLs in an expander
            with st.expander("Best URLs"):
                st.info(urls)

            # Display extracted data in an expander
            with st.expander("Data"):
                # Fetch and display similarity search data from a FAISS database
                data_raw = " ".join(
                    d.page_content for d in data.similarity_search(query, k=4)
                )
                st.info(data_raw)

            # Display summaries in an expander
            with st.expander("Summaries"):
                st.info(summaries)

            # Display the generated newsletter thread in an expander
            with st.expander("Newsletter:"):
                st.info(newsletter_thread)

        # Display success message when the process is completed
        st.success("Done!")


with st.sidebar:
    OPENAI_API_KEY = st.text_input(
        "OpenAI API Key", key="chatbot_openai_api_key", type="password"
    )
    "[Get an OpenAI API key](https://platform.openai.com/account/api-keys)"

    SERPER_API_KEY = st.text_input(
        "Serp API Key", key="chatbot_serperai_api_key", type="password"
    )
    "[Get an Serper API key (FREE)](https://serper.dev/)"
    "[View the source code](https://github.com/navneetsingh-cpu/youtube-summarizer)"
    OpenAI.api_key = OPENAI_API_KEY
    os.environ["SERPER_API_KEY"] = SERPER_API_KEY
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


with st.form("my_form"):
    st.write("## Enter Topic for News Summary")
    query = st.text_input("", placeholder="Ukraine War")
    submitted = st.form_submit_button("Submit")
    if not OPENAI_API_KEY:
        st.info("Please add your OpenAI API key to continue.")
    if not SERPER_API_KEY:
        st.info("Please add your Serper API key to continue.")

    elif submitted:
        if query:
            OpenAI.api_key = OPENAI_API_KEY
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
            main(query, OPENAI_API_KEY)


# footer
st.write("---")
st.write("Made by [Navneet](https://taplink.cc/navneetskahlon)")
