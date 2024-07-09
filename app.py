import streamlit as st 
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers

## Function to get response from LLaMa 2 model 
def get_response(input_text, no_words, blog_style):
    try:
        # Calling the model
        llm = CTransformers(model='Models/llama-2-7b-chat.ggmlv3.q8_0.bin',
                            model_type='llama',
                            config={'max_new_tokens': 256,
                                    'temperature': 0.01})

        # Writing the prompt template 
        template = """
            Write a blog for a {blog_style} job profile on the topic "{input_text}" within {no_words} words.
            """

        prompt = PromptTemplate(input_variables=["blog_style", "input_text", "no_words"],
                                template=template)

        ## Generate the response from the LLaMa 2 model
        prompt_text = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
        response = llm(prompt_text)
        return response
    except Exception as e:
        return f"An error occurred: {e}"

st.set_page_config(page_title="Blog Generation",
                   page_icon="🤖",
                   layout="centered",
                   initial_sidebar_state='collapsed'
                   )

st.header("Blog Generation 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating the column fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input("Number of words")
with col2:
    blog_style = st.selectbox('Writing the Blog for',
                              ('Researchers', 'Data Scientists', 'Common People'), index=0)

submit = st.button("Generate")

## Final Response
if submit:
    if input_text and no_words.isdigit():
        response = get_response(input_text, int(no_words), blog_style)
        st.write(response)
    else:
        st.write("Please enter a valid blog topic and number of words.")
