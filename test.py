from GPTpackage.LLMopenai import GPT, Embedding
from GPTpackage.Buffer import ImageBuffer, TextBuffer, encode_image
from GPTpackage.PromptTemplate import PromptTemplate

# inter your openai key
API_KEY="your key"

# inter model name from openai api
MODEL="gpt-4o-mini"


def embedding_retrieval():
    docs = ['家庭', '籃球', '學生', '婚禮']
    model = Embedding(
        openai_api_key = API_KEY,
        documents=docs
    )
    query = '夫妻'
    text = model.retrieve(query, k=3)
    print(text)

def chat_gpt():
    agent = GPT(
        openai_api_key=API_KEY, 
        model=MODEL,
        prompt=PromptTemplate('./prompts/vision_prompt.txt'), 
        text_memory=TextBuffer(buffer_size=3),
        img_memory=ImageBuffer()
    )
    text_dict = {'what': '你看到了幾個杯子?他們是什麼顏色的?'}
    path = './input_pictures/IMG_8405.jpg'
    img_list = [encode_image(path)]
    result = agent.run(text_dict, img_list)
    print(result)

if __name__ == "__main__":
    embedding_retrieval()
    chat_gpt()
