from GPTpackage.LLMopenai import GPT
from GPTpackage.Buffer import ImageBuffer, TextBuffer, encode_image
from GPTpackage.PromptTemplate import PromptTemplate

# inter your openai key
API_KEY="your key"
# inter model name from openai api
MODEL="gpt-4o-mini"

def chat_gpt():
    stm = TextBuffer(buffer_size=3)
    agent = GPT(
        openai_api_key=API_KEY, 
        model=MODEL,
        temperature=0,
        prompt=PromptTemplate('./prompts/vision_prompt.txt'), 
        text_memory=stm,
        img_memory=ImageBuffer()
    )
    text_dict = {'what': '你看到了幾個杯子?他們是什麼顏色的?'}
    path = './input_pictures/IMG_8405.jpg'
    img_list = [encode_image(path)]
    result = agent.run(text_dict, img_list)
    print(result)
    stm.set(['你看到了幾個杯子?他們是什麼顏色的?', result])
    print(stm.get())

