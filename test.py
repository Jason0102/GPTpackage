from GPTpackages.GPTopenai import GPTopenai
from GPTpackages.PromptTemplate import PromptTemplate
from GPTpackages.ImageBufferMemory import ImageBufferMemory, encode_image
from datetime import datetime
from pathlib import Path
import configparser
from langchain.memory import ConversationBufferMemory

def showtime():
    currentDateAndTime = datetime.now()
    currentTime = currentDateAndTime.strftime("%H-%M-%S")
    print(currentTime)

if __name__ == '__main__':
    config = configparser.ConfigParser()
    config.read('config.ini')
    path = Path('./prompts') / Path('vision_prompt.txt')
    agent = GPTopenai(
        openai_api_key=config.get('openai', 'key1'), 
        prompt=PromptTemplate(path), 
        text_memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True, k=3),
        img_memory=ImageBufferMemory()
    )
    text_dict = {'what': '你看到了幾個杯子?他們是什麼顏色的?'}
    path = Path('./input_pictures') / Path('IMG_8405.jpg')
    showtime()
    img_list = [encode_image(path)]
    result = agent.run(text_dict, img_list)
    showtime()
    print(result)