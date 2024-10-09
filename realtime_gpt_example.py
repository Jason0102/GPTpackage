from GPTpackages.Realtime_openai import Realtime_openai
from GPTpackages.PromptTemplate import PromptTemplate
from GPTpackages.Buffer import TextBuffer
import threading

KEY = "your key"

def display_reatime_output(obj:Realtime_openai):
    last_result = ''
    while True:
        try:
            result = obj.get_full_text_output()
            if result != last_result:
                print('Result: ', result)
                last_result = result
                obj.text_stm.set([obj.input_dict, {'output':result}])
        except:
            break

def realtime_text_to_text():
        buffer = TextBuffer(buffer_size=3)
        obj = Realtime_openai(
            key=KEY,
            prompt=PromptTemplate('./prompts/text_prompt.txt'),
            text_memory=buffer,
            mode='text'
        )
        t = threading.Thread(target=display_reatime_output, args=(obj, ), daemon=True)
        t.start()
        while True:
            try:
                text = input('Type in: ')
                input_dict = {'what': text}
                obj.send_text(input_dict)
            except:
                break

def realtime_audio_to_text():
    obj = Realtime_openai(
        key=KEY,
        prompt=PromptTemplate('./prompts/audio_prompt.txt'),
        mode='audio'
    )
    t1 = threading.Thread(target=display_reatime_output, args=(obj, ), daemon=True)
    t1.start()
    t2 = None
    while True:
        audio = obj.listen()
        obj.send_audio(audio)
     