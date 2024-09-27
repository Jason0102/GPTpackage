import requests
import numpy as np

CHAT_URL = 'https://api.openai.com/v1/chat/completions'
EMBEDDING_URL = 'https://api.openai.com/v1/embeddings'

class GPT():
    def __init__(self, openai_api_key:str, prompt, temperature = 0, model="gpt-4-turbo", text_memory=None, img_memory=None) -> None:
        self.key = openai_api_key
        self.prompt = prompt
        self.text_stm = text_memory
        self.img_stm = img_memory
        self.temperature = temperature
        self.model = model

    def run(self, text_dict: dict, img_list=None) -> str:
        send = []
        # load img
        if self.img_stm != None:
            if img_list != []:
                self.img_stm.refresh()
                for img in img_list:
                    send.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "detail": "low"
                    }})
            else: 
                for img in self.img_stm.get_img():
                    send.append({
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img}",
                            "detail": "low"
                        }})
        if img_list != None:
            for img in img_list:
                send.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{img}",
                        "detail": "low"
                }})
                    
        # load text 
        if self.text_stm != None:
            chat_history = self.text_stm.get()
            text_dict['chat_history'] = chat_history

        text = self.prompt.format(text_dict)
        send.append({
            "type": "text",
            "text": text
        })

        # form request
        message = [{
            "role": "user",
            "content": send
            }]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.key}"
            }
        payload = {
            "model": self.model,
            "messages":  message,
            "temperature": self.temperature,
            "max_tokens": 1024
            }

        for i in range(5):
            try:
                response = requests.post(CHAT_URL, headers=headers, json=payload)
                j = response.json()
                output = str(j['choices'][0]['message']['content'])

                return j['choices'][0]['message']['content']
            except Exception as e:
                print(e)
                print(j["error"]["message"])
                continue
        return 'gpt error'
    
class Embedding():
    def __init__(self, openai_api_key:str, documents:list) -> None:
        self.key = openai_api_key
        self.documents = documents
        self.vector_store = []
        for doc in self.documents:
            self.vector_store.append(self.get_embedding(doc))
        return 0

  
    def mmr(self, doc_embeddings, query_embedding, lambda_param=0.5):

        # 計算查詢與每個文件的相似度
        query_similarities = np.array([np.dot(query_embedding, doc_embedding) for doc_embedding in doc_embeddings])
        
        # 初始化選擇的文件列表
        selected_docs = [doc_embeddings[np.argmax(query_similarities)]]
        doc_embeddings = np.delete(doc_embeddings, np.argmax(query_similarities), axis=0)
        query_similarities = np.delete(query_similarities, np.argmax(query_similarities))
        
        while len(doc_embeddings) > 0:
            # 計算MMR分數
            mmr_scores = lambda_param * query_similarities - (1 - lambda_param) * np.max(
                [np.dot(doc_embeddings, selected_doc) for selected_doc in selected_docs])
            
            # 選擇具有最高MMR分數的文件
            best_doc_index = np.argmax(mmr_scores)
            selected_docs.append(doc_embeddings[best_doc_index])
            
            # 移除選擇的文件
            doc_embeddings = np.delete(doc_embeddings, best_doc_index, axis=0)
            query_similarities = np.delete(query_similarities, best_doc_index)
        
        return selected_docs

    def get_embedding(self, text:str):
        headers = {
            'Authorization': f'Bearer {self.key}',
            'Content-Type': 'application/json'
        }
        data = {
            "input": text,
            "model": "text-embedding-ada-002"
        }
        for i in range(5):
            try:
                response = requests.post(EMBEDDING_URL, headers=headers, json=data)
                j = response.json()
                return j['data'][0]['embedding']    
            except Exception as e:
                print(e)
                print(j["error"]["message"])
                continue
        return 'embedding error'
    

    def retrieve(self, query, k=1, method='mmr') -> str:
        if k < 1:
            return 'error of k'
        
        text = ''
        if method == 'mmr':
            query_embedding = self.get_embedding(query)
            sort_embeddings = self.mmr(self.vector_store, query_embedding) 
            sorted_docs = [self.documents[np.argmax([np.dot(sort_embedding, vector) for vector in self.vector_store])] for sort_embedding in sort_embeddings]
            if k > len(sorted_docs):
                k = len(sorted_docs)
            for i in range(k):
                text = text + sorted_docs[i] + '\n'
            
        return text



    