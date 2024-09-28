from GPTpackage.LLMopenai import Embedding

API_KEY='your key'
DIR='folder of documents'

# 建立並儲存向量資料庫
def construct_vector_db():
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR
    )
    model.build_db()
    model.save_db('text_db.json')

# 載入並提取相關資料
def load_and_retrieval():
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR
    )
    model.load_db('text_db.json') 
    query = '聖誕節'
    text = model.retrieve(query, k=3)
    print(text)

# 載入並添加單一文件至向量資料庫
def load_and_add_file():
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR
    )
    model.load_db('text_db.json') 
    model.add_doc('test.txt') # file test.txt must in folder ./conversation_history/
    model.save_db('text_db.json')

# 載入並從向量資料庫移除單一文件
def load_and_remove_file():
    model = Embedding(
        openai_api_key = API_KEY,
        folder=DIR
    )
    model.load_db('text_db.json') 
    model.remove_doc('test.txt') # file test.txt must in folder ./conversation_history/

