import os
from dotenv import load_dotenv
import pickle
import faiss
import asyncio
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)
TELEGRAM_TOKEN= os.getenv("TELEGRAM_TOKEN")
TEXT_FOLDER = "C:/Users/ASUS/Desktop/AIqanun/qanunlar"
CHUNKS_FILE = "law_chunks.pkl"
INDEX_FILE = "law_index.faiss"
top_k = 3



def read_txt_file(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def read_all_txts(folder):
    all_texts = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder, filename)
            all_texts.append(read_txt_file(file_path))
    return all_texts

def split_text(text, size=500):
    return [text[i:i+size] for i in range(0, len(text), size)]

def prepare_law_data():
    if os.path.exists(INDEX_FILE) and os.path.exists(CHUNKS_FILE):
        print("Mövcud indekslər yüklənir, gözləyin...")
        index = faiss.read_index(INDEX_FILE)
        with open(CHUNKS_FILE, "rb") as f:
            chunks = pickle.load(f)
    else:
        print("Yeni indekslər yaradılır...")
        texts = read_all_txts(TEXT_FOLDER)
        combined_text = "\n".join(texts)
        chunks = split_text(combined_text)
    
        vectorizer = TfidfVectorizer()
        
        X = vectorizer.fit_transform(chunks).toarray()
        
        #FAISS indeksi yaradır
        dim = X.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(X)
        
        #İndeksləri yadda saxlayır
        faiss.write_index(index, INDEX_FILE)
        with open(CHUNKS_FILE, "wb") as f:
            pickle.dump(chunks, f)

    return index, chunks

def find_relevant_chunks(query, index, chunks):
    
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(chunks + [query]).toarray()
    
    query_vec = X[-1]
    chunk_vecs = X[:-1]

    similarities = cosine_similarity([query_vec], chunk_vecs)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def generate_answer(query, context, temperature=0.3):
    prompt = (
        "Sən Azərbaycan Respublikasının qanunlarına əsasən cavab verən hüquqi bot-san. Yalnız sənə təqdim olunan text fayllar əsasında cavab ver.\n"
        f"Sual: {query}\n"
        f"Əlaqəli qanun hissələri:\n{context}\n\n"
        "Qanunlara əsaslanaraq cavab ver:"
    )

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Azərbaycan qanunları üzrə hüquqi məsləhət verən bir botsan. Yalnız sənə təqdim olunan text fayllar əsasında cavab ver."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        top_p=1 
    )

    return response.choices[0].message.content

index, chunks = prepare_law_data()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Salam! AIQanuna xoş gəldin! Səni maraqlandıran hüquqi sualını yaz:")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.message.from_user
    query = update.message.text
    relevant = find_relevant_chunks(query, index, chunks)
    chat_id = update.effective_chat.id

    print(f"[USER] {user.username or user.first_name}: {query}")

    loading_msg = await context.bot.send_message(
        chat_id=chat_id,
        text="🧠 Cavab axtarılır, bu biraz vaxt apara bilər..."
    )
    for dots in ["Bot düşünür.", "Bot düşünür..", "Bot düşünür..."]:
        await asyncio.sleep(0.5)
        await loading_msg.edit_text(dots)

    context_text = "\n\n".join(relevant)
    answer = generate_answer(query, context_text)
    print(f"[BOT] Cavab verir {user.username or user.first_name}: {answer}")

    await update.message.reply_text(answer)


def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("Bot işə düşdü...")
    app.run_polling()

if __name__ == "__main__":
    main()