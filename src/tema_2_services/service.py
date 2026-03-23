import json
import os
import hashlib

from dotenv import load_dotenv
import numpy as np
import tensorflow_hub as hub
import tensorflow as tf
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import faiss

load_dotenv()

DATA_DIR = os.environ.get("DATA_DIR", "/app/data")
CHUNKS_JSON_PATH = os.path.join(DATA_DIR, "data_chunks.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")
FAISS_META_PATH = os.path.join(DATA_DIR, "faiss.index.meta")
USE_MODEL_URL = os.environ.get(
    "USE_MODEL_URL",
    "https://tfhub.dev/google/universal-sentence-encoder/4",
)

WEB_URLS = [u for u in os.environ.get("WEB_URLS", "").split(";") if u]

class RAGAssistant:
    """Asistent cu RAG din surse web si un LLM pentru raspunsuri."""

    def __init__(self) -> None:
        """Initializeaza clientul LLM, embedderul si prompturile."""
        self.groq_api_key = os.environ.get("GROQ_API_KEY")
        if not self.groq_api_key:
            raise ValueError("Seteaza GROQ_API_KEY in variabilele de mediu.")

        self.client = OpenAI(
            api_key=self.groq_api_key,
            base_url=os.environ.get("GROQ_BASE_URL"))

        os.makedirs(DATA_DIR, exist_ok=True)
        self.embedder = None

        self.relevance = self._embed_texts(
            "Industrial asset inspection, visual intelligence, predictive maintenance, "
            "equipment inventory digitization, computer vision for infrastructure, "
            "drone inspection, corrosion detection, telecom tower audit, "
            "asset condition monitoring, digital twin, equipment defect detection, "
            "VAIP platform, sensor monitoring, industrial IoT, "
            "inspectia activelor industriale, mentenanta predictiva, detectia coroziunii, "
            "inventar echipamente, inspectie drone, monitorizare stare echipamente",
        )[0]

        self.system_prompt = (
            "You are VAIP Assistant, the expert AI advisor for the Visual Asset Intelligence Platform. "
            "VAIP is a SaaS product that converts field photos, videos and drone footage of industrial assets "
            "into structured inventory data, condition analytics dashboards and predictive maintenance insights. "
            "Your expertise covers: computer vision for industrial equipment detection, automated inventory structuring, "
            "condition and risk analytics (corrosion, leaks, damage), drone and video processing for site scanning, "
            "integration with ERP/CMMS systems, and deployment across telecom, energy, utilities and heavy industry. "
            "Answer questions clearly and professionally. Use concrete examples when possible. "
            "If you are given context from our knowledge base, prioritize that information in your answer. "
            "Do not answer questions unrelated to industrial asset management, visual inspection or the VAIP platform."
        )


    def _load_documents_from_web(self) -> list[str]:
        """Incarca si chunked documente de pe site-uri prin WebBaseLoader."""
        if os.path.exists(CHUNKS_JSON_PATH):
            try:
                with open(CHUNKS_JSON_PATH, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                if isinstance(cached, list) and cached:
                    return cached
            except (OSError, json.JSONDecodeError):
                pass

        all_chunks = []
        for url in WEB_URLS:
            try:
                loader = WebBaseLoader(url)
                docs = loader.load()
                for doc in docs:
                    chunks = self._chunk_text(doc.page_content)
                    all_chunks.extend(chunks)
            except Exception:
                continue

        if all_chunks:
            with open(CHUNKS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, ensure_ascii=False)

        return all_chunks

    def _send_prompt_to_llm(
        self,
        user_input: str,
        context: str
    ) -> str:
        """Trimite promptul catre LLM si returneaza raspunsul."""

        system_msg = self.system_prompt

        messages = [
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": (
                    f"Use the following context from our knowledge base to answer the question.\n\n"
                    f"--- Context ---\n{context}\n--- End Context ---\n\n"
                    f"User question: {user_input}\n\n"
                    f"Provide a helpful, structured answer based on the context above. "
                    f"If the context does not contain enough information, say so honestly "
                    f"and offer what you know about the topic from your general knowledge."
                ),
            },
        ]

        try:
            response = self.client.chat.completions.create(
                messages=messages,
                model=os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile"),
            )
            return response.choices[0].message.content
        except Exception:
            return (
                "Asistent: Nu pot ajunge la modelul de limbaj acum. "
                "Te rog incearca din nou in cateva momente."
            )
        
    def _embed_texts(self, texts: str | list[str], batch_size: int = 32) -> np.ndarray:
        """Genereaza embeddings folosind Universal Sentence Encoder."""
        if isinstance(texts, str):
            texts = [texts]
        if self.embedder is None:
            self.embedder = hub.load(USE_MODEL_URL)
        if callable(self.embedder):
            embeddings = self.embedder(texts)
        else:
            infer = self.embedder.signatures.get("default")
            if infer is None:
                raise ValueError("Model USE nu expune semnatura 'default'.")
            outputs = infer(tf.constant(texts))
            embeddings = outputs.get("default")
            if embeddings is None:
                raise ValueError("Model USE nu a returnat cheia 'default'.")
        return np.asarray(embeddings, dtype="float32")

    def _chunk_text(self, text: str) -> list[str]:
        """Imparte textul in bucati cu RecursiveCharacterTextSplitter."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=20,
        )
        chunks = splitter.split_text(text or "")
        return chunks if chunks else [""]

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculeaza similaritatea cosine intre doi vectori."""
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _build_faiss_index_from_chunks(self, chunks: list[str]) -> faiss.IndexFlatIP:
        """Construieste index FAISS din chunks text si il salveaza pe disc."""
        if not chunks:
            raise ValueError("Lista de chunks este goala.")

        embeddings = self._embed_texts(chunks).astype("float32")
        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_PATH)
        with open(FAISS_META_PATH, "w", encoding="utf-8") as f:
            f.write(self._compute_chunks_hash(chunks))
        return index

    def _compute_chunks_hash(self, chunks: list[str]) -> str:
        """Hash determinist pentru lista de chunks si model."""
        payload = json.dumps(
            {
                "model": USE_MODEL_URL,
                "chunks": chunks,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _load_index_hash(self) -> str | None:
        """Incarca hash-ul asociat indexului FAISS."""
        if not os.path.exists(FAISS_META_PATH):
            return None
        try:
            with open(FAISS_META_PATH, "r", encoding="utf-8") as f:
                return f.read().strip()
        except OSError:
            return None

    def _retrieve_relevant_chunks(self, chunks: list[str], user_query: str, k: int = 5) -> list[str]:
        """Rankeaza chunks folosind FAISS si returneaza top-k relevante."""
        if not chunks:
            return []

        current_hash = self._compute_chunks_hash(chunks)
        stored_hash = self._load_index_hash()

        query_embedding = self._embed_texts(user_query).astype("float32")

        index = None
        if os.path.exists(FAISS_INDEX_PATH) and stored_hash == current_hash:
            try:
                index = faiss.read_index(FAISS_INDEX_PATH)
                if index.ntotal != len(chunks) or index.d != query_embedding.shape[1]:
                    index = None
            except Exception:
                index = None

        if index is None:
            index = self._build_faiss_index_from_chunks(chunks)

        faiss.normalize_L2(query_embedding)

        k = min(k, len(chunks))
        if k == 0:
            return []

        _, indices = index.search(query_embedding, k=k)
        return [chunks[i] for i in indices[0] if i < len(chunks)]

    def calculate_similarity(self, text: str) -> float:
        """Returneaza similaritatea cu o propozitie de referinta despre inspectia si managementul activelor industriale."""
        embedding = self._embed_texts(text.strip())[0]
        return self._cosine_similarity(embedding, self.relevance)

    def is_relevant(self, user_input: str) -> bool:
        """Verifica daca intrarea utilizatorului e despre inspectia si managementul activelor industriale."""
        if self.calculate_similarity(user_input) >= 0.15:
            return True
        keywords = [
            "vaip", "inspectie", "inspectia", "echipament", "coroziune", "mentenanta",
            "predictiv", "inventar", "activ industrial", "active industriale", "drone",
            "senzor", "defect", "monitorizare", "digital twin", "teren", "telecom",
            "tower", "asset", "maintenance", "inspection", "corrosion", "equipment",
            "inventory", "condition", "detection", "industrial", "infrastructure",
        ]
        lower_input = user_input.lower()
        return any(kw in lower_input for kw in keywords)

    def assistant_response(self, user_message: str) -> str:
        """Directioneaza mesajul utilizatorului catre calea potrivita."""
        if not user_message:
            return (
                "Te rog scrie un mesaj legat de inspectia activelor industriale, "
                "managementul inventarului sau mentenanta predictiva. "
                "De exemplu: 'Cum detecteaza platforma coroziunea pe echipamente?'"
            )

        if not self.is_relevant(user_message):
            return (
                "Intrebarea ta nu pare sa fie legata de inspectia activelor industriale "
                "sau de platforma VAIP. Te rog sa pui intrebari despre: detectia echipamentelor, "
                "inventarierea automata, analiza starii activelor, mentenanta predictiva "
                "sau procesarea imaginilor si videourilor de pe teren."
            )

        chunks = self._load_documents_from_web()
        relevant_chunks = self._retrieve_relevant_chunks(chunks, user_message)
        context = "\n\n".join(relevant_chunks)
        return self._send_prompt_to_llm(user_message, context)

if __name__ == "__main__":
    assistant = RAGAssistant()
    print("=== Test relevant ===")
    print(assistant.assistant_response("Cum poate platforma VAIP sa detecteze coroziunea pe echipamentele industriale?"))
    print("\n=== Test relevant 2 ===")
    print(assistant.assistant_response("What are the benefits of drone inspection for telecom towers?"))
    print("\n=== Test irelevant ===")
    print(assistant.assistant_response("Care este cea mai buna reteta de pizza?"))
    print("\n=== Test mesaj gol ===")
    print(assistant.assistant_response(""))