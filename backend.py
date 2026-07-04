from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from WebScraper.scraper import Scraper
from Preprocessor.preprocessing_pipeline import Preprocessing_Pipeline
from Database.data_entities import Claim, Answer
from Database.sqldb import Database
from GraphRAG.rag_pipeline import RAG_Pipeline
from GraphRAG.graph_manager import GraphManager
import urllib.parse
import requests
from bs4 import BeautifulSoup

backend_app = FastAPI()

db = Database()

class InputText(BaseModel):
    text: str

@backend_app.post("/run_pipeline")
def process_text(input_text: InputText):
    text = input_text.text
    
    # --- NUOVA LOGICA: Controllo se l'input è un URL ---
    parsed_url = urllib.parse.urlparse(text)
    # Rileva sia URL web classici che localhost (es. http://localhost:8080/news)
    if parsed_url.scheme in ["http", "https"] and parsed_url.netloc:
        try:
            # Effettua la richiesta HTTP (gestendo localhost se necessario)
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            }
            # Timeout breve per non bloccare il sistema se l'URL non risponde
            response = requests.get(text, headers=headers, timeout=10)
            response.raise_for_status()
            
            # Estrazione del contenuto con BeautifulSoup
            soup = BeautifulSoup(response.content, 'html.parser')
            page_title = soup.title.string if soup.title else ""
            body = soup.get_text(separator=' ', strip=True)
            
            # Combiniamo il titolo e il body. Limitiamo a 3000 caratteri per non far impazzire 
            # il Gatekeeper e il Summarizer con articoli lunghissimi.
            text = f"Verifica questa notizia estratta dal web: Titolo: {page_title}. Contenuto: {body}"[:3000]
            
        except Exception as e:
            # Se l'URL fallisce (sito down, localhost non accessibile), blocchiamo con grazia
            raise HTTPException(status_code=400, detail=f"Impossibile estrarre il contenuto dall'URL fornito: {str(e)}")
    # --- FINE NUOVA LOGICA ---

    preprocessor = Preprocessing_Pipeline()
    
    try:
        # FASE 1: Preprocessing e Gatekeeper (utilizzerà 'text' che ora contiene il testo dell'URL o il testo originale)
        claim_title, web_search_query, claim_summary = preprocessor.run_claim_pipe(text)
    # ... RESTO DEL CODICE INVARIATO (da except ValueError as e: in poi) ...
    
    except ValueError as e:
        # Se il Gatekeeper rifiuta la notizia (es. non è politica), restituiamo un errore 400
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Fallback per altri errori imprevisti nel preprocessing
        raise HTTPException(status_code=500, detail=f"Errore di preprocessing: {str(e)}")

    claim = Claim(text, claim_title, claim_summary)
    
    try:
        # FASE 2: Web Scraping
        scraper = Scraper()
        sources = scraper.search_and_extract(claim_title, web_search_query, num_results=10)
        if not sources:
            # Interrompiamo la pipeline qui e restituiamo il messaggio pulito per la dashboard
            return {
                "claim_title": claim_title, 
                "claim_summary": "No reliable sources were found to verify this query. The system exhausted all search attempts.", 
                "sources": [], 
                "query_result": "Unverifiable news, no source found 🟡", 
                "graphs_folder": ""
            }
        preprocessed_sources = preprocessor.run_sources_pipe(sources)
        claim.add_sources(preprocessed_sources)
        
        # FASE 3: Graph RAG
        rag = RAG_Pipeline()
        query_result, graphs_folder = rag.run_pipeline(preprocessed_sources, claim.text, claim.id)

        answer = Answer(claim.id, query_result, graphs_folder)
        
        return {
            "claim_title": claim_title, 
            "claim_summary": claim_summary, 
            "sources": preprocessed_sources, 
            "query_result": query_result, 
            "graphs_folder": graphs_folder
        }
    except Exception as e:
        # Evitiamo che errori di scraping o generazione crashino brutalmente il backend
        raise HTTPException(status_code=500, detail=f"Errore interno durante l'elaborazione: {str(e)}")


@backend_app.post("/delete_db")
def delete_database():
    # Cancella la cronologia SQLite e i file locali
    db.delete_all_conversations()
    
    # AGGIUNGI QUESTO: Resetta anche il grafo persistente su Neo4j
    try:
        gm = GraphManager()
        gm.reset_data()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Errore nel reset del grafo: {str(e)}")


@backend_app.get("/get_history")
def get_history():
    history = db.get_history()
    return history