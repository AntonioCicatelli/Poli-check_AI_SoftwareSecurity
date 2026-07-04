import dotenv
from groq import Groq
from Preprocessor.ner import NER
from Preprocessor.summarizer import Summarizer
from log import Logger

class Preprocessing_Pipeline():
    def __init__(self, env_file="key.env", config=None):
        """
        Initializes the preprocessing pipeline.
        """
        dotenv.load_dotenv(env_file, override=True)

        self.logger = Logger(self.__class__.__name__).get_logger()
        self.ner = NER()
        self.summarizer = Summarizer()

        self.config = {
            "summarize": True,
            "NER": True
        }
        if config:
            self.config.update(config)


    def run_claim_pipe(self, claim, max_lenght=250):
        """
        Processes a news by validating its domain, then translating it to English and summarizing it.
        """
        self.logger.info("Starting news preprocessing...")

        # FASE 1: GATEKEEPER
        is_political = self.summarizer.is_political_claim(claim)
        
        if not is_political:
            self.logger.warning("News rejected by Gatekeeper: Not related to politics.")
            raise ValueError("DOMAIN_REJECTED: Input mismatch: not a political news.")

        # FASE 2: SUMMARIZATION
        if self.config.get("summarize", True):
            # 1. Query corta e restrittiva per Google Fact Check API (2-3 parole)
            claim_title = self.summarizer.claim_title_summarize(claim, max_lenght)
            # 2. NUOVO: Query estesa e semantica per Tavily (ottimizzata per il web)
            web_search_query = self.summarizer.web_search_summarize(claim)

            claim_summary = self.summarizer.generate_summary(claim, max_lenght)
            
            self.logger.info("News preprocessing completed successfully.")
            return claim_title, web_search_query, claim_summary

        self.logger.info("News preprocessing completed (Summarization disabled).")
        return claim, claim, claim

    def run_sources_pipe(self, sources, max_lenght=1024):
        """
        Processes a list of sources by translating and/or summarizing each source as required.
        """
        self.logger.info("Starting sources preprocessing...")

        if self.config.get("summarize", True):
            new_bodies = self.summarizer.summarize_texts([d['body'] for d in sources], max_lenght)
            for d, new_body in zip(sources, new_bodies):
                d['body'] = new_body if new_body else ""
        
        if self.config.get("NER", True):
            for source in sources:
                topic_and_entities = self.ner.extract_entities_and_topic(source['body'])
                
                if topic_and_entities:
                    source['topic'] = topic_and_entities.get('topic', 'Sconosciuto')
                    source['entities'] = topic_and_entities.get('entities', {})
                else:
                    self.logger.warning("Failed to extract entities. Assigning empty defaults.")
                    source['topic'] = 'Sconosciuto'
                    source['entities'] = {}
            
            sources = self.ner.merge_entities(sources)

        self.logger.info("Sources preprocessing completed.")
        return sources