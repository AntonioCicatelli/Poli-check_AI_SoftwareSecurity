import os
import time
import dotenv
from groq import Groq
from log import Logger

class Summarizer:
    def __init__(self, env_file="key.env"):
        """
        Initializes the Summarizer class with a specific model and configures the Groq API client.
        """
        self.logger = Logger(self.__class__.__name__).get_logger()
        dotenv.load_dotenv(env_file, override=True)
        self.model = os.getenv("GROQ_MODEL_NAME")
        self.low_model = os.getenv("GROQ_LOW_MODEL_NAME")
        self.client = Groq()

    def is_political_claim(self, text, temperature=0.0):
        """
        Classifies whether the given text belongs to the political domain.
        """
        self.logger.info("Starting domain classification for the claim.")
        self.logger.debug("Claim to classify: %s...", text[:200])

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a domain classifier. 
                                                    Your task is to determine if the user's claim is related to politics, government, elections, legislation, geopolitics, international relations, or actions by political figures and states.
                                                    Even if the claim is an absurd conspiracy theory, an act of war, or extreme, if it involves politicians, governments, or countries, it MUST be classified as political.
                                                    Reply ONLY with the exact word 'True' if it is political, or 'False' if it is not. Do not add any other text."""},
                    {"role": "user", "content": text}
                ],
                model=self.low_model,
                temperature=temperature,
                max_completion_tokens=10
            )
            
            result = response.choices[0].message.content.strip().lower()
            is_political = 'true' in result
            self.logger.info("Domain classification result: %s (Is Political: %s)", result, is_political)
            return is_political

        except Exception as e:
            self.logger.error("Error during domain classification: %s", e)
            return True

    def claim_title_summarize(self, text, max_tokens=1024, temperature=0.0, stop=None):
        """
        Generates a summary for the given claim using the Groq API.
        """
        self.logger.info("Starting summarization process.")
        self.logger.info("Input text: %s...", text[:200]) 

        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are a strict keyword extractor for a fact-checking search engine. 
                                                    Your task is to extract exactly 2 or 3 core keywords from the user's claim.
                                                    CRITICAL: You MUST output the keywords in the EXACT SAME LANGUAGE as the user's claim.
                                                    CRITICAL: Output ONLY the keywords separated by a space. Do NOT write sentences. Do NOT use punctuation or quotes.
                                                    Example Input: "Zelensky ha comprato due yacht di lusso con i soldi americani"
                                                    Example Output: Zelensky yacht americani"""},
                    {"role": "user", "content": text}
                ],
                model=self.model,
                temperature=temperature, # Impostato a 0.0 tramite il parametro per renderlo preciso e robotico
                max_completion_tokens=15, # Limite stringente per evitare che scriva frasi
                stop=stop
            )

            summary = response.choices[0].message.content.strip()
            
            # Pulizia extra: rimuoviamo virgolette o punti che potrebbero confondere Google
            summary = summary.replace('"', '').replace("'", "").replace(".", "")
            
            self.logger.info("Summarization completed successfully.")
            self.logger.info("Generated scraping summary: %s", summary)
            return summary

        except Exception as e:
            self.logger.error("Error generating summary: %s", e)
            return None
        
    def web_search_summarize(self, text, temperature=0.0, stop=None):
        """
        Generates an optimized, detailed search query for semantic web search (e.g., Tavily).
        """
        self.logger.info("Starting web search summarization process.")
        
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": """You are an expert SEO specialist. 
                                                    Your task is to optimize the user's claim for a semantic web search engine like Tavily.

                                                    CRITICAL RULES:
                                                    1. Keep the query detailed enough to find specific articles (usually 4 to 8 words).
                                                    2. Retain all Named Entities, specific numbers, and the core action using the EXACT ORIGINAL WORDS.
                                                    3. Remove conversational filler words.
                                                    5. NEVER TRANSLATE THE TEXT. You MUST output the query in the EXACT SAME LANGUAGE and vocabulary as the user's claim.
                                                    5. Output ONLY the optimized query string.

                                                    EXAMPLES:
                                                    Input: "Zelensky ha comprato due yacht di lusso da 75 milioni con i soldi americani"
                                                    Output: Zelensky comprato yacht fondi americani
                                                    
                                                    Input: "President Trump has openly criticized the Pontiff, suggesting that the Vatican's stance is undermining national security"
                                                    Output: Trump criticized Pontiff Vatican undermining national security """},
                    {"role": "user", "content": text}
                ],
                model=self.low_model, # Usiamo il modello veloce per non perdere tempo
                temperature=temperature,
                max_completion_tokens=25,
                stop=stop
            )

            query = response.choices[0].message.content.strip()
            # Pulizia di sicurezza
            query = query.replace('"', '').replace("'", "")
            
            self.logger.info("Web search query generated: %s", query)
            return query

        except Exception as e:
            self.logger.error("Error generating web search query: %s", e)
            # Fallback di sicurezza: se l'API fallisce, passa a Tavily la frase originale 
            # (che essendo semantico, la capirà comunque molto bene)
            return f"{text} fact-check"
    
    
    def generate_summary(self, text, max_tokens=1024, temperature=0.5, stop=None):
        """
        Generates a summary for the given text using the specified model.
        """
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": """You are a neutral summarizer. 
                                                Your task is to summarize the user's input EXACTLY as it is presented, maintaining its original premise.
                                                CRITICAL: DO NOT fact-check, debunk, or correct the user's claim. Act only as a mirror summarizing what was said.
                                                Don't use lists or bullet points. Provide only the string without specifying that it is a summary.
                                                CRITICAL: Detect the language of the user's text and write the summary exclusively in that EXACT SAME LANGUAGE."""},
                {"role": "user", "content": text}
            ],
            model=self.low_model,
            temperature=temperature,
            max_completion_tokens=max_tokens,
            stop=stop
        )
        return response.choices[0].message.content.strip()

    def summarize_texts(self, texts, max_tokens=1024, temperature=0.5, stop=None, token_cut=20000, sleep_temperature=0.0):
        """
        Generates summaries for a list of texts.
        """
        self.logger.info("Starting batch summarization process for %d texts.", len(texts))
        summaries = []

        for index, text in enumerate(texts):
            self.logger.info("Summarizing text %d/%d...", index + 1, len(texts))
            self.logger.debug("Text %d content: %s", index + 1, text[:200])
            
            cutted_text = text[:token_cut]

            try:
                summary = self.generate_summary(
                                text=cutted_text,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                stop=stop
                            )
                if summary:
                    summaries.append(summary)
                    self.logger.info("Text %d summarized successfully.", index + 1)
                else:
                    self.logger.warning("No summary returned for text %d.", index + 1)
            except Exception as e:
                self.logger.error("Error summarizing text %d: %s", index + 1, str(e))
                summaries.append(None) 
            
            if index < len(texts) - 1:
                self.logger.info("Pausa di 2 secondi per rispettare i limiti API di Groq...")
                time.sleep(2.0)

        self.logger.info("Batch summarization process completed.")
        return summaries