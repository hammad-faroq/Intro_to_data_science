# imports

import os
import re
import math
import json
from typing import List, Dict
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import chromadb
from items import Item
from testing import Tester
from agents.agent import Agent
import requests


class FrontierAgent(Agent):

    name = "Frontier Agent"
    color = Agent.BLUE

    
    def __init__(self, collection):
        """
        This is the changed version of the agentic framework in which i have used a locally running model of llama3.1 insted of 
        paying for the api cost of the Openai_api service (:
        """
        self.log("Initializing Frontier Agent")
        self.log("Frontier Agent is setting up with LLama")
        self.collection = collection
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.log("Frontier Agent is ready")

    def make_context(self, similars: List[str], prices: List[float]) -> str:
        """
        Create context that can be inserted into the prompt
        :param similars: similar products to the one being estimated
        :param prices: prices of the similar products
        :return: text to insert in the prompt that provides context
        """
        message = "To provide some context, here are some other items that might be similar to the item you need to estimate.\n\n"
        for similar, price in zip(similars, prices):
            message += f"Potentially related product:\n{similar}\nPrice is ${price:.2f}\n\n"
        return message

    def messages_for(self,item, similars, prices):
        system_message = "You estimate prices of items. Reply only with the price, no explanation."
        user_prompt = self.make_context(similars, prices)
        user_prompt += "And now the question for you:\n\n"
        user_prompt += "How much does this cost?\n\n" + item
    
        # Combine everything into a single natural language instruction
        return f"{system_message}\n\n{user_prompt}\n\nPrice is $"

    def find_similars(self, description: str):
        """
        Return a list of items similar to the given one by looking in the Chroma datastore
        """
        self.log("Frontier Agent is performing a RAG search of the Chroma datastore to find 5 similar products")
        vector = self.model.encode([description])
        results = self.collection.query(query_embeddings=vector.astype(float).tolist(), n_results=2)
        documents = results['documents'][0][:]
        prices = [m['price'] for m in results['metadatas'][0][:]]
        self.log("Frontier Agent has found similar products")
        return documents, prices

    def get_price(self, s) -> float:
        """
        A utility that plucks a floating point number out of a string
        """
        s = s.replace('$','').replace(',','')
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        return float(match.group()) if match else 0.0

    def price(self,item):
        documents, prices = self.find_similars(item)
        prompt = self.messages_for(item, documents, prices)
    
        with requests.post("http://127.0.0.1:11434/api/generate", json={
            "model": "llama3.2",
            "prompt": prompt,
            "temperature": 0.1
        }, stream=True) as resp:
    
            if resp.status_code != 200:
                raise Exception(f"Ollama error: {resp.status_code}, {resp.text}")
    
            reply = ""
            for line in resp.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        reply += data.get("response", "")
                    except json.JSONDecodeError:
                        continue  # Skip malformed chunks
    
        return self.get_price(reply.strip())
        