import requests
import json
from typing import Dict, Any, List
import aiohttp
import re
import asyncio

from app.config import get_settings

class OllamaService:
    def __init__(self):
        self.settings = get_settings()
        self.base_url = self.settings.OLLAMA_BASE_URL
        self.model = self.settings.LLM_MODEL
        self.generate_endpoint = f"{self.base_url}/api/generate"
    
    async def generate_answer(self, question: str, context: str) -> str:
        """
        Generate an answer to a question based on provided context using Ollama
        
        Args:
            question: The question to answer
            context: Document context retrieved from vector search
            
        Returns:
            Generated answer
        """
        prompt = self._format_prompt(question, context)
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.generate_endpoint,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9,
                    }
                }
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Error from Ollama: {error_text}")
                
                result = await response.json()
                return self._process_response(result.get("response", ""))
    
    def _format_prompt(self, question: str, context: str) -> str:
        """Format prompt for the LLM"""
        return f"""<context>
{context}
</context>

Based on the document context above, answer the following question:
Question: {question}

Your answer should be based only on the information in the context. If the answer isn't in the context, say "The provided documents don't contain information to answer this question."

Provide a clear, factual answer with relevant details from the document. Format your answer as JSON with an "answer" field containing your response, like this:

{{
  "answer": "Your detailed answer here"
}}
"""
    
    def _process_response(self, response: str) -> str:
        """Process and extract structured response from LLM output"""
        return self._extract_json_from_response(response).get("answer", response.strip())
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response more robustly"""
        try:
            # First try to parse the entire response as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # If that fails, try to find JSON using balanced brackets approach
            # Looking for the outermost matching brackets
            json_pattern = r'({(?:[^{}]|(?:\{[^{}]*\}))*})'
            matches = re.findall(json_pattern, response)
            
            if not matches:
                # Try a different approach - find any JSON-like structure
                json_pattern = r'({[\s\S]*?})'
                matches = re.findall(json_pattern, response)
                
            if not matches:
                print("Warning: No JSON found in response")
                return {"answer": response.strip()}
                
            # Try parsing each potential JSON match
            for match in matches:
                try:
                    result = json.loads(match)
                    # If it has an answer field, it's likely what we want
                    if "answer" in result:
                        return result
                except json.JSONDecodeError:
                    continue
                    
            # Last attempt - try to extract just the text between the last set of braces
            try:
                last_open = response.rindex('{')
                last_close = response.rindex('}')
                if last_open < last_close:
                    json_str = response[last_open:last_close+1]
                    return json.loads(json_str)
            except (ValueError, json.JSONDecodeError):
                pass
                
            # If we get here, no valid JSON was found
            print("Warning: Could not parse any JSON from response")
            return {"answer": response.strip()}

    async def get_completion(self, prompt, max_retries=3, retry_delay=2):
        """Get completion with retry mechanism"""
        retries = 0
        
        while retries <= max_retries:
            try:
                print(f"🤖 Calling LLM service (attempt {retries+1})")
                # Use existing generate_answer instead of _call_model
                response = await self.generate_answer("", prompt)  # Empty question, using prompt as context
                print("✅ LLM response received successfully")
                return response
            except Exception as e:
                retries += 1
                if retries <= max_retries:
                    print(f"⚠️ LLM call failed: {str(e)}. Retrying in {retry_delay} seconds...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    print(f"❌ LLM call failed after {max_retries} attempts: {str(e)}")
                    raise