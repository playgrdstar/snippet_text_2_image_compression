import os
import json
import requests
from typing import List, Dict, Any, Optional, Union, Tuple, Type
from retrying import retry
from pathlib import Path
from openai import OpenAI
import logging
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
import openai

from tqdm import tqdm
import instructor

BASE_DIR = Path(__file__).resolve()
ENV_DIR = os.path.join(BASE_DIR, '.env')

logger = logging.getLogger(__name__)

# Lazy loading of dotenv
def load_env_vars():
    """Load environment variables from .env file."""
    from dotenv import load_dotenv, find_dotenv
    env_file = find_dotenv()
    if env_file:
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from: {env_file}")
    else:
        logger.warning("No .env file found")

class CustomHuggingFaceEndpoint:
    def __init__(self, repo_id: str, api_key: str, **kwargs):
        """Initialize a HuggingFace endpoint for text generation."""
        self.api_url = f"https://api-inference.huggingface.co/models/{repo_id}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.default_params = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.95),
        }

    def invoke(self, prompt: str, **kwargs) -> str:
        """Generate text using the HuggingFace endpoint."""
        payload = {
            "inputs": prompt,
            "parameters": {**self.default_params, **kwargs}
        }
        response = requests.post(self.api_url, headers=self.headers, json=payload)
        response.raise_for_status()
        return response.json()[0]["generated_text"]

class CustomHuggingFaceInferenceAPIEmbeddings:
    def __init__(self, api_key: str, model_name: str):
        """Initialize a HuggingFace endpoint for embeddings generation."""
        # self.api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"
        self.api_url = f"https://router.huggingface.co/hf-inference/models/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        logger.info(f"Initialized HuggingFace embeddings model from api: {self.api_url}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        response = requests.post(self.api_url, headers=self.headers, json={"inputs": texts})
        response.raise_for_status()
        return response.json()

class CustomTogetherEmbeddings:
    def __init__(self, api_key: str, model_name: str):
        """Initialize Together AI endpoint for embeddings generation."""
        from together import Together
        self.client = Together(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized Together AI embeddings model: {self.model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        response = self.client.embeddings.create(
            model=self.model_name,
            input=texts
        )
        return [item.embedding for item in response.data]

class ModelConfig:
    def __init__(self):
        """Initialize model configuration and load environment variables."""
        load_env_vars()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        self.hf_api_key = os.environ.get("HF_READ_API_KEY")
        self.openrouter_api_key = os.environ.get("OPENROUTER_API_KEY")

      
        # Added for specific use cases in AnnotationService
        self.vision_model_name = os.environ.get("VISION_MODEL_NAME", "gpt-4o-mini") # Default to gpt-4o-mini
        self.chat_model_name = os.environ.get("CHAT_MODEL_NAME", "gpt-4o-mini") # Default to gpt-4o-mini
        logger.info(f"Vision model name: {self.vision_model_name}")
        logger.info(f"Chat model name: {self.chat_model_name}")

        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables after attempting to load .env. LLM features requiring OpenAI will fail to initialize.")
        else:
            logger.info("OPENAI_API_KEY found in environment variables.")
        if not self.hf_api_key:
            logger.warning("HF_READ_API_KEY not found in environment variables")
        if not self.openrouter_api_key:
            logger.warning("OPENROUTER_API_KEY not found in environment variables")

    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=60000,
           retry_on_exception=lambda e: isinstance(e, (requests.exceptions.RequestException, ConnectionError, TimeoutError)))
    def initialize_llm(self, 
                       provider: str = "openai", 
                       model_name: Optional[str] = None,
                       **kwargs) -> Any:
        """Initialize a language model based on the provider."""
        if provider == "openai":
            model_name = model_name or "gpt-4o-mini"
            return OpenAI(api_key=self.openai_api_key)
        elif provider == "openrouter":
            model_name = model_name or "meta-llama/llama-3.2-1b-instruct:free"
            return OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=self.openrouter_api_key
            )
        elif provider == "huggingface":
            model_name = model_name or "mistralai/Mixtral-8x7B-Instruct-v0.1"
            return CustomHuggingFaceEndpoint(
                repo_id=model_name,
                api_key=self.hf_api_key,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=60000,
           retry_on_exception=lambda e: isinstance(e, (requests.exceptions.RequestException, ConnectionError, TimeoutError)))
    def initialize_embeddings(self, 
                              provider: str = "openai", 
                              model_name: Optional[str] = None) -> Any:
        """Initialize an embeddings generator based on the provider."""
        if provider == "openai":
            model_name = model_name or "text-embedding-ada-002"
            return OpenAI(api_key=self.openai_api_key)
        elif provider == "huggingface":
            model_name = model_name or "BAAI/bge-large-en-v1.5"
            return CustomHuggingFaceInferenceAPIEmbeddings(
                api_key=self.hf_api_key,
                model_name=model_name
            )
        elif provider == "together":
            model_name = model_name or "BAAI/bge-large-en-v1.5"
            return CustomTogetherEmbeddings(
                api_key=os.environ.get("TOGETHER_API_KEY"),
                model_name=model_name
            )
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=60000,
           retry_on_exception=lambda e: isinstance(e, (requests.exceptions.RequestException, ConnectionError, TimeoutError)))
    def api_generate_text(self, llm: Any, prompt: str, system_prompt: str = None, model_name: str = None, **kwargs) -> str:
        """Generate text using the specified language model."""
        if isinstance(llm, OpenAI):
            messages = [
                {"role": "system", "content": system_prompt} if system_prompt else {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ]
            
            # Add OpenRouter specific headers if base_url is OpenRouter
            extra_headers = {}
            if str(getattr(llm, 'base_url', '')).startswith('https://openrouter.ai'):
                model = model_name or 'meta-llama/llama-3.2-1b-instruct:free'
            else:
                model = model_name or 'gpt-4o-mini'
                
            response = llm.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        elif isinstance(llm, CustomHuggingFaceEndpoint):
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            return llm.invoke(full_prompt, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM type: {type(llm)}")

    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=60000,
           retry_on_exception=lambda e: isinstance(e, (requests.exceptions.RequestException, ConnectionError, TimeoutError)))
    def api_generate_embeddings(self, embeddings, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate embeddings for text(s)."""
        if isinstance(texts, str):
            texts = [texts]    
        if isinstance(embeddings, OpenAI):
            response = embeddings.embeddings.create(input=texts, model="text-embedding-ada-002")
            embeddings_list = [item.embedding for item in response.data]
        elif isinstance(embeddings, CustomHuggingFaceInferenceAPIEmbeddings):
            embeddings_list = embeddings.embed_documents(texts)
        elif isinstance(embeddings, CustomTogetherEmbeddings):
            embeddings_list = embeddings.embed_documents(texts)
        else:
            raise ValueError(f"Unsupported embeddings type: {type(embeddings)}")
        
        return embeddings_list[0] if len(texts) == 1 else embeddings_list

    def local_get_huggingface_embedding(self, texts: List[str], model_name: str = "BAAI/bge-large-en-v1.5", batch_size: int = 512) -> List[List[float]]:
        """Generate embeddings locally using a HuggingFace model."""
        from transformers import AutoTokenizer, AutoModel
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).to(device)
        model.eval()

        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings", unit="batch"):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            
            batch_embeddings = outputs.last_hidden_state[:, 0]  # Use CLS token
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)  # Normalize
            all_embeddings.extend(batch_embeddings.cpu().tolist())

        return all_embeddings

    @retry(stop_max_attempt_number=10, wait_exponential_multiplier=1000, wait_exponential_max=60000,
           retry_on_exception=lambda e: isinstance(e, (requests.exceptions.RequestException, ConnectionError, TimeoutError)))
    def api_generate_structured(
        self, 
        llm: Any, 
        system_prompt: str, 
        user_content: List[Dict[str, Any]],
        response_model: Type[BaseModel] = None,
        model_name_override: Optional[str] = None,
        max_tokens_override: Optional[int] = None
    ) -> Union[BaseModel, Dict[str, Any]]:
        """
        Generate structured responses using an OpenAI model, supports multimodal input.
        'user_content' should be a list of dicts, e.g., 
        [{"type": "text", "text": "What's in this image?"}, 
         {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}]
        """
        if not isinstance(llm, OpenAI):
            # Check if any image_url types are in user_content
            has_images = any(item.get("type") == "image_url" for item in user_content)
            if has_images:
                logger.error("Image content provided to api_generate_structured, but LLM is not an OpenAI client. Images will be ignored or cause an error.")
                # Optionally, filter out image content if you want to proceed text-only, or raise ValueError
                # For now, we'll let it proceed and potentially fail at the OpenAI API call if the client type is wrong.
                # A better check would be if the llm client object has vision capabilities.
                # raise ValueError("Image inputs are currently only supported for compatible OpenAI models.")
        
        # Patch client with instructor
        # Note: instructor.patch() should ideally be done once when the client is initialized if possible,
        # but patching here ensures it's applied for this specific call context.
        # If 'llm' is already a patched client, instructor might handle it gracefully or re-patch.
        try:
            patched_client = instructor.patch(llm)
        except Exception as e:
            logger.error(f"Failed to patch LLM client with instructor: {e}. Using unpatched client for structured response.")
            patched_client = llm # Fallback, though response_model might not work as expected.


        enhanced_system_prompt = f"""
        {system_prompt}
        
        Please generate a structured response based on the following user prompt.
        Adhere to the provided response_model schema for your output.
        """
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_content} # user_content is now the list of blocks
        ]

        # Determine model and max_tokens
        default_model = "gpt-4o-mini" # Good default for multimodal
        has_images = any(item.get("type") == "image_url" for item in user_content)
        
        model_to_use = model_name_override or default_model
        
        # If images are present, ensure a vision model is selected (this is a basic check)
        # More robust would be to have a list of known vision models.
        if has_images and not ("gpt-4o" in model_to_use or "vision" in model_to_use or "gpt-4-turbo" == model_to_use):
            logger.warning(f"Images provided, but model '{model_to_use}' might not be optimal for vision. Consider 'gpt-4o-mini', 'gpt-4o', or 'gpt-4-turbo'. Using '{model_to_use}'.")
            # If strict, could force to a vision model: model_to_use = default_model
        
        # Set max_tokens, higher for vision models by default
        # OpenAI's vision models can handle up to 4096 output tokens. Input tokens vary.
        # Vision analysis can be verbose.
        current_max_tokens = max_tokens_override or 4096 

        try:
            response = patched_client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                response_model=response_model,
                max_tokens=current_max_tokens, 
                temperature=0.1 # Added for more deterministic structured output
            )
            return response

        except ValidationError as e: # Pydantic validation error from Instructor
            logger.error(f"Pydantic validation error for structured response (model: {model_to_use}): {e.errors()}", exc_info=True)
            raise
        except openai.APIError as e: # Catch OpenAI specific errors
            logger.error(f"OpenAI API error during structured generation (model: {model_to_use}): {e}", exc_info=True)
            raise
        except Exception as e: # Catch other general errors
            logger.error(f"Generic error during structured generation (model: {model_to_use}): {e}", exc_info=True)
            raise

# Main method for testing
def main():

    from pydantic import BaseModel, Field

    """Demonstration of ModelConfig usage."""
    config = ModelConfig()

    # Initialize language model
    try:
        llm = config.initialize_llm(provider="openai", model_name="gpt-4o-mini")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        return

    # Generate structured response example
    system_prompt = "You are a helpful assistant for generating structured data."
    user_prompt = "Create a structured representation of a task management system with tasks, priorities, and deadlines."

    # Pydantic model for structured response
    class Task(BaseModel):
        task: str = Field(..., title="Task name")
        priority: str = Field(..., title="Task priority")
        deadline: str = Field(..., title="Task deadline")

    try:
        response = config.api_generate_structured(llm, system_prompt, [{"type": "text", "text": user_prompt}], response_model=Task)
        print("Generated Structured Response:")
        print(response)
    except Exception as e:
        logger.error(f"Failed to generate structured response: {e}")

    try:
        response = config.api_generate_text(llm, "What is the capital of France?")
        print("Generated Text Response:")
        print(response)
    except Exception as e:
        logger.error(f"Failed to generate text response: {e}")


if __name__ == "__main__":
    main()
