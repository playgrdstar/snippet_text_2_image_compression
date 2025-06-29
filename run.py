import os
import shutil
import logging
import base64
import asyncio
import math
import csv
from pathlib import Path
from typing import List, Optional, Type, Dict

# --- Import Required Libraries ---
# This script requires several packages. Ensure you have run:
# pip install -r requirements.txt
try:
    from PIL import Image, ImageDraw, ImageFont
    from pydantic import BaseModel, Field, ValidationError
    from dotenv import load_dotenv, find_dotenv
    from openai import OpenAI
    import openai as openai_lib # Use alias to avoid conflict with the OpenAI class
    from retrying import retry
    import instructor
    import tiktoken
except ImportError as e:
    print(f"Error: A required library is not installed. {e}")
    print("Please create a 'requirements.txt' file and run: pip install -r requirements.txt")
    exit()

# --- 1. Script Configuration ---

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

TEXT_INPUT = """
The Keeper of Lost Tides
The inheritance came not as a windfall, but as a summons. Elara Vance, a woman whose life was a quiet composition of card catalogs, archival dust, and the hushed reverence of the British Library's reading rooms, had never heard of her great-aunt Isolde MacLeod, let alone the estate she was now the sole heir to. The solicitor's letter was crisp and formal, speaking of a property named 'Tide-End Manor' on a remote, windswept peninsula on the far north-western coast of Scotland. The name alone felt like a piece of gothic fiction.

Two weeks of meticulous planning, a week of delegating her duties at the library, and a final, jarringly long train journey north, and Elara found herself in a rented Land Rover, navigating roads that narrowed from tarmac to gravel, and finally to little more than a pair of muddy ruts carving through an expanse of heather and sea-tormented grass. The air itself had changed; it was sharp, saline, and carried on it the cries of gulls and a profound, unnerving silence.

Tide-End Manor did not disappoint the gothic expectations its name had conjured. It was a severe, granite-built house, standing defiant on a promontory with the slate-grey waters of the North Atlantic churning on three sides. It seemed less a piece of architecture and more a geological feature, an extension of the cliff face it clung to. Its windows were dark eyes staring out at the unforgiving sea, its stone walls streaked with green lichen and the white stains of seabird droppings. The only sound, apart from the wind and the waves, was the rhythmic clang of a rusted bell-pull against the heavy oak door.

The solicitor, a local man named Mr. Finlay with a face as kind and craggy as the surrounding landscape, had left the keys for her under a loose flagstone. Stepping inside was like stepping into a sealed jar. The air was still and thick with the scent of salt, beeswax, and something else, something older—the scent of cold ashes and secrets. Dust motes danced like frantic sprites in the thin shafts of light that managed to pierce the grime-cates windows. Furniture huddled under white dust sheets, their forms like sleeping beasts in the gloom.

Elara's archivist instincts, usually reserved for vellum manuscripts and forgotten correspondence, kicked in. This was not just a house; it was a primary source, an archive of a life she knew nothing about. Her first week was spent in a frenzy of organization. She aired out the rooms, cataloged the contents of cupboards, and created a rudimentary floor plan, discovering a warren of forgotten pantries, sea-facing drawing rooms, and bedrooms where the wallpaper peeled like sunburnt skin.

The heart of the house, she soon discovered, was not its grand, sea-damaged ballroom or its cavernous kitchen, but a small, unassuming study tucked away at the top of a spiral staircase in the west wing. This had clearly been the domain of her true predecessor, not her great-aunt Isolde, but a man whose presence permeated every corner of the room: Alistair MacLeod, her great-great-uncle, a name she found on the flyleaf of dozens of books. The room was a shrine to the sea and to knowledge. Sextants, chronometers, and astrolabes lay on shelves next to tide charts and books on marine biology, celestial navigation, and local folklore.

And on the great oak desk, placed squarely in the center as if its owner had just stepped out for a moment, was a set of leather-bound journals. There were five of them, their covers worn smooth by time and handling, their titles embossed in faded gold leaf: The Tide-End Journals, 1888-1892.

They became her world. Alistair's hand was neat, his ink a sepia brown. He was a man of science and poetry, his entries a meticulous blend of tidal measurements, observations of seabird migrations, and philosophical musings inspired by the relentless power of the ocean. He was, Elara learned, a marine biologist and a folklorist, a man fascinated by the line where science and superstition met, where the known world dipped into the abyss of myth. He wrote of his dredging expeditions, of creatures brought up from the depths that defied classification, of the songs of the seals and the stories the local fishermen told in hushed tones around their peat fires.

Elara would spend her days exploring the rugged coastline, matching Alistair's descriptions of hidden coves and sea-carved arches to the landscape around her. In the evenings, curled by the fire with a glass of whisky, she would lose herself in his world, the rhythmic crash of the waves outside a constant percussion to her reading. She learned of his theories on rogue waves, his catalog of local kelp varieties, his fascination with the bioluminescent algae that sometimes set the bay aglow on moonless nights. The journals were a portrait of a brilliant, isolated mind, a man in deep conversation with the sea.

(First embedded clue appears below)

Deep in the second volume, in an entry dated June 1889, Alistair shifted from his scientific observations to a detailed ethnography of the nearby fishing village of Port Sgail, a tiny cluster of cottages a few miles down the coast. He sketched the faces of the fishermen, documented their unique dialect, and recorded their stories. His descriptions were vivid, painting a picture of a hardy, close-knit community. One entry, in particular, stood out for its detail.

"The most singular character in Port Sgail," he wrote, "is undoubtedly the blacksmith, a man who calls himself Mr. Graeme. He is a man of immense physical presence, with arms as thick as mooring ropes and a quiet, watchful demeanor that sets him apart from the more boisterous fishermen. He keeps to himself, his forge glowing at all hours, the rhythmic ring of his hammer a constant, lonely sound. The villagers say he arrived years ago, a castaway from a storm, literally washed ashore with nothing but the clothes on his back. He speaks little of his past, but there is a mark on him that tells a story he will not—a peculiar tattoo on his right forearm, one he makes no effort to hide. It is a serpent, coiled into a perfect circle, devouring its own tail. The artistry is remarkable, clearly the work of a master. The locals, in their simple way, are wary of it, calling it a 'devil's coil' or a 'bad omen.' I see it for what it is—an Ouroboros, an ancient symbol of eternity and cycles. A curious emblem for a simple village blacksmith."

Elara found the description captivating. A man of mystery in a village of forthright folk. She made a mental note to see if the old forge was still standing next time she drove through the village. The entry was followed by pages of notes on net-mending techniques and the local names for different types of fish, and the image of the tattooed blacksmith soon faded, absorbed into the vast sea of Alistair's observations. She continued her reading, progressing through the summer and autumn of 1889, learning of a harsh winter and a particularly dramatic spring tide that flooded the lower cellars of the manor.

The third and fourth journals were filled with more esoteric research. Alistair grew obsessed with a particular legend—the lost treasure of the pirate captain Calico Jack, rumored to have been buried somewhere on this stretch of coastline. But Alistair dismissed this as fanciful nonsense, a story for tourists. He was, however, deeply interested in the historical reality of piracy in the region. He spent months corresponding with archivists in Edinburgh and London, obtaining copies of naval records, trial transcripts, and ship's manifests from the 18th and early 19th centuries.

(Second embedded clue appears below)

This section of the journal was dense, at times tedious. Elara, the professional archivist, appreciated the meticulous nature of his research, but it lacked the personal charm of his earlier entries. He listed vessels, their tonnage, their armament, their captains. He was trying to create a comprehensive history of illicit maritime activity in the North Minch. It was in a long, dry list, compiled in the fourth journal in an entry from late 1891, that a particular name was mentioned. The list was titled 'Known Pirate Vessels and Their Fates, 1780-1840.'

"...The Black Thistle, scuttled off the Isle of Skye, 1821. Captain Ewan 'The Red' MacPhee, hanged in Inverness.

The Siren's Call, wrecked off Stornoway, 1828. Crew presumed lost.

The Sea Serpent, a brigantine of notorious reputation. Last sighted near the Shiant Islands in the winter of 1832 before vanishing in a storm. Its captain, a man of terrifying repute known only as 'Oro,' was said to command with an iron fist. Naval reports speculated he went down with his ship, though his body, nor any wreckage, was ever recovered. The manifest, recovered years later from a captured associate, lists him as the sole commander. The vessel was infamous for its speed and its flag—a simple black banner with a white serpent.

The Fortune's Folly, captured by the Royal Navy, 1835. Captain Henry Giles, transported to Australia..."

Elara's eyes scanned the list, registering the names and dates with a professional detachment. 'Oro.' An unusual name. The entry seemed no more or less significant than any of the others. Alistair had made no special note of it. It was just one more data point in his exhaustive research, one more ghost ship to add to his historical ledger. She turned the page, and the journal moved on to Alistair's theories about the decline of herring stocks, a subject of more immediate concern to the 19th-century inhabitants of Port Sgail. The name of the pirate and his ship were washed away by the tide of new information.

The final journal was different. The tone was more urgent, more personal. Alistair had abandoned his historical research and returned to the sea itself. He wrote of building a diving bell, a dangerous, primitive contraption of iron and reinforced glass, with which he intended to explore the deepwater trenches just off the peninsula. His final entries were filled with frantic, excited notes about the creatures he was seeing, forms of life he believed were entirely new to science.

Then, on a page dated October 5th, 1892, the entries stopped. The rest of the book was blank.

Elara closed the final journal, the silence of the room pressing in on her. Mr. Finlay, the solicitor, had told her the official story. Alistair MacLeod had been lost at sea during a storm while testing his diving apparatus. A tragic accident. A man of science pushing his luck too far.

She looked out the window of the study. The sea was calm today, a vast sheet of hammered silver under a high, pale sky. The journals sat on the desk, their stories told. Yet she felt a strange sense of incompleteness, a loose thread she couldn't quite grasp. The house, the sea, and the journals had given her the story of Alistair's life. But sitting there, with the scent of old paper and salt in the air, she felt the profound, unspoken weight of its secrets. She had organized his life, but she had not yet understood it. The answer to something important, she felt, lay sleeping within the pages she had just read, waiting for the right two tides to meet.

""".strip().replace('\n', ' ')

QUESTION = "Based on the clues embedded in the text, what was the true identity of the village blacksmith, Mr. Graeme?"

# --- Experiment Variables ---
# Add the models you want to test here. The script will use the correct cost logic for each.
MODELS_TO_TEST = ["gpt-4o-mini", "gpt-4o", "gpt-4.1", "o1"]

FONTS_TO_TEST = {
    "arial": "arial.ttf",
    "comic sans ms": "comic.ttf",
    "impact": "impact.ttf",
}
FONT_SIZES_TO_TEST = [6,8,12]

IMAGE_OUTPUT_DIR = Path("./generated_text_images")
if IMAGE_OUTPUT_DIR.exists(): shutil.rmtree(IMAGE_OUTPUT_DIR)
IMAGE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Pricing and Model Logic ---
# Costs are per 1 MILLION tokens. Updated based on the latest pricing sheet.
MODEL_CONFIG = {
    "gpt-4o": {
        "cost_logic": "tile_512px",
        "input_cost": 2.50,
        "output_cost": 10.0,
        "tile_base": 85,
        "tile_cost": 170,
    },
    "gpt-4o-mini": {
        "cost_logic": "tile_512px",
        "input_cost": 0.15,
        "output_cost": 0.60,
        "tile_base": 2833,
        "tile_cost": 5667,
    },
    "gpt-4.1": {
        "cost_logic": "tile_512px",
        "input_cost": 2.00,
        "output_cost": 8.00,
        "tile_base": 85,
        "tile_cost": 170,
    },
    "o1": {
        "cost_logic": "tile_512px",
        "input_cost": 15.00,
        "output_cost": 60.00,
        "tile_base": 75,
        "tile_cost": 150,
    },
    "default": {
        "cost_logic": "tile_512px",
        "input_cost": 0.15, # Defaulting to cheaper model
        "output_cost": 0.60,
        "tile_base": 85,
        "tile_cost": 170,
        "patch_multiplier": 1.0,
    }
}

# --- 2. Core Logic (Adapted from your config.py) ---

def load_env_vars():
    """Load environment variables from a .env file."""
    if find_dotenv(): load_dotenv(find_dotenv()); logger.info("Loaded .env file.")
    else: logger.warning(".env file not found.")

class ModelConfig:
    """Manages API configuration and client initialization."""
    def __init__(self):
        load_env_vars()
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key: logger.error("FATAL: OPENAI_API_KEY not found.")
        else: logger.info("OpenAI API Key loaded successfully.")

    def initialize_llm(self) -> Optional[OpenAI]:
        """Initializes the OpenAI client."""
        return OpenAI(api_key=self.openai_api_key) if self.openai_api_key else None

    @retry(stop_max_attempt_number=1, wait_exponential_multiplier=1000, wait_exponential_max=10000)
    def api_generate_structured(self, llm: OpenAI, model_name: str, system_prompt: str, user_content: List[dict], response_model: Type[BaseModel]) -> Optional[BaseModel]:
        """Generates a structured response using a specific OpenAI model."""
        try:
            patched_client = instructor.patch(llm)
            response = patched_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                response_model=response_model,
                max_completion_tokens=5000,
                # temperature=0.0
            )
            return response
        except (ValidationError, openai_lib.APIError, Exception) as e:
            logger.error(f"API call with model {model_name} failed: {e}", exc_info=True)
            raise

# --- 3. Helper Functions & Real LLM Service ---

def create_text_image(text: str, font_path: str, font_size: int, output_path: Path):
    """
    Generates a PNG image from a block of text, optimized for tile-based models.
    It renders the text at a high resolution and then resizes it to have a short
    side of 768px, which is optimal for models like gpt-4o.
    """
    # 1. Render text at a higher resolution (e.g., 4x) for better quality before downscaling.
    scaling_factor = 4
    # Use a fixed, sensible width for the high-res canvas. This value is chosen
    # to encourage the text to wrap into a more balanced aspect ratio, which allows
    # it to be optimally resized (short side of 768px) without the other
    # dimension exceeding its 2048px limit.
    high_res_width = 1536 * scaling_factor
    high_res_font_size = font_size * scaling_factor

    try:
        font = ImageFont.truetype(font_path, high_res_font_size)
    except IOError:
        logger.warning(f"Font not found: '{font_path}'. Using default.")
        font = ImageFont.load_default(size=high_res_font_size)
    
    img_temp = Image.new('RGB', (1, 1))
    draw_temp = ImageDraw.Draw(img_temp)

    lines, current_line = [], ""
    for word in text.split():
        if draw_temp.textlength(f"{current_line} {word}", font=font) <= high_res_width - (40 * scaling_factor):
            current_line += f" {word}"
        else:
            lines.append(current_line.strip())
            current_line = word
    lines.append(current_line.strip())

    line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1] + (5 * scaling_factor)
    high_res_height = int((len(lines) * line_height) + (20 * scaling_factor))
    
    img = Image.new('RGB', (high_res_width, high_res_height), color='white')
    d = ImageDraw.Draw(img)
    y = 10 * scaling_factor
    for line in lines:
        d.text((20 * scaling_factor, y), line, fill='black', font=font)
        y += line_height

    # 2. Resize the high-res image to optimal dimensions for the API.
    # The goal is a short side of 768px and a long side <= 2048px.
    w_initial, h_initial = img.size
    
    if w_initial == 0 or h_initial == 0:
        logger.warning(f"Generated an empty or invalid image: {output_path.name}")
        img.save(output_path)
        return

    is_landscape = w_initial > h_initial
    short_side, long_side = (h_initial, w_initial) if is_landscape else (w_initial, h_initial)

    # Calculate new long side if we scale the short side to 768px
    ratio = 768 / short_side
    new_long_side = int(long_side * ratio)

    if new_long_side > 2048:
        # Aspect ratio is too extreme, long side is the constraint.
        final_long = 2048
        downscale_ratio = 2048 / long_side
        final_short = int(short_side * downscale_ratio)
    else:
        # Aspect ratio is fine, short side is the constraint.
        final_short = 768
        final_long = new_long_side

    final_w = final_long if is_landscape else final_short
    final_h = final_short if is_landscape else final_long
        
    logger.info(f"Optimizing image. Original: {w_initial}x{h_initial}, Resizing to: {final_w}x{final_h}")
    img = img.resize((final_w, final_h), Image.Resampling.LANCZOS)

    img.save(output_path)
    logger.info(f"Generated optimized image: {output_path.name}")

def calculate_levenshtein_similarity(original: str, extracted: str) -> float:
    """Calculates a similarity percentage between two strings."""
    if not original or not extracted: return 0.0
    s1, s2 = original.lower(), extracted.lower()
    dist = [[j for j in range(len(s2) + 1)] for i in range(len(s1) + 1)]
    for i in range(1, len(s1) + 1): dist[i][0] = i
    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dist[i][j] = min(dist[i-1][j] + 1, dist[i][j-1] + 1, dist[i-1][j-1] + cost)
    distance = dist[len(s1)][len(s2)]
    return (1 - distance / max(len(s1), len(s2))) * 100

def calculate_text_token_cost(text: str, model: str, is_output: bool = False) -> tuple[int, float]:
    """Calculates token count and cost for a text string."""
    config = MODEL_CONFIG.get(model, MODEL_CONFIG["default"])
    try: encoding = tiktoken.encoding_for_model(model)
    except KeyError: encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(text))
    cost_rate = config["output_cost"] if is_output else config["input_cost"]
    cost = (num_tokens / 1_000_000) * cost_rate
    return num_tokens, cost

def _calculate_cost_tile_512px(w: int, h: int, model: str) -> int:
    """Cost logic for models like gpt-4o."""
    config = MODEL_CONFIG.get(model, MODEL_CONFIG["default"])
    
    # 1. Scale to fit in a 2048px x 2048px square
    if max(w, h) > 2048:
        ratio = 2048 / max(w, h)
        w, h = int(w * ratio), int(h * ratio)
    
    # 2. Scale so that the image's shortest side is 768px long.
    # This is not conditional; it should always be scaled up or down.
    ratio = 768 / min(w, h)
    w, h = int(w * ratio), int(h * ratio)

    # 3. Count the number of 512px squares
    tiles = math.ceil(w / 512) * math.ceil(h / 512)
    return config["tile_base"] + (tiles * config["tile_cost"])

def _calculate_cost_patch_32px(w: int, h: int, model: str) -> int:
    """Cost logic for models like o4-mini, updated to match documentation."""
    # 1. Calculate patches needed for the original image
    patches_w = math.ceil(w / 32)
    patches_h = math.ceil(h / 32)
    total_patches = patches_w * patches_h

    if total_patches <= 1536:
        image_tokens = total_patches
    else:
        # 2. If > 1536, scale down the image.
        # This logic follows the complex example in the OpenAI documentation.
        # First, a primary shrink factor based on raw pixel area.
        shrink_factor = math.sqrt((1536 * 32**2) / (w * h))
        w_shrunk = w * shrink_factor
        h_shrunk = h * shrink_factor

        # Then, a secondary adjustment to fit one dimension perfectly.
        patches_w_float = w_shrunk / 32
        secondary_shrink_factor = math.floor(patches_w_float) / patches_w_float

        w_final = int(w_shrunk * secondary_shrink_factor)
        h_final = int(h_shrunk * secondary_shrink_factor)

        # Final patch calculation
        patches_w_final = math.ceil(w_final / 32)
        patches_h_final = math.ceil(h_final / 32)
        image_tokens = patches_w_final * patches_h_final
    
    # 3. Apply model-specific multiplier to get final token cost
    config = MODEL_CONFIG.get(model, MODEL_CONFIG["default"])
    multiplier = config.get("patch_multiplier", 1.0)
    return int(image_tokens * multiplier)

def calculate_image_token_cost(image_path: Path, model_name: str, detail: str) -> tuple[int, float]:
    """Calculates image token cost by routing to the correct logic based on the model."""
    config = MODEL_CONFIG.get(model_name, MODEL_CONFIG["default"])
    logic_type = config["cost_logic"]
    
    # Handle low detail case for tile-based models first
    if logic_type == "tile_512px" and detail == "low":
        num_tokens = config["tile_base"]
    else:
        try:
            with Image.open(image_path) as img:
                w, h = img.size
                if logic_type == "tile_512px":
                    num_tokens = _calculate_cost_tile_512px(w, h, model_name)
                elif logic_type == "patch_32px":
                    num_tokens = _calculate_cost_patch_32px(w, h, model_name)
                else:
                    logger.error(f"Unknown cost logic '{logic_type}' for model '{model_name}'.")
                    return 0, 0.0
        except Exception as e:
            logger.error(f"Could not calculate image cost for {image_path}: {e}"); return 0, 0.0

    cost = (num_tokens / 1_000_000) * config["input_cost"]
    return num_tokens, cost

class LLMAnswerResponse(BaseModel):
    answer: str = Field(description="The answer to the user's question, based on the text in the image.")

class RealLLMService:
    """Service to connect to a real LLM."""
    def __init__(self):
        self.config = ModelConfig(); self.client = self.config.initialize_llm()

    async def get_answer_from_image(self, image_path: Path, model_name: str, detail: str) -> Optional[str]:
        if not self.client: logger.error("LLM Client not initialized."); return None
        image_data_uri = f"data:image/png;base64,{base64.b64encode(image_path.read_bytes()).decode('utf-8')}"
        system_prompt = "You are an expert at reading text from an image and answering questions based on it. Your sole task is to understand the text from the provided image and answer the user's question. If you cannot find the answer, say 'I don't know.'"
        user_content = [{"type": "text", "text": QUESTION}, {"type": "image_url", "image_url": {"url": image_data_uri, "detail": detail}}]
        try:
            response = await asyncio.to_thread(self.config.api_generate_structured, self.client, model_name, system_prompt, user_content, LLMAnswerResponse)
            return response.answer if isinstance(response, LLMAnswerResponse) else None
        except Exception: return None

    async def get_answer_from_text(self, context_text: str, question: str, model_name: str) -> Optional[str]:
        """Gets an answer from the LLM based on plain text context and a question."""
        if not self.client: logger.error("LLM Client not initialized."); return None
        system_prompt = "You are an expert at reading text and answering questions based on it. Your sole task is to understand the provided text and answer the user's question. If you cannot find the answer, say 'I don't know.'"
        
        user_content = [{"type": "text", "text": f"CONTEXT:\n---\n{context_text}\n---\n\nQUESTION: {question}"}]
        try:
            response = await asyncio.to_thread(self.config.api_generate_structured, self.client, model_name, system_prompt, user_content, LLMAnswerResponse)
            return response.answer if isinstance(response, LLMAnswerResponse) else None
        except Exception: return None

# --- 4. Main Experiment Execution ---

def save_results_to_csv(results: List[Dict], output_path: Path):
    """Saves the experiment results to a CSV file."""
    if not results:
        logger.warning("No results to save.")
        return

    # Use the keys from the first result as headers
    fieldnames = results[0].keys()
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        logger.info(f"Results successfully saved to {output_path}")
    except IOError as e:
        logger.error(f"Failed to save results to CSV: {e}")


async def main():
    print("\nStarting Multi-Model LLM OCR & Cost Experiment...")
    print("=" * 60)
    
    llm_service = RealLLMService()
    if not llm_service.client:
        print("\nExiting script. Please set up your .env file with a valid OPENAI_API_KEY.")
        return

    results = []
    for model_name in MODELS_TO_TEST:
        print(f"\n===== TESTING MODEL: {model_name.upper()} =====")
        
        # --- Text-based VQA Baseline ---
        print("\n--- Getting baseline answer from plain text ---")
        baseline_answer = await llm_service.get_answer_from_text(TEXT_INPUT, QUESTION, model_name)

        # Calculate the true baseline cost for the text-to-answer transaction
        text_input_for_baseline = f"CONTEXT:\n---\n{TEXT_INPUT}\n---\n\nQUESTION: {QUESTION}"
        _, input_cost = calculate_text_token_cost(text_input_for_baseline, model=model_name, is_output=False)
        
        baseline_output_cost = 0.0
        if baseline_answer:
            _, baseline_output_cost = calculate_text_token_cost(baseline_answer, model=model_name, is_output=True)
        
        total_text_cost = input_cost + baseline_output_cost
        
        print(f"Text-to-Text Baseline Cost: ${total_text_cost:.6f}")
        print(f"  - Baseline Answer: {baseline_answer.strip() if baseline_answer else 'FAILED'}")
        print("-" * 60)
        
        for font_name, font_file in FONTS_TO_TEST.items():
            for size in FONT_SIZES_TO_TEST:
                # For patch-based models, "detail" level doesn't apply, so we run once with "auto".
                model_logic = MODEL_CONFIG.get(model_name, MODEL_CONFIG["default"])["cost_logic"]
                detail_levels = ["low", "high"] if model_logic == "tile_512px" else ["auto"]
                # detail_levels = ["high"] if model_logic == "tile_512px" else ["auto"]
                for detail in detail_levels:
                    print(f"\n--- Testing Font: '{font_name}', Size: {size}, Detail: '{detail}' ---")
                    image_path = IMAGE_OUTPUT_DIR / f"{model_name}_{font_name}_sz{size}_detail_{detail}.png"
                    create_text_image(TEXT_INPUT, font_file, size, image_path)
                    
                    image_tokens, image_cost = calculate_image_token_cost(image_path, model_name, detail)
                    print(f"  - Image Input Cost: {image_tokens} tokens = ${image_cost:.6f}")

                    print("  - Sending image to LLM for answering...")
                    answer = await llm_service.get_answer_from_image(image_path, model_name, detail)
                    
                    if answer:
                        output_tokens, output_cost = calculate_text_token_cost(answer, model=model_name, is_output=True)
                        total_cost = image_cost + output_cost
                        print(f"  - Answer received. Output: {output_tokens} tokens = ${output_cost:.6f}")
                        print(f"  - TOTAL VQA COST (Image Input + Text Output): ${total_cost:.6f}")
                        
                        print(f"  - Answer: {answer.strip()}")
                        results.append({"model": model_name, "font": font_name, "size": size, "detail": detail, "answer": answer, "cost": total_cost, "baseline_cost": total_text_cost, "baseline_answer": baseline_answer})
                    else:
                        print("  - FAILED to get an answer. Check logs for API errors.")
                        results.append({"model": model_name, "font": font_name, "size": size, "detail": detail, "answer": "FAILED", "cost": image_cost, "baseline_cost": total_text_cost, "baseline_answer": baseline_answer})

    print("\n\n" + "=" * 120)
    print(" " * 48 + "Experiment Summary" + " " * 48)
    print("=" * 120)
    sorted_results = sorted(results, key=lambda x: (x['model'], x['font'], x['size']))
    header = f"{'Model':<14} | {'Font':<10} | {'Size':<5} | {'Detail':<7} | {'VQA Cost ($)':<15} | {'Baseline Cost ($)':<18} | {'Answer Preview'}"
    print(header)
    print("-" * len(header))
    for res in sorted_results:
        answer_preview = res['answer'].replace('\n', ' ').strip()
        if len(answer_preview) > 60:
            answer_preview = answer_preview[:57] + "..."
        print(f"{res['model']:<14} | {res['font']:<10} | {res['size']:<5} | {res['detail']:<7} | {f'{res["cost"]:.6f}':<15} | {f'{res["baseline_cost"]:.6f}':<18} | {answer_preview}")
    print("=" * len(header))

    # Save the results to a CSV file
    csv_output_path = Path("./experiment_results.csv")
    save_results_to_csv(sorted_results, csv_output_path)
    
    print(f"\nGenerated images are stored in: {IMAGE_OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    asyncio.run(main())
