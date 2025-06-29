# Cost comparison - Images instead of text for QA

The script systematically generates images of a given text in different fonts and sizes, then asks a model to answer a question based on the image. It compares the model's answer and the associated cost against a text-only baseline.

## Features

-   **Image Generation**: Creates PNG images from a source text using different fonts (`Arial`, `Comic Sans MS`, `Impact`) and font sizes.
-   **Optimal Image Sizing**: Automatically resizes generated images to the optimal dimensions for OpenAI's tile-based vision models (e.g., GPT-4o), ensuring the shortest side is 768px for maximum quality.
-   **Multi-Model Testing**: Runs the VQA experiment across a configurable list of OpenAI models (e.g., `gpt-4o`, `gpt-4o-mini`, `gpt-4.1`, `o1`).
-   **Cost Calculation**: Accurately calculates the cost of each VQA task (image input tokens + text output tokens) based on up-to-date OpenAI pricing.
-   **Baseline Comparison**: For each model, it also performs a text-only question-answering task to establish a baseline for cost and answer quality.
-   **Results Summarization**: Outputs a summary table to the console and saves a detailed report of all experimental runs to `experiment_results.csv`.

## How to Run

### 1. Prerequisites

-   Python 3.7+
-   An OpenAI API Key

### 2. Setup

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```

2.  **Install Dependencies**
    Create a `requirements.txt` file with the necessary libraries and install them.
    ```
    # requirements.txt
    openai
    instructor
    python-dotenv
    Pillow
    retrying
    tiktoken
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set Up Environment Variables**
    Create a file named `.env` in the project root and add your OpenAI API key:
    ```
    # .env
    OPENAI_API_KEY="sk-..."
    ```

4.  **Add Fonts**
    The script requires TrueType Font (`.ttf`) files. Place the required font files (e.g., `arial.ttf`, `comic.ttf`, `impact.ttf`) in the project's root directory or update the paths in the `FONTS_TO_TEST` dictionary in `run.py`.

### 3. Execute the Script

Run the experiment from your terminal:
```bash
python run.py
```

## Configuration

You can customize the experiment by modifying the following variables at the top of `run.py`:

-   `TEXT_INPUT`: The source text to be rendered into an image.
-   `QUESTION`: The question that the LLM will be asked to answer based on the text.
-   `MODELS_TO_TEST`: A list of strings specifying which OpenAI models to test.
-   `FONTS_TO_TEST`: A dictionary mapping a font name to its `.ttf` file.
-   `FONT_SIZES_TO_TEST`: A list of integers for the different font sizes to use.

## Output

-   **Console**: Prints real-time progress and a final summary table of the results.
-   `generated_text_images/`: A directory containing all the PNG images generated for the experiment.
-   `experiment_results.csv`: A CSV file containing the detailed results for every combination of model, font, size, and detail level, including costs and the answers provided by the model.

## Question and Answer Task

For this experiment, to test the ability of the LLM to still answer the question even with images (which indirectly tests their text extraction capabilities), we hide the answer (which requires two pieces of information in separate places) quite obviously and even signpost where it is. They just have to be able to extract the text, connect the two pieces of information, and answer the question:

###QUESTION###
Based on the clues embedded in the text, what was the true identity of the village blacksmith, Mr. Graeme?

###ANSWER###
Any answer Ouros or Sea Serpent should be correct


###TEXT###
####The Keeper of Lost Tides####
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
