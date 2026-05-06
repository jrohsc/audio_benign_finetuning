#!/work/anon/miniconda3/envs/audio-llm/bin/python3
"""
Generate benign instruction dataset that mimics AdvBench wording style.

AdvBench uses patterns like:
- "Write a tutorial on how to..."
- "Provide step-by-step instructions for..."
- "Create a guide on..."
- "Develop a [thing] that..."

This script generates ~6000 benign instructions using the same linguistic patterns
but with completely harmless topics (cooking, gardening, learning, fitness, etc.).

Creates two versions:
1. American English only: ~6000 samples with USA accent
2. Multi-accent: 550 samples per accent × 11 accents = 6050 total

Usage:
    python generate_benign_instructions.py                      # Generate both versions
    python generate_benign_instructions.py --version usa_only   # Only American English
    python generate_benign_instructions.py --version accents    # Only multi-accent
    python generate_benign_instructions.py --dry_run            # Preview without generating
"""

import argparse
import asyncio
import json
import random
from pathlib import Path
from tqdm.asyncio import tqdm
import edge_tts

# Configuration
DEFAULT_SEED = 42
MAX_CONCURRENT = 10

# Paths
BASE_DIR = Path("/work/anon/audio_benign_finetuning")

# Shared audio directories
SHARED_AUDIO_DIR_USA = BASE_DIR / "shared_audio/benign_instructions_usa"
SHARED_AUDIO_DIR_ACCENTS = BASE_DIR / "shared_audio/benign_instructions_accents"

# JSON/JSONL output directories
AF_DATA_DIR_USA = BASE_DIR / "audio-flamingo/data/benign_instructions_usa"
AF_DATA_DIR_ACCENTS = BASE_DIR / "audio-flamingo/data/benign_instructions_accents"
KIMI_DATA_DIR_USA = BASE_DIR / "Kimi-Audio/finetune_codes/data/benign_instructions_usa"
KIMI_DATA_DIR_ACCENTS = BASE_DIR / "Kimi-Audio/finetune_codes/data/benign_instructions_accents"

# Target sample counts
NUM_SAMPLES_USA = 6000  # All American English
NUM_SAMPLES_PER_ACCENT = 550  # Per accent for multi-accent version

# VoiceBench accent to Edge-TTS voice mapping
ACCENT_VOICES = {
    "aus": "en-AU-NatashaNeural",      # Australian
    "gbr": "en-GB-SoniaNeural",         # British
    "ind_n": "en-IN-NeerjaNeural",      # Indian (North)
    "ind_s": "en-IN-PrabhatNeural",     # Indian (South)
    "irl": "en-IE-EmilyNeural",         # Irish
    "kenya": "en-KE-AsiliaNeural",      # Kenyan
    "nga": "en-NG-EzinneNeural",        # Nigerian
    "nzl": "en-NZ-MollyNeural",         # New Zealand
    "phl": "en-PH-RosaNeural",          # Philippines
    "usa": "en-US-JennyNeural",         # American
    "zaf": "en-ZA-LeahNeural",          # South African
}

REGIONS = list(ACCENT_VOICES.keys())

# ============================================================================
# BENIGN INSTRUCTION TEMPLATES (mimicking AdvBench style)
# ============================================================================

# Template patterns extracted from AdvBench
TEMPLATES = [
    # Tutorial patterns
    "Write a tutorial on how to {activity}",
    "Create a tutorial on how to {activity}",
    "Provide a tutorial on how to {activity}",
    "Write a step-by-step tutorial for {activity}",

    # Step-by-step instruction patterns
    "Provide step-by-step instructions for {activity}",
    "Give step-by-step instructions for how to {activity}",
    "Provide detailed instructions on how to {activity}",
    "Give detailed instructions on how to {activity}",
    "Outline the steps for {activity}",
    "Provide a step-by-step guide on how to {activity}",

    # Guide patterns
    "Write a guide on how to {activity}",
    "Create a guide for {activity}",
    "Provide a guide on how to {activity}",
    "Offer a guide on how to {activity}",

    # Explanation patterns
    "Explain how to {activity}",
    "Describe how to {activity}",
    "Describe in detail how to {activity}",

    # Development/Design patterns
    "Develop a plan for {activity}",
    "Outline a plan to {activity}",
    "Design a plan for {activity}",
    "Create a plan for {activity}",

    # Script/Program patterns (benign)
    "Write a script that {tech_activity}",
    "Develop a program that {tech_activity}",
    "Create a program that {tech_activity}",
    "Design a program that {tech_activity}",

    # App/Website patterns (benign)
    "Develop an app that {app_activity}",
    "Design an app that {app_activity}",
    "Create a website that {website_activity}",
    "Design a website that {website_activity}",
    "Develop a website that {website_activity}",

    # Teaching patterns
    "Teach how to {activity}",
    "Teach people how to {activity}",
    "Instruct someone on how to {activity}",

    # Content creation patterns
    "Write a blog post about {topic}",
    "Create a video tutorial on how to {activity}",
    "Produce a podcast episode about {topic}",
    "Generate a tutorial on how to {activity}",

    # Other patterns
    "Demonstrate how to {activity}",
    "Provide instructions on how to {activity}",
    "Provide a recipe for {recipe}",
    "Provide a list of {list_item}",
]

# ============================================================================
# BENIGN TOPICS AND ACTIVITIES
# ============================================================================

# General activities (for {activity} placeholder)
ACTIVITIES = [
    # Cooking & Food
    "bake a chocolate cake from scratch",
    "make homemade pasta",
    "prepare a traditional Italian pizza",
    "cook a healthy vegetarian meal",
    "make fresh bread at home",
    "prepare sushi rolls",
    "grill the perfect steak",
    "make homemade ice cream",
    "brew coffee like a barista",
    "ferment vegetables for probiotics",
    "make cheese at home",
    "prepare a five-course dinner",
    "cook authentic Thai curry",
    "bake sourdough bread",
    "make fresh fruit preserves",
    "prepare a traditional Thanksgiving dinner",
    "make homemade chocolate truffles",
    "cook perfect scrambled eggs",
    "prepare a healthy smoothie bowl",
    "make artisan bread",

    # Gardening & Plants
    "start a vegetable garden",
    "grow tomatoes indoors",
    "create a compost bin",
    "plant a flower garden",
    "maintain a herb garden",
    "grow succulents",
    "start seeds indoors",
    "prune fruit trees",
    "create a butterfly garden",
    "build raised garden beds",
    "grow mushrooms at home",
    "maintain an indoor garden",
    "create a vertical garden",
    "grow organic vegetables",
    "plant a pollinator garden",
    "care for orchids",
    "grow hydroponic lettuce",
    "create a rock garden",
    "maintain a bonsai tree",
    "grow herbs on a windowsill",

    # Home Improvement & DIY
    "paint a room like a professional",
    "install floating shelves",
    "repair a leaky faucet",
    "build a bookshelf",
    "tile a bathroom floor",
    "install a ceiling fan",
    "refinish hardwood floors",
    "build a deck",
    "install crown molding",
    "weatherstrip doors and windows",
    "build a shed",
    "install a backsplash",
    "repair drywall",
    "build custom cabinets",
    "install a smart thermostat",
    "build a fire pit",
    "create a home office",
    "install recessed lighting",
    "build a treehouse",
    "refinish furniture",

    # Arts & Crafts
    "knit a sweater",
    "create watercolor paintings",
    "make pottery on a wheel",
    "sew a quilt",
    "crochet a blanket",
    "do calligraphy",
    "make candles at home",
    "create jewelry",
    "do paper mache crafts",
    "make soap from scratch",
    "paint with acrylics",
    "do embroidery",
    "make leather crafts",
    "create stained glass art",
    "do woodcarving",
    "make origami",
    "create macrame wall hangings",
    "do screen printing",
    "make mosaic art",
    "create resin art",

    # Music & Performance
    "play guitar for beginners",
    "learn piano basics",
    "read sheet music",
    "tune a guitar",
    "play drums",
    "sing with proper technique",
    "write song lyrics",
    "compose music",
    "play ukulele",
    "learn music theory",
    "produce electronic music",
    "mix audio tracks",
    "record vocals at home",
    "play violin",
    "learn to DJ",
    "play harmonica",
    "write a melody",
    "arrange music for a band",
    "play bass guitar",
    "learn beatboxing",

    # Fitness & Health
    "start a running routine",
    "do yoga for beginners",
    "build muscle with bodyweight exercises",
    "practice meditation",
    "improve flexibility",
    "train for a marathon",
    "do pilates at home",
    "create a workout schedule",
    "improve posture",
    "practice mindfulness",
    "do strength training",
    "stretch properly before exercise",
    "do high-intensity interval training",
    "practice breathing exercises",
    "improve balance and coordination",
    "do tai chi",
    "create a healthy meal plan",
    "track fitness progress",
    "do swimming exercises",
    "practice stress management",

    # Technology & Computing
    "set up a home network",
    "build a personal computer",
    "create a website from scratch",
    "learn Python programming",
    "set up a home server",
    "configure a router",
    "create a mobile app",
    "learn data analysis",
    "set up cloud storage",
    "automate home tasks",
    "create digital art",
    "edit videos professionally",
    "set up a podcast studio",
    "learn machine learning basics",
    "create 3D models",
    "set up a smart home",
    "learn web development",
    "create animations",
    "edit photos professionally",
    "set up a streaming setup",

    # Languages & Education
    "learn a new language",
    "improve English writing skills",
    "study for exams effectively",
    "memorize vocabulary",
    "practice public speaking",
    "write a research paper",
    "improve reading comprehension",
    "learn sign language",
    "take effective notes",
    "prepare for a job interview",
    "write a compelling resume",
    "study mathematics",
    "learn history effectively",
    "improve critical thinking",
    "write persuasive essays",
    "learn speed reading",
    "study science concepts",
    "prepare presentations",
    "learn debate skills",
    "improve memory retention",

    # Finance & Business
    "create a budget",
    "start saving for retirement",
    "invest in stocks",
    "manage personal finances",
    "start a small business",
    "write a business plan",
    "manage time effectively",
    "negotiate a salary",
    "build professional networks",
    "manage a team",
    "create financial reports",
    "plan for taxes",
    "invest in real estate",
    "start freelancing",
    "create passive income",
    "manage debt effectively",
    "build an emergency fund",
    "diversify investments",
    "start a side business",
    "improve productivity",

    # Travel & Adventure
    "plan a backpacking trip",
    "travel on a budget",
    "pack efficiently for travel",
    "navigate a foreign city",
    "learn travel photography",
    "plan a road trip",
    "find cheap flights",
    "travel solo safely",
    "camp in the wilderness",
    "hike a mountain trail",
    "plan an international trip",
    "travel with children",
    "book affordable accommodations",
    "travel sustainably",
    "prepare for long flights",
    "navigate public transportation",
    "travel with pets",
    "plan adventure activities",
    "explore national parks",
    "travel during off-season",

    # Pets & Animals
    "train a puppy",
    "care for a new kitten",
    "set up an aquarium",
    "train a dog basic commands",
    "groom a pet at home",
    "care for a pet rabbit",
    "set up a bird cage",
    "train a parrot to talk",
    "care for reptiles",
    "socialize a new pet",
    "prepare homemade pet food",
    "handle pet emergencies",
    "introduce pets to each other",
    "care for senior pets",
    "train cats to use a litter box",
    "set up a hamster habitat",
    "care for fish properly",
    "understand pet behavior",
    "travel with pets safely",
    "adopt a rescue animal",

    # Photography & Media
    "take professional photos",
    "edit photos in Photoshop",
    "shoot portrait photography",
    "do landscape photography",
    "create a photo portfolio",
    "do night photography",
    "shoot macro photography",
    "edit videos in Premiere Pro",
    "create YouTube content",
    "do food photography",
    "shoot real estate photos",
    "create time-lapse videos",
    "do wildlife photography",
    "create social media content",
    "shoot wedding photography",
    "do product photography",
    "create vlogs",
    "shoot drone footage",
    "do street photography",
    "create documentary films",

    # Science & Nature
    "identify local plants",
    "observe birds in nature",
    "study astronomy basics",
    "do simple science experiments",
    "understand weather patterns",
    "identify rocks and minerals",
    "observe wildlife ethically",
    "study marine biology basics",
    "do citizen science projects",
    "understand ecosystems",
    "study botany basics",
    "observe insects and bugs",
    "understand climate science",
    "study geology basics",
    "do nature journaling",
    "understand chemistry basics",
    "study physics concepts",
    "observe stars and planets",
    "understand biology basics",
    "do environmental conservation",

    # Social & Community
    "organize community events",
    "volunteer effectively",
    "plan fundraising activities",
    "lead a community group",
    "organize neighborhood cleanup",
    "mentor young people",
    "plan charity drives",
    "build community gardens",
    "organize book clubs",
    "host dinner parties",
    "plan surprise parties",
    "organize sports leagues",
    "create support groups",
    "plan family reunions",
    "organize potluck dinners",
    "lead community workshops",
    "plan cultural events",
    "organize recycling programs",
    "create neighborhood watch",
    "plan holiday celebrations",
]

# Tech activities (for {tech_activity} placeholder)
TECH_ACTIVITIES = [
    "organizes files automatically",
    "tracks daily habits",
    "manages to-do lists",
    "converts file formats",
    "creates backups automatically",
    "monitors system performance",
    "generates reports from data",
    "automates email responses",
    "schedules social media posts",
    "analyzes text sentiment",
    "transcribes audio to text",
    "compresses images efficiently",
    "extracts data from PDFs",
    "creates digital flashcards",
    "generates random passwords",
    "tracks time spent on tasks",
    "converts currencies",
    "calculates statistics",
    "sorts data alphabetically",
    "filters duplicate entries",
    "formats documents consistently",
    "validates email addresses",
    "generates QR codes",
    "creates calendar reminders",
    "synchronizes folders",
    "renames files in bulk",
    "extracts text from images",
    "creates simple databases",
    "logs daily activities",
    "generates word clouds",
]

# App activities (for {app_activity} placeholder)
APP_ACTIVITIES = [
    "helps users track their fitness goals",
    "reminds users to drink water",
    "tracks daily meditation progress",
    "helps users learn new vocabulary",
    "manages personal budgets",
    "tracks reading habits",
    "helps plan meals for the week",
    "tracks sleep patterns",
    "helps users practice gratitude",
    "manages grocery shopping lists",
    "tracks plant watering schedules",
    "helps users learn music",
    "tracks carbon footprint",
    "helps plan travel itineraries",
    "manages recipe collections",
    "tracks daily step counts",
    "helps users learn languages",
    "manages household chores",
    "tracks mood and emotions",
    "helps plan study schedules",
    "manages pet care routines",
    "tracks medication schedules",
    "helps practice mindfulness",
    "manages contact lists",
    "tracks hobby progress",
    "helps organize home inventory",
    "manages event planning",
    "tracks savings goals",
    "helps practice pronunciation",
    "manages workout routines",
]

# Website activities (for {website_activity} placeholder)
WEBSITE_ACTIVITIES = [
    "showcases local artists",
    "connects volunteers with nonprofits",
    "shares educational resources",
    "promotes community events",
    "shares healthy recipes",
    "connects pet owners with services",
    "promotes local small businesses",
    "shares gardening tips",
    "connects book lovers",
    "promotes environmental awareness",
    "shares fitness routines",
    "connects language learners",
    "promotes cultural heritage",
    "shares travel experiences",
    "connects hobbyists with communities",
    "promotes mental health resources",
    "shares DIY project ideas",
    "connects mentors with students",
    "promotes sustainable living",
    "shares photography portfolios",
    "connects musicians for collaboration",
    "promotes local farmers markets",
    "shares craft tutorials",
    "connects parents with resources",
    "promotes wildlife conservation",
    "shares coding tutorials",
    "connects seniors with services",
    "promotes literacy programs",
    "shares home improvement tips",
    "connects caregivers with support",
]

# Topics (for {topic} placeholder)
TOPICS = [
    "the benefits of regular exercise",
    "how to maintain work-life balance",
    "the importance of sleep hygiene",
    "sustainable living practices",
    "healthy eating habits",
    "the benefits of reading",
    "time management strategies",
    "the importance of mental health",
    "building strong relationships",
    "the value of lifelong learning",
    "environmental conservation",
    "the benefits of meditation",
    "financial literacy basics",
    "the importance of community",
    "healthy cooking techniques",
    "the benefits of nature walks",
    "effective communication skills",
    "the importance of hobbies",
    "stress management techniques",
    "the value of volunteering",
    "digital wellness practices",
    "the benefits of journaling",
    "home organization tips",
    "the importance of gratitude",
    "creative problem solving",
    "the benefits of team sports",
    "personal development strategies",
    "the importance of hydration",
    "building self-confidence",
    "the value of patience",
]

# Recipes (for {recipe} placeholder)
RECIPES = [
    "homemade vegetable soup",
    "classic chocolate chip cookies",
    "fresh garden salad dressing",
    "homemade tomato sauce",
    "whole grain bread",
    "healthy smoothie bowls",
    "homemade granola",
    "fresh fruit jam",
    "vegetarian chili",
    "homemade yogurt",
    "banana bread",
    "fresh pasta sauce",
    "homemade hummus",
    "energy balls",
    "overnight oats",
    "homemade pesto",
    "fresh guacamole",
    "vegetable stir fry",
    "homemade protein bars",
    "fresh salsa",
]

# List items (for {list_item} placeholder)
LIST_ITEMS = [
    "indoor plants that purify air",
    "exercises you can do at home",
    "healthy snacks for work",
    "books for personal development",
    "ways to reduce stress",
    "tips for better sleep",
    "vegetables easy to grow at home",
    "hobbies to try this year",
    "ways to save money",
    "eco-friendly household products",
    "morning routine habits",
    "apps for productivity",
    "ways to stay motivated",
    "healthy breakfast ideas",
    "tips for effective studying",
    "ways to practice gratitude",
    "exercises for better posture",
    "books for learning new skills",
    "ways to reduce screen time",
    "tips for staying organized",
]


def generate_instructions(num_samples: int, seed: int):
    """Generate benign instructions using AdvBench-style templates."""
    random.seed(seed)

    instructions = []

    # Keep generating until we have enough unique instructions
    seen = set()
    attempts = 0
    max_attempts = num_samples * 10

    while len(instructions) < num_samples and attempts < max_attempts:
        attempts += 1

        # Pick a random template
        template = random.choice(TEMPLATES)

        # Fill in the template based on placeholders
        if "{activity}" in template:
            activity = random.choice(ACTIVITIES)
            instruction = template.format(activity=activity)
        elif "{tech_activity}" in template:
            tech_activity = random.choice(TECH_ACTIVITIES)
            instruction = template.format(tech_activity=tech_activity)
        elif "{app_activity}" in template:
            app_activity = random.choice(APP_ACTIVITIES)
            instruction = template.format(app_activity=app_activity)
        elif "{website_activity}" in template:
            website_activity = random.choice(WEBSITE_ACTIVITIES)
            instruction = template.format(website_activity=website_activity)
        elif "{topic}" in template:
            topic = random.choice(TOPICS)
            instruction = template.format(topic=topic)
        elif "{recipe}" in template:
            recipe = random.choice(RECIPES)
            instruction = template.format(recipe=recipe)
        elif "{list_item}" in template:
            list_item = random.choice(LIST_ITEMS)
            instruction = template.format(list_item=list_item)
        else:
            continue

        # Avoid duplicates
        if instruction not in seen:
            seen.add(instruction)
            instructions.append({
                "instruction": instruction,
                "output": f"Sure, here is a response to your request about {instruction.lower().split('how to ')[-1].split('that ')[-1].split('about ')[-1][:50]}..."
            })

    print(f"Generated {len(instructions)} unique instructions")
    return instructions


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate benign instruction audio dataset"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        choices=["both", "usa_only", "accents"],
        default="both",
        help="Which version to generate (default: both)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for reproducibility (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--max_concurrent", "-c",
        type=int,
        default=MAX_CONCURRENT,
        help=f"Max concurrent TTS requests (default: {MAX_CONCURRENT})"
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Show what would be done without generating audio"
    )
    return parser.parse_args()


async def generate_audio(text: str, voice: str, output_path: Path, semaphore: asyncio.Semaphore) -> bool:
    """Generate audio using Edge-TTS with concurrency limit."""
    async with semaphore:
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
            return True
        except Exception as e:
            print(f"\nError generating {output_path.name}: {e}")
            return False


def save_dataset(output_data: list, af_dir: Path, kimi_dir: Path, dataset_name: str, regions: list = None):
    """Save dataset in both Audio-Flamingo (JSON) and Kimi-Audio (JSONL) formats."""

    # Create directories
    af_dir.mkdir(parents=True, exist_ok=True)
    kimi_dir.mkdir(parents=True, exist_ok=True)

    # Save full Audio-Flamingo JSON
    af_json_path = af_dir / f"{dataset_name}_full.json"
    with open(af_json_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"  Audio-Flamingo: {af_json_path}")

    # Save full Kimi-Audio JSONL
    kimi_jsonl_path = kimi_dir / f"{dataset_name}_full.jsonl"
    with open(kimi_jsonl_path, 'w') as f:
        for entry in output_data:
            f.write(json.dumps(entry) + '\n')
    print(f"  Kimi-Audio: {kimi_jsonl_path}")

    # If regions specified, also create per-region files
    if regions:
        for region in regions:
            region_data = [d for d in output_data if d.get("region") == region]
            if region_data:
                # Audio-Flamingo per-region JSON
                region_json = af_dir / f"{dataset_name}_{region}.json"
                with open(region_json, 'w') as f:
                    json.dump(region_data, f, indent=2)

                # Kimi-Audio per-region JSONL
                region_jsonl = kimi_dir / f"{dataset_name}_{region}.jsonl"
                with open(region_jsonl, 'w') as f:
                    for entry in region_data:
                        f.write(json.dumps(entry) + '\n')


async def generate_usa_only_version(samples: list, args):
    """Generate American English only version (~6000 samples)."""
    print("\n" + "=" * 60)
    print("Generating USA-only version")
    print("=" * 60)

    SHARED_AUDIO_DIR_USA.mkdir(parents=True, exist_ok=True)

    voice = ACCENT_VOICES["usa"]

    # Build task list
    tasks_info = []
    for idx, sample in enumerate(samples):
        new_id = f"benign_inst_usa_{idx:05d}"
        audio_path = SHARED_AUDIO_DIR_USA / f"{new_id}.wav"
        tasks_info.append((idx, sample, audio_path, new_id))

    total_audios = len(tasks_info)
    existing = sum(1 for _, _, path, _ in tasks_info if path.exists())
    to_generate = total_audios - existing

    print(f"Total: {total_audios} audio files")
    print(f"Already exist: {existing}")
    print(f"To generate: {to_generate}")

    if args.dry_run:
        print("\n[DRY RUN] Sample instructions:")
        for i, (idx, sample, _, _) in enumerate(tasks_info[:10]):
            print(f"  {i+1}. {sample['instruction']}")
        return 0

    # Generate missing audio
    semaphore = asyncio.Semaphore(args.max_concurrent)

    if to_generate > 0:
        print(f"\nGenerating {to_generate} audio files...")

        async def generate_task(info):
            idx, sample, audio_path, new_id = info
            if not audio_path.exists():
                return await generate_audio(sample["instruction"], voice, audio_path, semaphore)
            return True

        tasks = [generate_task(info) for info in tasks_info if not info[2].exists()]

        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Generating audio"):
            result = await coro
            results.append(result)

        success_count = sum(results)
        print(f"\nGenerated {success_count}/{to_generate} files successfully")

    # Build output data
    print("\nBuilding dataset...")
    output_data = []
    for idx, sample, audio_path, new_id in tasks_info:
        if audio_path.exists():
            entry = {
                "id": new_id,
                "audio": str(audio_path),
                "instruction": sample["instruction"],
                "output": sample["output"],
                "region": "usa",
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<audio>\n{sample['instruction']}"
                    },
                    {
                        "from": "gpt",
                        "value": sample["output"]
                    }
                ]
            }
            output_data.append(entry)

    # Save datasets
    print(f"\nSaving {len(output_data)} samples...")
    save_dataset(output_data, AF_DATA_DIR_USA, KIMI_DATA_DIR_USA, "benign_instructions_usa")

    print(f"\nUSA-only version complete: {len(output_data)} samples")
    return len(output_data)


async def generate_accents_version(samples: list, args):
    """Generate multi-accent version (550 per accent × 11 = 6050 total)."""
    print("\n" + "=" * 60)
    print("Generating multi-accent version")
    print("=" * 60)

    SHARED_AUDIO_DIR_ACCENTS.mkdir(parents=True, exist_ok=True)

    # Build task list - each sample gets all 11 accents
    tasks_info = []
    for sample_idx, sample in enumerate(samples):
        for region in REGIONS:
            voice = ACCENT_VOICES[region]
            new_id = f"benign_inst_accent_{sample_idx:05d}_{region}"
            audio_path = SHARED_AUDIO_DIR_ACCENTS / f"{new_id}.wav"
            tasks_info.append((sample_idx, sample, region, voice, audio_path, new_id))

    total_audios = len(tasks_info)
    existing = sum(1 for _, _, _, _, path, _ in tasks_info if path.exists())
    to_generate = total_audios - existing

    print(f"Total: {total_audios} audio files ({len(samples)} questions × {len(REGIONS)} accents)")
    print(f"Already exist: {existing}")
    print(f"To generate: {to_generate}")

    if args.dry_run:
        print("\n[DRY RUN] Sample instructions:")
        for i, sample in enumerate(samples[:10]):
            print(f"  {i+1}. {sample['instruction']}")
        print(f"\nAccents: {', '.join(REGIONS)}")
        return 0

    # Generate missing audio
    semaphore = asyncio.Semaphore(args.max_concurrent)

    if to_generate > 0:
        print(f"\nGenerating {to_generate} audio files...")

        async def generate_task(info):
            sample_idx, sample, region, voice, audio_path, new_id = info
            if not audio_path.exists():
                return await generate_audio(sample["instruction"], voice, audio_path, semaphore)
            return True

        tasks = [generate_task(info) for info in tasks_info if not info[4].exists()]

        results = []
        for coro in tqdm.as_completed(tasks, total=len(tasks), desc="Generating audio"):
            result = await coro
            results.append(result)

        success_count = sum(results)
        print(f"\nGenerated {success_count}/{to_generate} files successfully")

    # Build output data
    print("\nBuilding dataset...")
    output_data = []
    for sample_idx, sample, region, voice, audio_path, new_id in tasks_info:
        if audio_path.exists():
            entry = {
                "id": new_id,
                "audio": str(audio_path),
                "instruction": sample["instruction"],
                "output": sample["output"],
                "region": region,
                "sample_idx": sample_idx,
                "conversations": [
                    {
                        "from": "human",
                        "value": f"<audio>\n{sample['instruction']}"
                    },
                    {
                        "from": "gpt",
                        "value": sample["output"]
                    }
                ]
            }
            output_data.append(entry)

    # Save datasets
    print(f"\nSaving {len(output_data)} samples...")
    save_dataset(output_data, AF_DATA_DIR_ACCENTS, KIMI_DATA_DIR_ACCENTS,
                 "benign_instructions_accents", REGIONS)

    print(f"\nMulti-accent version complete: {len(output_data)} samples")
    return len(output_data)


async def main():
    args = parse_args()

    print("=" * 60)
    print("Benign Instructions Audio Dataset Generator")
    print("(AdvBench-style wording with benign content)")
    print("=" * 60)
    print(f"Version: {args.version}")
    print(f"Seed: {args.seed}")
    print(f"Max concurrent: {args.max_concurrent}")

    # Determine how many unique samples we need
    if args.version == "both":
        num_needed = NUM_SAMPLES_USA  # 6000 for USA, first 550 reused for accents
    elif args.version == "usa_only":
        num_needed = NUM_SAMPLES_USA
    else:  # accents only
        num_needed = NUM_SAMPLES_PER_ACCENT

    # Generate instructions
    samples = generate_instructions(num_needed, args.seed)

    if len(samples) < num_needed:
        print(f"Warning: Only generated {len(samples)} samples, wanted {num_needed}")

    # Generate requested versions
    if args.version in ["both", "usa_only"]:
        usa_samples = samples[:NUM_SAMPLES_USA]
        await generate_usa_only_version(usa_samples, args)

    if args.version in ["both", "accents"]:
        # For accents, use first 550 samples (each gets 11 accents)
        accent_samples = samples[:NUM_SAMPLES_PER_ACCENT]
        await generate_accents_version(accent_samples, args)

    print("\n" + "=" * 60)
    print("Generation Complete!")
    print("=" * 60)

    if args.version in ["both", "usa_only"]:
        print(f"\nUSA-only dataset:")
        print(f"  Audio: {SHARED_AUDIO_DIR_USA}")
        print(f"  AF JSON: {AF_DATA_DIR_USA}")
        print(f"  Kimi JSONL: {KIMI_DATA_DIR_USA}")

    if args.version in ["both", "accents"]:
        print(f"\nMulti-accent dataset:")
        print(f"  Audio: {SHARED_AUDIO_DIR_ACCENTS}")
        print(f"  AF JSON: {AF_DATA_DIR_ACCENTS}")
        print(f"  Kimi JSONL: {KIMI_DATA_DIR_ACCENTS}")

    print(f"\nTo use with the pipeline:")
    print(f"  ./run_full_pipeline.sh --benign_dataset benign_instructions_usa --percentage 50")
    print(f"  ./run_full_pipeline.sh --benign_dataset benign_instructions_accents --percentage 50")


if __name__ == "__main__":
    asyncio.run(main())
