from flask import Flask, render_template, request, redirect, url_for, flash
import os
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
import requests
from bs4 import BeautifulSoup
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired
import json
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
import re
from html import escape
from urllib.parse import quote_plus
from datetime import datetime, timedelta

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "best_model.pth")  # FIX 1: was MODEL_PATH (undefined), now model_path
class_indices_path = os.path.join(working_dir, "class_indices.json")
upload_folder = os.path.join(working_dir, "static", "uploads")

# Load the pre-trained model when available
model = None
MODEL_LOAD_ERROR = None

label_map = {
    'Potato___Early_blight': 0,
    'Potato___Late_blight': 1,
    'Potato___healthy': 2,
    'Tomato___Bacterial_spot': 3,
    'Tomato___Early_blight': 4,
    'Tomato___Late_blight': 5,
    'Tomato___Leaf_Mold': 6,
    'Tomato___Septoria_leaf_spot': 7,
    'Tomato___Spider_mites Two-spotted_spider_mite': 8,
    'Tomato___Target_Spot': 9,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 10,
    'Tomato___Tomato_mosaic_virus': 11,
    'Tomato___healthy': 12
}

reverse_label_map = {int(v): k for k, v in label_map.items()}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def format_prediction_details(predicted_class):
    plant_name, _, disease_name = predicted_class.partition("___")
    disease_name = disease_name.replace("_", " ")
    return plant_name, disease_name


def build_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, len(label_map))
    state_dict = torch.load(model_path, map_location=DEVICE)  # FIX 1: MODEL_PATH -> model_path
    m.load_state_dict(state_dict)
    m.to(DEVICE)
    m.eval()
    return m


try:
    model = build_model()
    MODEL_LOAD_ERROR = None
except Exception as exc:
    model = None
    MODEL_LOAD_ERROR = str(exc)


def predict_disease(filepath):
    if model is None:
        raise RuntimeError(f"Model unavailable: {MODEL_LOAD_ERROR}")

    image = Image.open(filepath).convert("RGB")
    tensor = val_transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        class_id = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][class_id].item() * 100

    predicted_class = reverse_label_map[class_id]
    return predicted_class, confidence


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}

PRICE_REPORT_URL = "https://api.agmarknet.gov.in/v1/prices-and-arrivals/market-report/daily"
AGMARKNET_FILTERS_URL = "https://api.agmarknet.gov.in/v1/daily-price-arrival/filters"
AGMARKNET_FILTERS_CACHE = None

TRUSTED_GUIDE_PRIORITY = {
    "general": ["ICAR", "TNAU Agritech", "NHB", "PAU", "ANGRAU", "PJTSAU", "FAO"],
    "market": ["Agmarknet", "DMI", "eNAM", "agriwelfare.gov.in"],
    "cultivation": ["ICAR", "TNAU Agritech", "PAU", "ANGRAU", "PJTSAU", "NHB"],
    "protection": ["ICAR", "TNAU Agritech", "PAU", "ANGRAU", "PJTSAU"],
    "postharvest": ["NHB", "ICAR", "APEDA", "TNAU Agritech", "FAO"],
}

PRIORITY_MARKETS = [
    {"state_id": 16, "market_id": 112, "state_name": "Karnataka", "market_name": "Kolar APMC"},
    {"state_id": 32, "market_id": 1868, "state_name": "Telangana", "market_name": "Bowenpally APMC"},
    {"state_id": 20, "market_id": 2140, "state_name": "Maharashtra", "market_name": "Nashik(Devlali) APMC"},
    {"state_id": 25, "market_id": 3445, "state_name": "Delhi", "market_name": "Azadpur APMC"},
    {"state_id": 16, "market_id": 743, "state_name": "Karnataka", "market_name": "Chintamani APMC"},
    {"state_id": 2, "market_id": 938, "state_name": "Andhra Pradesh", "market_name": "Madanapalli APMC"},
    {"state_id": 2, "market_id": 1987, "state_name": "Andhra Pradesh", "market_name": "Mulakalacheruvu APMC"},
    {"state_id": 16, "market_id": 108, "state_name": "Karnataka", "market_name": "Hubli (Amaragol) APMC"},
]

CROP_HINTS = {
    "tomato": {
        "scientific_name": "Solanum lycopersicum",
        "common_names": ["Love apple"],
        "plant_type": "A warm-season fruiting vine that is usually grown as an annual plant.",
        "overview": "Tomato is a widely grown fruit-vegetable crop valued for its juicy berries, kitchen use, and strong market demand.",
        "climate": "Grows best in warm, sunny weather with temperatures around 21 to 30°C and no frost.",
        "soil": "Prefers fertile, well-drained sandy loam or loam soil rich in organic matter, usually around pH 6.0 to 7.0.",
        "planting_season": "Usually planted in mild to warm seasons when frost is not expected and the soil has warmed up.",
        "watering_fertilizer": "Keep the soil evenly moist. Water deeply about 2 to 3 times a week after establishment and feed every 2 to 3 weeks with compost or a balanced fertilizer.",
        "pest_summary": "Common problems include aphids, whiteflies, hornworms, blight, and wilt. Good spacing, field hygiene, neem-based sprays, and timely fungicide use help control them.",
        "harvest": "Usually ready in about 70 to 100 days after transplanting. Start picking when fruits reach full size and show the right colour.",
        "post_harvest": "Sort fruits gently, keep them in shade, and store mature tomatoes in a cool, airy place to reduce bruising and rotting.",
        "yield": {"low": 20, "high": 30, "unit": "tonnes per acre", "text": "Typical open-field yield: 20 to 30 tonnes per acre under good management."},
        "soil_moisture": "Keep the soil evenly moist, but not soggy or waterlogged.",
        "guidance_source": "standard crop guidance",
    },
    "rice": {
        "scientific_name": "Oryza sativa",
        "common_names": ["Paddy"],
        "plant_type": "An annual cereal grass crop.",
        "overview": "Rice is a staple cereal crop grown mainly in flooded or well-irrigated fields for grain production.",
        "climate": "Grows best in warm, humid weather with full sun, good water supply, and temperatures roughly between 20 and 35°C.",
        "soil": "Prefers fertile clayey or loamy soil that can hold moisture well and support standing water during growth.",
        "planting_season": "Usually planted in the monsoon or kharif season in many regions, though irrigated crops may also be grown in other seasons.",
        "watering_fertilizer": "Needs regular moisture or shallow standing water during active growth. Add farmyard manure before planting and split nitrogen fertilizer during early growth and tillering.",
        "pest_summary": "Common problems include stem borer, leaf folder, blast, and bacterial blight. Clean fields, resistant varieties, and approved sprays help control them.",
        "harvest": "Usually ready in about 100 to 150 days, when most panicles turn golden and the grains become hard.",
        "post_harvest": "Dry the harvested grain well, clean it properly, and store it in a dry, pest-free place with low moisture.",
        "yield": {"low": 20, "high": 30, "unit": "quintals per acre", "text": "Typical yield: about 20 to 30 quintals per acre, depending on variety and field conditions."},
        "soil_moisture": "Needs consistently moist to wet soil during active growth.",
        "guidance_source": "standard crop guidance",
    },
    "wheat": {
        "scientific_name": "Triticum aestivum",
        "plant_type": "A cool-season annual cereal grass.",
        "overview": "Wheat is a major cereal crop grown for its grain, which is used for flour, bread, and many staple foods.",
        "climate": "Grows best in cool, dry weather during early growth and clear, dry weather during grain maturity.",
        "soil": "Prefers fertile, well-drained loam to clay-loam soil with moderate moisture and good nutrient supply.",
        "planting_season": "Usually planted in the cool rabi season after the monsoon, when temperatures are mild and the field is well prepared.",
        "watering_fertilizer": "Needs light to moderate irrigation, especially at crown root initiation, tillering, flowering, and grain filling. Apply compost before sowing and balanced fertilizer in split doses.",
        "pest_summary": "Common problems include rust, aphids, termites, and smut. Use clean seed, crop rotation, balanced fertilizer, and timely plant protection when needed.",
        "harvest": "Usually ready in about 110 to 140 days, when the crop turns golden and the grains become hard and dry.",
        "post_harvest": "Harvest in dry weather, dry the grain well, and store it in clean, dry containers away from moisture and storage pests.",
        "yield": {"low": 16, "high": 24, "unit": "quintals per acre", "text": "Typical yield: about 16 to 24 quintals per acre with good crop management."},
        "soil_moisture": "Prefers moderately moist soil with good drainage.",
        "guidance_source": "standard crop guidance",
    },
    "maize": {
        "scientific_name": "Zea mays",
        "common_names": ["Corn"],
        "plant_type": "A tall annual cereal plant.",
        "overview": "Maize is a fast-growing cereal crop grown for grain, fodder, and food use across many climates.",
        "climate": "Grows best in warm, sunny weather with temperatures around 18 to 32°C and moderate rainfall or irrigation.",
        "soil": "Prefers fertile, well-drained loam soil with good organic matter and moderate moisture.",
        "planting_season": "Can be planted in monsoon, winter, or spring depending on the region, but it grows best when the soil is warm and drainage is good.",
        "watering_fertilizer": "Water regularly, especially at knee-high growth, tasseling, silking, and grain filling. Apply compost before sowing and nitrogen-rich fertilizer in split doses.",
        "pest_summary": "Common problems include fall armyworm, stem borer, leaf blight, and downy mildew. Field scouting, clean cultivation, and approved control measures are important.",
        "harvest": "Usually ready in about 90 to 120 days, when cobs are filled and the grains become firm and mature.",
        "post_harvest": "Dry cobs and grain properly after harvest and store them in a dry, well-ventilated area to prevent mould and insect damage.",
        "yield": {"low": 18, "high": 30, "unit": "quintals per acre", "text": "Typical yield: around 18 to 30 quintals per acre, depending on seed, irrigation, and fertilizer use."},
        "soil_moisture": "Needs evenly moist soil, especially during tasseling and grain filling.",
        "guidance_source": "standard crop guidance",
    },
    "potato": {
        "scientific_name": "Solanum tuberosum",
        "plant_type": "A cool-season tuber crop grown as a low herbaceous plant.",
        "overview": "Potato is a tuber crop grown for its underground storage stems and is widely used as a food staple and vegetable.",
        "climate": "Grows best in cool, frost-free weather, usually around 15 to 25°C, with good sunlight and moderate moisture.",
        "soil": "Prefers loose, fertile, well-drained sandy loam or loam soil so the tubers can expand properly.",
        "planting_season": "Usually planted in the cool season when the soil is workable and temperatures are not too high.",
        "watering_fertilizer": "Keep the soil lightly moist and avoid waterlogging. Water regularly during tuber formation and feed with compost plus balanced fertilizer in split doses.",
        "pest_summary": "Common problems include late blight, early blight, aphids, and tuber rot. Use healthy seed tubers, rotate crops, and avoid excess leaf wetness.",
        "harvest": "Usually ready in about 90 to 120 days, when the tops yellow and the tuber skin becomes firm.",
        "post_harvest": "Cure the tubers in shade after harvest, remove damaged potatoes, and store them in a cool, dark, well-ventilated place.",
        "yield": {"low": 80, "high": 120, "unit": "quintals per acre", "text": "Typical yield: about 80 to 120 quintals per acre under good cultivation."},
        "soil_moisture": "Keep the soil lightly moist and loose; avoid standing water.",
        "guidance_source": "standard crop guidance",
    },
    "onion": {
        "scientific_name": "Allium cepa",
        "plant_type": "A bulb-forming herbaceous crop.",
        "overview": "Onion is a bulb crop grown for its layered underground bulb and green leaves used in cooking and trade.",
        "climate": "Grows best in mild weather, with cooler conditions during early growth and drier weather during bulb maturity and harvest.",
        "soil": "Prefers fertile, loose, well-drained sandy loam to silt-loam soil and does not perform well in heavy, compact soil.",
        "planting_season": "Usually planted in cool to mild weather so the crop can establish well before bulb formation begins.",
        "watering_fertilizer": "Water lightly but regularly during establishment and bulb growth, then reduce watering before harvest. Apply compost before planting and split fertilizer doses during early growth.",
        "pest_summary": "Common problems include thrips, purple blotch, damping off, and neck rot. Good drainage, field hygiene, and timely sprays help manage them.",
        "harvest": "Usually ready in about 90 to 150 days, when the tops bend over and the bulb necks soften.",
        "post_harvest": "Dry and cure the bulbs well after harvest, trim the tops, and store them in a cool, dry, airy place.",
        "yield": {"low": 80, "high": 120, "unit": "quintals per acre", "text": "Typical yield: around 80 to 120 quintals per acre, depending on variety and management."},
        "soil_moisture": "Prefers evenly moist but well-drained soil.",
        "guidance_source": "standard crop guidance",
    },
    "banana": {
        "scientific_name": "Musa spp.",
        "common_names": ["Plantain"],
        "plant_type": "A large tropical herb with a pseudostem; often mistaken for a tree.",
        "overview": "Banana is a tropical fruit crop grown for its long clustered fruits and year-round market demand.",
        "climate": "Grows best in warm, humid tropical weather with good sunlight, regular moisture, and temperatures roughly between 20 and 35°C.",
        "soil": "Prefers deep, fertile, well-drained loam soil with plenty of organic matter and steady moisture.",
        "planting_season": "Usually planted at the start of the rainy season or under irrigation when enough moisture is available for quick establishment.",
        "watering_fertilizer": "Needs frequent watering because the plant is large and fast growing. Keep the root zone moist and feed regularly with organic manure and balanced fertilizer through the growing period.",
        "pest_summary": "Common problems include sigatoka leaf spot, pseudostem borer, aphids, and Panama wilt. Clean planting material, good drainage, and timely protection are important.",
        "harvest": "Usually ready in about 9 to 12 months after planting, when the fingers are full and the bunch is mature.",
        "post_harvest": "Handle bunches carefully to avoid bruising, keep fruits shaded, and store them in a cool, clean place during ripening and transport.",
        "yield": {"low": 12, "high": 18, "unit": "tonnes per acre", "text": "Typical yield: about 12 to 18 tonnes per acre under good tropical growing conditions."},
        "soil_moisture": "Needs moist, rich soil with good drainage throughout the year.",
        "guidance_source": "standard crop guidance",
    },
}


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def clean_display_text(text, max_sentences=2):
    cleaned = clean_text(text)
    cleaned = re.sub(r"\[[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"/[^/]{2,40}/", "", cleaned)
    cleaned = re.sub(r"\(\s*(?:US|UK)[^)]*\)", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\(\s*[A-Z][a-z]+(?:\s+[a-z]+)+\s*\)", "", cleaned)
    cleaned = re.sub(r"\s+([,.;:])", r"\1", cleaned)
    cleaned = re.sub(r",{2,}", ",", cleaned).strip(" ,;")

    sentences = [
        clean_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", cleaned)
        if clean_text(sentence)
    ]
    return " ".join(sentences[:max_sentences]) if sentences else cleaned


def first_non_empty(*values):
    for value in values:
        cleaned_value = clean_text(value)
        if cleaned_value:
            return cleaned_value
    return ""


def build_crop_slugs(crop_name):
    base_slug = re.sub(r"[^a-z0-9]+", "-", crop_name.lower()).strip("-")
    candidates = [base_slug]

    if base_slug and not base_slug.endswith("s"):
        candidates.append(f"{base_slug}s")
    if base_slug.endswith("o"):
        candidates.append(f"{base_slug}es")

    return list(dict.fromkeys([slug for slug in candidates if slug]))


def build_crop_terms(crop_name):
    hint = get_crop_hint(crop_name)
    crop_name = clean_text(crop_name).lower()
    terms = {crop_name, crop_name.rstrip("s")}

    for name in hint.get("common_names", []):
        terms.add(clean_text(name).lower())
    scientific_name = clean_text(hint.get("scientific_name", "")).lower()
    if scientific_name:
        terms.add(scientific_name)
        terms.add(scientific_name.split()[0])

    alias_map = {
        "banana": ["bananas", "plantain"],
        "maize": ["corn"],
        "rice": ["paddy"],
        "potato": ["potatoes"],
        "onion": ["onions"],
        "tomato": ["tomatoes"],
    }
    for alias in alias_map.get(crop_name, []):
        terms.add(alias)

    return {term for term in terms if term}


def is_relevant_crop_text(text, crop_name):
    lower_text = clean_text(text).lower()
    if not lower_text:
        return False

    unrelated_terms = ["film", "movie", "album", "song", "actor", "actress", "director", "television", "comedian"]
    crop_terms = build_crop_terms(crop_name)
    plant_terms = ["plant", "crop", "fruit", "vegetable", "grain", "cereal", "tuber", "bulb", "leaf", "flower", "seed"]

    if any(term in lower_text for term in unrelated_terms) and not any(term in lower_text for term in plant_terms):
        return False

    return any(term in lower_text for term in crop_terms if len(term) > 2)


def get_preferred_text(hint_value, source_text, keywords, fallback, max_sentences=2):
    preferred_text = clean_text(source_text)
    if preferred_text:
        return direct_answer(preferred_text, keywords, fallback, max_sentences=max_sentences)
    if clean_text(hint_value):
        return clean_display_text(hint_value, max_sentences=max_sentences)
    return clean_display_text(fallback, max_sentences=max_sentences)


def pick_attribute_source(hint_value, source_text, default_source, hint_source):
    if clean_text(source_text):
        return default_source
    if clean_text(hint_value):
        return hint_source
    return default_source or hint_source or "CropCare trusted fallback guidance"


def fetch_soup(url):
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=12)
    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def get_section_text(soup, heading_keywords, wikipedia=False):
    heading_elements = soup.select("span.mw-headline") if wikipedia else soup.select("h2, h3")

    for heading in heading_elements:
        title = clean_text(heading.get_text(" ", strip=True)).lower()
        if any(keyword in title for keyword in heading_keywords):
            section_anchor = heading.parent if wikipedia else heading
            collected_text = []
            node = section_anchor.find_next_sibling()

            while node and node.name not in ["h1", "h2", "h3"]:
                text = clean_text(node.get_text(" ", strip=True))
                if text and "sign up" not in text.lower() and "read next" not in text.lower():
                    collected_text.append(text)
                node = node.find_next_sibling()

                if len(" ".join(collected_text)) > 2200:
                    break

            if collected_text:
                return " ".join(collected_text)

    return ""


def format_number(value):
    try:
        number = float(value)
        return f"{number:,.0f}" if number.is_integer() else f"{number:,.2f}"
    except (TypeError, ValueError):
        return str(value)


def is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def scrape_britannica_overview(crop_name):
    for slug in build_crop_slugs(crop_name):
        url = f"https://www.britannica.com/plant/{slug}"
        try:
            soup = fetch_soup(url)
            paragraphs = [
                clean_text(paragraph.get_text(" ", strip=True))
                for paragraph in soup.select("article p")
                if clean_text(paragraph.get_text(" ", strip=True))
                and "our editors will review" not in clean_text(paragraph.get_text(" ", strip=True)).lower()
            ]
            title_text = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
            heading_text = clean_text(soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else "")
            preview_text = " ".join([title_text, heading_text, " ".join(paragraphs[:2])])

            if paragraphs and is_relevant_crop_text(preview_text, crop_name):
                return {
                    "overview": " ".join(paragraphs[:2]),
                    "source": "Britannica",
                }
        except requests.RequestException:
            continue

    return {}


def scrape_wikipedia_sections(crop_name):
    hint = get_crop_hint(crop_name)
    title_candidates = [
        crop_name.replace(" ", "_"),
        crop_name.title().replace(" ", "_"),
        clean_text(hint.get("scientific_name", "")).replace(" ", "_"),
        *[clean_text(name).replace(" ", "_") for name in hint.get("common_names", [])],
    ]

    for title in dict.fromkeys([candidate for candidate in title_candidates if clean_text(candidate)]):
        url = f"https://en.wikipedia.org/wiki/{quote_plus(title)}"
        try:
            soup = fetch_soup(url)
            paragraphs = [
                clean_text(paragraph.get_text(" ", strip=True))
                for paragraph in soup.select("div.mw-parser-output > p")
                if clean_text(paragraph.get_text(" ", strip=True))
            ]
            title_text = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
            heading_text = clean_text(soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else "")
            preview_text = " ".join([title_text, heading_text, " ".join(paragraphs[:2])])

            if paragraphs and is_relevant_crop_text(preview_text, crop_name):
                return {
                    "intro": " ".join(paragraphs[:2]),
                    "cultivation": get_section_text(soup, ["cultivation", "growing"], wikipedia=True),
                    "harvest": get_section_text(soup, ["picking and ripening", "harvest"], wikipedia=True),
                    "production": get_section_text(soup, ["production"], wikipedia=True),
                    "pests": get_section_text(soup, ["pests and diseases", "diseases", "pests"], wikipedia=True),
                    "storage": get_section_text(soup, ["storage"], wikipedia=True),
                    "source": "Wikipedia",
                }
        except requests.RequestException:
            continue

    return {}


def scrape_almanac_sections(crop_name):
    for slug in build_crop_slugs(crop_name):
        url = f"https://www.almanac.com/plant/{slug}"
        try:
            soup = fetch_soup(url)
            page_title = clean_text(soup.title.get_text(" ", strip=True) if soup.title else "")
            preview_text = " ".join([
                page_title,
                clean_text(soup.select_one("h1").get_text(" ", strip=True) if soup.select_one("h1") else ""),
            ])
            if not page_title or "page not found" in page_title.lower() or not is_relevant_crop_text(preview_text, crop_name):
                continue

            return {
                "overview": get_section_text(soup, ["about"]),
                "when_to_plant": get_section_text(soup, ["when to plant"]),
                "planting": get_section_text(soup, ["planting"]),
                "how_to_plant": get_section_text(soup, ["how to plant"]),
                "growing": get_section_text(soup, ["growing"]),
                "watering": get_section_text(soup, ["watering"]),
                "feeding": get_section_text(soup, ["feeding", "fertilizing"]),
                "harvest": get_section_text(soup, ["harvesting"]),
                "pests": get_section_text(soup, ["pests/diseases", "pests", "diseases"]),
                "storage": get_section_text(soup, ["storage"]),
                "source": "The Old Farmer's Almanac",
            }
        except requests.RequestException:
            continue

    return {}


def extract_relevant_detail(source_text, keywords, fallback):
    sentences = [
        clean_text(sentence)
        for sentence in re.split(r"(?<=[.!?])\s+", source_text or "")
        if clean_text(sentence)
    ]

    for sentence in sentences:
        lower_sentence = sentence.lower()
        if any(keyword in lower_sentence for keyword in keywords):
            return sentence

    if sentences:
        return sentences[0]

    return fallback


def direct_answer(source_text, keywords, fallback, max_sentences=2):
    detail = extract_relevant_detail(source_text, keywords, fallback)
    return clean_display_text(detail, max_sentences=max_sentences)


def with_source(text, source_name):
    cleaned_text = clean_display_text(text)
    if not cleaned_text:
        return "General crop guidance from trusted references is being used here."
    return f"{cleaned_text} (Source: {source_name})"


def build_source_note(source_name, guide_type="general"):
    actual_source = clean_text(source_name) or "trusted agriculture reference"
    trusted_sources = ", ".join(TRUSTED_GUIDE_PRIORITY.get(guide_type, TRUSTED_GUIDE_PRIORITY["general"]))
    return f"<p><em>Source used: {escape(actual_source)} | Trusted guide priority: {escape(trusted_sources)}</em></p>"


def build_sowing_steps(crop_name, source_text, source_name):
    crop_label = clean_text(crop_name).title()
    source_text = clean_text(source_text)
    lead_time = ""
    seed_depth = ""
    soil_temp = ""

    lead_match = re.search(r"(\d+\s*weeks?)\s+before", source_text, flags=re.IGNORECASE)
    depth_match = re.search(r"(\d+(?:/\d+)?)\s*inch\s+deep", source_text, flags=re.IGNORECASE)
    temp_match = re.search(r"soil (?:is at least|reaches?)\s*(\d+\s*°?\s*[FC])", source_text, flags=re.IGNORECASE)

    if lead_match:
        lead_time = clean_text(lead_match.group(1))
    if depth_match:
        seed_depth = clean_text(depth_match.group(1)) + " inch"
    if temp_match:
        soil_temp = clean_text(temp_match.group(1))

    steps = [
        f"Choose a sunny, well-drained spot for {crop_label} and mix compost or aged manure into the soil.",
        f"Start the seeds about {lead_time} before the last frost, or buy healthy seedlings from a nursery." if lead_time else f"Start with healthy seeds or seedlings at the right planting time for your area.",
        f"Sow the seeds about {seed_depth} deep and keep the soil lightly moist until they sprout." if seed_depth else "Sow the seeds shallowly in moist soil and cover them lightly.",
        f"Move the seedlings outdoors only when the soil is warm enough ({soil_temp}) and frost danger is over." if soil_temp else "Transplant or thin the seedlings only after the weather becomes warm and stable.",
        "Water gently after planting, keep the soil evenly moist, and add stakes or cages early if the crop needs support." if crop_name.lower() == "tomato" or any(word in source_text.lower() for word in ["stake", "cage", "support"]) else "Water gently after planting and keep the bed weed-free and evenly moist.",
    ]

    items = "".join(f"<li>{escape(clean_display_text(step, max_sentences=1))}</li>" for step in steps)
    return f"<ol>{items}</ol>" + build_source_note(source_name, "cultivation")


def build_html_list(items, ordered=False):
    tag = "ol" if ordered else "ul"
    clean_items = [escape(clean_display_text(item, max_sentences=2)) for item in items if clean_text(item)]
    return f"<{tag}>" + "".join(f"<li>{item}</li>" for item in clean_items) + f"</{tag}>" if clean_items else "<p>General crop guidance from trusted references is shown here.</p>"


def get_crop_hint(crop_name):
    aliases = {
        "corn": "maize",
        "paddy": "rice",
    }
    key = aliases.get(clean_text(crop_name).lower(), clean_text(crop_name).lower())
    return CROP_HINTS.get(key, {})


def get_season_context(preferred_season=None):
    preferred = clean_text(preferred_season) or "Current season"
    season_map = {
        "Spring": "February to April",
        "Summer": "March to June",
        "Monsoon": "June to September",
        "Autumn": "October to November",
        "Winter": "December to February",
        "Rainy": "June to September",
    }

    if preferred.lower() == "current season":
        current_month = datetime.now().month
        if current_month in [12, 1, 2]:
            return "Winter", season_map["Winter"]
        if current_month in [3, 4]:
            return "Spring", season_map["Spring"]
        if current_month in [5, 6]:
            return "Summer", season_map["Summer"]
        if current_month in [7, 8, 9]:
            return "Monsoon", season_map["Monsoon"]
        return "Autumn", season_map["Autumn"]

    return preferred, season_map.get(preferred, "months vary by local climate")


def get_live_weather_context(location):
    location = clean_text(location)
    if not location:
        return {}

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": location, "count": 1, "language": "en", "format": "json"},
            headers=BROWSER_HEADERS,
            timeout=12,
        )
        geo_response.raise_for_status()
        geo_data = geo_response.json()
        results = geo_data.get("results") or []
        if not results:
            return {}

        place = results[0]
        latitude = place.get("latitude")
        longitude = place.get("longitude")
        if latitude is None or longitude is None:
            return {}

        forecast_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,relative_humidity_2m,precipitation",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
                "timezone": "auto",
                "forecast_days": 7,
            },
            headers=BROWSER_HEADERS,
            timeout=12,
        )
        forecast_response.raise_for_status()
        forecast_data = forecast_response.json()

        city_name = clean_text(place.get("name", ""))
        state_name = clean_text(place.get("admin1", ""))
        country_name = clean_text(place.get("country", ""))
        location_name = clean_text(", ".join(
            part for part in [city_name, state_name, country_name] if clean_text(part)
        ))
        current = forecast_data.get("current", {})
        daily = forecast_data.get("daily", {})

        max_values = [value for value in daily.get("temperature_2m_max", []) if isinstance(value, (int, float))]
        rain_values = [value for value in daily.get("precipitation_sum", []) if isinstance(value, (int, float))]
        avg_max = sum(max_values) / len(max_values) if max_values else None
        weekly_rain = sum(rain_values) if rain_values else 0

        temp_now = current.get("temperature_2m")
        humidity_now = current.get("relative_humidity_2m")
        rain_now = current.get("precipitation")

        temp_text = f"{format_number(temp_now)}°C" if is_number(temp_now) else "temperature data unavailable"
        humidity_text = f"{format_number(humidity_now)}% humidity" if is_number(humidity_now) else "humidity data unavailable"
        rain_text = f"{format_number(rain_now)} mm current rain" if is_number(rain_now) else "rain data unavailable"

        if isinstance(avg_max, (int, float)) and avg_max >= 32:
            watering_adjustment = "The next few days look hot, so check the soil daily and plan deeper watering about 2 to 3 times a week."
        elif weekly_rain >= 20:
            watering_adjustment = "Rain is expected this week, so reduce extra watering and focus more on drainage and disease prevention."
        elif isinstance(avg_max, (int, float)) and avg_max <= 18:
            watering_adjustment = "The next few days look cool, so water less often and avoid keeping the soil too wet."
        else:
            watering_adjustment = "The next few days look moderate, so follow the normal season schedule and water when the topsoil starts drying."

        this_month = datetime.now().strftime("%B")
        next_month = (datetime.now().replace(day=1) + timedelta(days=32)).strftime("%B")

        return {
            "location_name": location_name,
            "city_name": city_name,
            "state_name": state_name,
            "country": country_name,
            "summary": f"Live weather for {location_name}: about {temp_text}, {humidity_text}, and {rain_text} right now.",
            "month_note": f"Weather-based planning window for {location_name}: {this_month} to {next_month} for short-term field decisions.",
            "watering_adjustment": watering_adjustment,
        }
    except Exception:
        return {}


def normalize_lookup_text(text):
    return re.sub(r"[^a-z0-9]+", " ", clean_text(text).lower()).strip()


def fetch_agmarknet_filters():
    global AGMARKNET_FILTERS_CACHE

    if AGMARKNET_FILTERS_CACHE is not None:
        return AGMARKNET_FILTERS_CACHE

    try:
        response = requests.get(
            AGMARKNET_FILTERS_URL,
            headers={**BROWSER_HEADERS, "Accept": "application/json"},
            timeout=15,
        )
        response.raise_for_status()
        AGMARKNET_FILTERS_CACHE = response.json().get("data", {})
    except Exception:
        AGMARKNET_FILTERS_CACHE = {}

    return AGMARKNET_FILTERS_CACHE


def get_location_market_candidates(location="", weather_context=None):
    weather_context = weather_context or {}
    filters = fetch_agmarknet_filters()
    location_label = first_non_empty(weather_context.get("city_name"), location)
    state_name = clean_text(weather_context.get("state_name", ""))
    country_name = clean_text(weather_context.get("country", ""))

    if country_name and "india" not in country_name.lower():
        return [], PRIORITY_MARKETS, location_label, state_name

    if not filters:
        return [], PRIORITY_MARKETS, location_label, state_name

    state_data = filters.get("state_data", [])
    market_data = filters.get("market_data", [])
    state_lookup = {
        normalize_lookup_text(item.get("state_name")): item.get("state_id")
        for item in state_data
        if clean_text(item.get("state_name"))
    }
    state_names = {
        str(item.get("state_id")): clean_text(item.get("state_name"))
        for item in state_data
        if item.get("state_id") is not None
    }
    state_id = state_lookup.get(normalize_lookup_text(state_name)) if state_name else None

    preferred_candidates = []
    fallback_candidates = []
    seen = set()
    location_terms = [term for term in {normalize_lookup_text(location), normalize_lookup_text(location_label)} if term]

    def add_candidate(target_list, state_id_value, market_id_value, state_name_value, market_name_value):
        try:
            state_id_num = int(state_id_value)
            market_id_num = int(market_id_value)
        except (TypeError, ValueError):
            return

        key = (state_id_num, market_id_num)
        if key in seen:
            return

        seen.add(key)
        target_list.append({
            "state_id": state_id_num,
            "market_id": market_id_num,
            "state_name": clean_text(state_name_value),
            "market_name": clean_text(market_name_value),
        })

    if state_id:
        exact_count = 0
        for market in market_data:
            if str(market.get("state_id")) != str(state_id):
                continue

            market_name = clean_text(market.get("mkt_name"))
            normalized_market = normalize_lookup_text(market_name)
            if any(term and term in normalized_market for term in location_terms):
                add_candidate(preferred_candidates, market.get("state_id"), market.get("id"), state_names.get(str(market.get("state_id")), state_name), market_name)
                exact_count += 1
                if exact_count >= 8:
                    break

        same_state_count = 0
        for market in market_data:
            if str(market.get("state_id")) != str(state_id):
                continue
            add_candidate(preferred_candidates, market.get("state_id"), market.get("id"), state_names.get(str(market.get("state_id")), state_name), market.get("mkt_name"))
            same_state_count += 1
            if same_state_count >= 12:
                break

        for market in PRIORITY_MARKETS:
            if str(market.get("state_id")) == str(state_id):
                add_candidate(preferred_candidates, market["state_id"], market["market_id"], market["state_name"], market["market_name"])

    for market in PRIORITY_MARKETS:
        add_candidate(fallback_candidates, market["state_id"], market["market_id"], market["state_name"], market["market_name"])

    return preferred_candidates, fallback_candidates or PRIORITY_MARKETS, location_label, state_name


def extract_scientific_name(crop_name, *texts):
    hint = get_crop_hint(crop_name)
    if hint.get("scientific_name"):
        return hint["scientific_name"]

    for text in texts:
        if not text:
            continue
        match = re.search(r"\(([A-Z][a-z]+(?:\s+[a-z]+){1,2})\)", text)
        if match:
            return clean_text(match.group(1))

    return ""


def infer_growth_habit(crop_name, source_text):
    hint = get_crop_hint(crop_name)
    if hint.get("plant_type"):
        return hint["plant_type"]

    lower_text = clean_text(source_text).lower()
    if any(word in lower_text for word in ["vine", "climber", "creeper", "crawler", "trailing"]):
        return "Usually grows as a vine, climber, or spreading plant."
    if "tree" in lower_text:
        return "Usually grows as a tree crop."
    if "shrub" in lower_text or "bush" in lower_text:
        return "Usually grows as a bushy or shrub-type plant."
    if any(word in lower_text for word in ["grass", "cereal"]):
        return "An annual cereal or grass-type crop."
    if any(word in lower_text for word in ["herb", "annual", "perennial"]):
        return "A cultivated herbaceous plant grown for food or produce."
    return "A cultivated crop plant grown for food, fruit, or produce."


def parse_yield_info(crop_name):
    hint = get_crop_hint(crop_name)
    return hint.get("yield")


def estimate_gross_return(price_value, price_unit, yield_info):
    if not yield_info or not is_number(price_value):
        return ""

    low = yield_info.get("low")
    high = yield_info.get("high")
    unit = clean_text(yield_info.get("unit", "")).lower()
    price_unit = clean_text(price_unit).lower()

    if low is None or high is None:
        return ""

    if "quintal" in price_unit and "tonne" in unit:
        low_value = low * 10 * float(price_value)
        high_value = high * 10 * float(price_value)
    elif "quintal" in price_unit and "quintal" in unit:
        low_value = low * float(price_value)
        high_value = high * float(price_value)
    else:
        return ""

    if round(low_value) == round(high_value):
        return f"Possible gross sale value: about ₹{format_number(low_value)} per acre before expenses."
    return f"Possible gross sale value: about ₹{format_number(low_value)} to ₹{format_number(high_value)} per acre before expenses."


def build_market_value_html(market_details, yield_info):
    items = [market_details.get("price_text", "Use the latest official mandi listing as the main price reference for this crop.")]
    if market_details.get("location_note"):
        items.append(market_details["location_note"])
    gross_return = estimate_gross_return(market_details.get("price_value"), market_details.get("price_unit"), yield_info)
    if gross_return:
        items.append(gross_return)
    items.append("Actual net profit will depend on labour, fertilizer, irrigation, transport, and local selling costs.")
    html = build_html_list(items)
    return html + build_source_note("Agmarknet official market data", "market")


def build_name_html(crop_name, overview_text, wikipedia_data, source_name):
    crop_label = clean_text(crop_name).title()
    hint = get_crop_hint(crop_name)
    scientific_name = extract_scientific_name(crop_name, wikipedia_data.get("intro", ""), overview_text)
    common_names = hint.get("common_names", [])

    items = [crop_label]
    if scientific_name:
        items.append(f"Scientific name: {scientific_name}")
    if common_names:
        items.append(f"Other common names: {', '.join(common_names)}")

    html = build_html_list(items)
    return html + build_source_note(source_name, "general")


def build_description_html(crop_name, overview_text, wikipedia_data, source_name):
    growth_habit = infer_growth_habit(crop_name, f"{wikipedia_data.get('intro', '')} {wikipedia_data.get('cultivation', '')}")
    physical_traits = direct_answer(
        f"{wikipedia_data.get('intro', '')} {wikipedia_data.get('cultivation', '')}",
        ["leaf", "flower", "fruit", "vine", "berry", "height", "bulb", "stem", "seed"],
        overview_text,
    )

    html = build_html_list([
        overview_text,
        f"Plant type: {growth_habit}",
        f"Physical traits: {physical_traits}",
    ])
    return html + build_source_note(source_name, "cultivation")


def build_climate_html(climate_text, preferred_season, source_name, weather_context=None):
    weather_context = weather_context or {}
    season_name, month_range = get_season_context(preferred_season)
    html = build_html_list([
        f"Best climate: {climate_text}",
        weather_context.get("summary") or f"Season window to watch: {season_name} ({month_range}).",
        weather_context.get("month_note") or f"Best months to focus on for this season: {month_range}.",
        "Healthy growth is easier when the crop gets good sunlight, fresh air, and protection from extreme weather.",
    ])
    return html + build_source_note(source_name, "cultivation")


def build_soil_html(crop_name, soil_text, source_name):
    moisture_note = get_crop_hint(crop_name).get("soil_moisture") or "Keep the soil moderately moist with good drainage; do not let it stay waterlogged."
    html = build_html_list([
        f"Best soil: {soil_text}",
        f"Soil moisture / humidity: {moisture_note}",
        "Loose, fertile, and well-drained soil usually helps roots grow strongly and reduces disease risk.",
    ])
    return html + build_source_note(source_name, "cultivation")


def build_planting_html(planting_text, preferred_season, source_name):
    season_name, month_range = get_season_context(preferred_season)
    html = build_html_list([
        f"Best planting season: {planting_text}",
        f"Suggested growing window for your selected plan: {season_name} ({month_range}).",
        "Start planting only when the weather is suitable and the soil is ready for healthy root growth.",
    ])
    return html + build_source_note(source_name, "cultivation")


def build_watering_fertilizer_html(watering_text, preferred_season, source_name, weather_context=None):
    weather_context = weather_context or {}
    season_name, month_range = get_season_context(preferred_season)
    season_lower = season_name.lower()

    watering_schedule = {
        "summer": "Water lightly every day for the first few days after planting. Once established, deep water about 2 to 3 times a week and check the soil daily during hot weather.",
        "monsoon": "Water only when the topsoil starts drying. During rainy periods, focus more on drainage and avoid standing water around the roots.",
        "winter": "Water less often, usually every 3 to 5 days or when the topsoil feels dry. Avoid overwatering in cool weather.",
        "spring": "Water about 2 to 3 times a week and adjust based on heat, wind, and soil dryness.",
        "autumn": "Water about 2 times a week and keep the soil evenly moist without making it soggy.",
    }.get(season_lower, "Water regularly and adjust the amount based on heat, rainfall, and how quickly the soil dries.")

    fertilizer_schedule = (
        "Mix compost or aged manure into the soil before planting. Then apply a balanced fertilizer every 2 to 3 weeks during active growth. "
        "Reduce very heavy nitrogen feeding once flowering or fruiting starts."
    )

    html = build_html_list([
        f"Season plan used: {season_name} ({month_range}).",
        f"General watering need: {watering_text}",
        f"Simple watering schedule: {watering_schedule}",
        f"Live weather adjustment: {weather_context.get('watering_adjustment')}" if weather_context.get("watering_adjustment") else "",
        f"Simple fertilizer schedule: {fertilizer_schedule}",
    ])
    return html + build_source_note(source_name, "cultivation")


def build_pest_management_html(pest_text, source_name):
    prevention_items = [
        "Keep the growing area clean and remove dead or infected leaves quickly.",
        "Avoid overwatering and give the plants enough spacing for good airflow.",
        "Inspect the leaves, stems, and fruits every few days so problems are caught early.",
        "Use crop rotation, resistant varieties, and neem-based or approved sprays when needed.",
    ]

    issue_library = {
        "aphid": "Aphids: spray neem oil or insecticidal soap and wash off badly affected leaves.",
        "whitefly": "Whiteflies: use yellow sticky traps, improve airflow, and spray neem-based solutions.",
        "hornworm": "Hornworms or caterpillars: remove them by hand or use a biological control such as Bt if necessary.",
        "blight": "Blight or fungal spots: remove infected leaves, avoid wet foliage, and use a suitable fungicide if needed.",
        "late blight": "Late blight: remove infected foliage quickly and use a suitable fungicide before the disease spreads.",
        "wilt": "Wilt: improve drainage, avoid waterlogging, and remove badly infected plants to stop spread.",
        "nematode": "Nematodes: rotate crops and use healthy seedlings or resistant varieties.",
        "virus": "Virus issues: remove infected plants quickly and control sap-sucking insects like aphids and whiteflies.",
        "stem borer": "Stem borer: destroy infested shoots, keep the field clean, and use approved control at the early stage.",
        "leaf folder": "Leaf folder: monitor early leaf damage and use timely biological or approved chemical control if needed.",
        "blast": "Blast disease: avoid excess nitrogen, use resistant varieties, and apply fungicide when required.",
        "thrips": "Thrips: keep weeds down, use sticky traps, and spray approved control if the population rises.",
        "purple blotch": "Purple blotch: improve airflow, avoid wet leaves for long periods, and use a suitable fungicide if needed.",
        "sigatoka": "Sigatoka leaf spot: remove badly infected leaves and follow a regular disease management program.",
        "armyworm": "Armyworm: inspect the whorl early and use biological or approved control before heavy feeding starts.",
    }

    lower_text = clean_text(pest_text).lower()
    common_issues = [solution for keyword, solution in issue_library.items() if keyword in lower_text]
    if not common_issues:
        common_issues = [
            "Leaf spots or blight: remove affected leaves and keep foliage dry.",
            "Chewing insects: hand-pick visible pests and use neem or approved biological control if needed.",
            "Wilting or root rot: improve drainage and reduce excess watering.",
        ]

    return (
        "<p><strong>Prevention tips:</strong></p>"
        + build_html_list(prevention_items)
        + "<p><strong>Common issues and simple solutions:</strong></p>"
        + build_html_list(common_issues)
        + build_source_note(source_name, "protection")
    )


def build_harvest_html(harvest_text, source_text, source_name):
    days_match = re.search(r"(\d+)\s*days?\s*(?:to|-)\s*(?:more than\s*)?(\d+)\s*days?\s*to\s*harvest", source_text, flags=re.IGNORECASE)
    time_line = f"Typical harvest time: about {days_match.group(1)} to {days_match.group(2)} days, depending on variety and growing conditions." if days_match else f"Harvest note: {harvest_text}"

    html = build_html_list([
        time_line,
        f"Start harvesting when: {harvest_text}",
        "Harvest during the cooler part of the day and handle produce gently to reduce damage.",
    ])
    return html + build_source_note(source_name, "cultivation")


def build_yield_html(crop_name, market_details):
    yield_info = parse_yield_info(crop_name)
    items = []

    if yield_info:
        items.append(yield_info.get("text"))
        gross_return = estimate_gross_return(market_details.get("price_value"), market_details.get("price_unit"), yield_info)
        if gross_return:
            items.append(gross_return)
    else:
        items.append("Exact yield per acre can vary widely by variety, climate, irrigation, and crop care. Check local extension guidance for a location-specific figure.")

    if market_details.get("arrivals"):
        items.append(
            f"Recent market arrival reference: {format_number(market_details['arrivals'])} {market_details.get('arrival_unit', '')} reported at {market_details.get('market_name', 'the market')} on {market_details.get('date_label', 'the latest available date')}."
        )

    return build_html_list(items) + build_source_note("Agmarknet official market data", "market")


def build_post_harvest_html(storage_text, source_name):
    html = build_html_list([
        f"Storage tip: {storage_text}",
        "After harvest, sort the produce and remove damaged, diseased, or overripe pieces.",
        "Keep the harvested produce clean, shaded, and well ventilated to reduce spoilage.",
        "Pack carefully and transport gently to protect quality and maintain sale value.",
    ])
    return html + build_source_note(source_name, "postharvest")


def fetch_live_market_details(crop_name, location="", weather_context=None):
    crop_name = clean_text(crop_name)
    weather_context = weather_context or {}
    preferred_markets, fallback_markets, location_label, preferred_state = get_location_market_candidates(location, weather_context)

    fallback = {
        "price_text": f"Official mandi prices for {crop_name} change by date, grade, and market, so use the latest Agmarknet listing as the main selling reference for your area.",
        "yield_text": f"Per-acre yield for {crop_name} depends on variety, season, irrigation, and field care, so local agronomy guidance can refine the estimate.",
        "price_value": None,
        "price_unit": "",
        "market_name": "",
        "state_name": "",
        "date_label": "",
        "arrivals": None,
        "arrival_unit": "",
        "location_note": "",
    }

    if not crop_name:
        return fallback

    def search_market_group(markets, prefer_location=False):
        for days_back in range(0, 7):
            date_text = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")

            for market in markets:
                payload = {
                    "date": date_text,
                    "liveDate": date_text,
                    "marketIds": [market["market_id"]],
                    "stateIds": [market["state_id"]],
                }

                try:
                    response = requests.post(
                        PRICE_REPORT_URL,
                        json=payload,
                        headers={**BROWSER_HEADERS, "Content-Type": "application/json", "Accept": "application/json"},
                        timeout=12,
                    )
                    response.raise_for_status()
                    report = response.json()

                    for state in report.get("states", []):
                        for market_entry in state.get("markets", []):
                            for commodity in market_entry.get("commodities", []):
                                commodity_name = clean_text(commodity.get("commodityName", ""))
                                if crop_name.lower() not in commodity_name.lower() and commodity_name.lower() not in crop_name.lower():
                                    continue

                                data_rows = commodity.get("data", [])
                                if not data_rows:
                                    continue

                                row = data_rows[0]
                                modal_price = row.get("modalPrice")
                                min_price = row.get("minimumPrice")
                                max_price = row.get("maximumPrice")
                                arrivals = row.get("arrivals")
                                arrival_unit = clean_text(row.get("unitOfArrivals", ""))

                                if not is_number(modal_price):
                                    continue

                                market_name = clean_text(market_entry.get("marketName", market["market_name"]))
                                state_name = clean_text(state.get("stateName", market["state_name"]))
                                date_label = datetime.strptime(date_text, "%Y-%m-%d").strftime("%d-%b-%Y")
                                unit_display = "per quintal" if "quintal" in str(row.get("unitOfPrice", "")).lower() else clean_text(row.get("unitOfPrice", ""))

                                if location_label and prefer_location:
                                    price_text = f"Latest official mandi price near {location_label} on {date_label}: modal price ₹{format_number(modal_price)} {unit_display} at {market_name}, {state_name}."
                                    location_note = f"Location-matched market used: {market_name}, {state_name}."
                                elif location_label:
                                    price_text = f"Latest official mandi price reference for {location_label} on {date_label}: modal price ₹{format_number(modal_price)} {unit_display} at {market_name}, {state_name}."
                                    location_note = f"No recent official mandi data was found directly for {location_label}, so the nearest available official market reference was used."
                                else:
                                    price_text = f"Latest official mandi price on {date_label}: modal price ₹{format_number(modal_price)} {unit_display} at {market_name}, {state_name}."
                                    location_note = ""

                                if is_number(min_price) and is_number(max_price):
                                    price_text += f" Price range: ₹{format_number(min_price)} to ₹{format_number(max_price)}."

                                if is_number(arrivals):
                                    yield_text = f"Recent market arrival on {date_label} at {market_name}, {state_name}: {format_number(arrivals)} {arrival_unit}."
                                else:
                                    yield_text = fallback["yield_text"]

                                return {
                                    "price_text": price_text,
                                    "yield_text": yield_text,
                                    "price_value": float(modal_price),
                                    "price_unit": clean_text(row.get("unitOfPrice", "")),
                                    "market_name": market_name,
                                    "state_name": state_name,
                                    "date_label": date_label,
                                    "arrivals": float(arrivals) if is_number(arrivals) else None,
                                    "arrival_unit": arrival_unit,
                                    "location_note": location_note,
                                }
                except Exception:
                    continue

        return None

    local_result = search_market_group(preferred_markets, prefer_location=True) if preferred_markets else None
    if local_result:
        return local_result

    fallback_result = search_market_group(fallback_markets, prefer_location=False)
    if fallback_result:
        return fallback_result

    return fallback


def get_crop_details(crop_name, preferred_season="Current season", location=""):
    hint = get_crop_hint(crop_name)
    britannica_data = scrape_britannica_overview(crop_name)
    wikipedia_data = scrape_wikipedia_sections(crop_name)
    almanac_data = scrape_almanac_sections(crop_name)
    weather_context = get_live_weather_context(location)
    market_details = fetch_live_market_details(crop_name, location, weather_context)

    hint_source = hint.get("guidance_source", "CropCare trusted fallback guidance")
    if clean_text(hint_source).lower() == "standard crop guidance":
        hint_source = "CropCare trusted fallback guidance"

    general_source = britannica_data.get("source") or wikipedia_data.get("source") or almanac_data.get("source") or hint_source
    growing_source = almanac_data.get("source") or wikipedia_data.get("source") or britannica_data.get("source") or hint_source

    overview_source_text = first_non_empty(
        britannica_data.get("overview"),
        wikipedia_data.get("intro"),
        almanac_data.get("overview"),
    )
    planting_source_text = first_non_empty(
        almanac_data.get("when_to_plant"),
        almanac_data.get("planting"),
        wikipedia_data.get("cultivation"),
    )
    climate_source_text = f"{almanac_data.get('overview', '')} {almanac_data.get('growing', '')} {wikipedia_data.get('cultivation', '')}"
    soil_source_text = f"{almanac_data.get('planting', '')} {almanac_data.get('how_to_plant', '')} {wikipedia_data.get('cultivation', '')}"
    watering_source_text = f"{almanac_data.get('watering', '')} {almanac_data.get('feeding', '')} {almanac_data.get('growing', '')}"
    pest_source_text = f"{almanac_data.get('pests', '')} {wikipedia_data.get('pests', '')}"
    harvest_source_text = f"{almanac_data.get('harvest', '')} {wikipedia_data.get('harvest', '')} {almanac_data.get('overview', '')}"
    storage_source_text = f"{wikipedia_data.get('storage', '')} {almanac_data.get('storage', '')} {almanac_data.get('harvest', '')}"

    overview_fallback = first_non_empty(
        hint.get("overview"),
        f"{clean_text(crop_name).title()} is an important crop grown for food, farm income, and regular market demand.",
    )

    overview_text = clean_display_text(
        first_non_empty(
            overview_source_text,
            hint.get("overview"),
            overview_fallback,
        ),
        max_sentences=2,
    )

    planting_text = get_preferred_text(
        hint.get("planting_season"),
        planting_source_text,
        ["spring", "frost", "sow", "plant", "season", "weather"],
        overview_fallback,
    )
    climate_text = get_preferred_text(
        hint.get("climate"),
        climate_source_text,
        ["warm", "sun", "climate", "temperature", "rainfall", "humid", "frost"],
        overview_fallback,
    )
    soil_text = get_preferred_text(
        hint.get("soil"),
        soil_source_text,
        ["soil", "compost", "manure", "drained", "drainage", "pH"],
        overview_fallback,
    )
    fertilizer_text = get_preferred_text(
        hint.get("watering_fertilizer"),
        watering_source_text,
        ["water", "watering", "fertiliz", "moisture", "compost", "feed", "mulch"],
        overview_fallback,
    )
    sowing_source_text = first_non_empty(
        planting_source_text,
        hint.get("planting_season"),
        overview_text,
    )
    pest_text = get_preferred_text(
        hint.get("pest_summary"),
        pest_source_text,
        ["disease", "pest", "blight", "fung", "rot", "wilt", "nematode", "virus", "hornworm", "thrips", "borer", "blast"],
        overview_fallback,
        max_sentences=3,
    )
    harvest_text = get_preferred_text(
        hint.get("harvest"),
        harvest_source_text,
        ["harvest", "ripe", "ripen", "maturity", "days"],
        overview_fallback,
    )
    storage_text = get_preferred_text(
        hint.get("post_harvest"),
        storage_source_text,
        ["storage", "store", "shelf", "post-harvest", "ripen", "room temperature"],
        harvest_text,
    )

    description_source = pick_attribute_source(hint.get("overview"), overview_source_text, general_source, hint_source)
    climate_source = pick_attribute_source(hint.get("climate"), climate_source_text, growing_source, hint_source)
    planting_source = pick_attribute_source(hint.get("planting_season"), planting_source_text, growing_source, hint_source)
    watering_source = pick_attribute_source(hint.get("watering_fertilizer"), watering_source_text, growing_source, hint_source)
    pest_source = pick_attribute_source(hint.get("pest_summary"), pest_source_text, wikipedia_data.get("source") or growing_source, hint_source)
    harvest_source = pick_attribute_source(hint.get("harvest"), harvest_source_text, growing_source, hint_source)
    storage_source = pick_attribute_source(hint.get("post_harvest"), storage_source_text, almanac_data.get("source") or wikipedia_data.get("source") or growing_source, hint_source)
    soil_source = pick_attribute_source(hint.get("soil"), soil_source_text, growing_source, hint_source)
    name_source = wikipedia_data.get("source") or britannica_data.get("source") or description_source

    return {
        "display_name": build_name_html(crop_name, overview_text, wikipedia_data, name_source),
        "weather_location": weather_context.get("location_name", ""),
        "q1": build_planting_html(planting_text, preferred_season, planting_source),
        "q2": build_market_value_html(market_details, parse_yield_info(crop_name)),
        "q3": build_description_html(crop_name, overview_text, wikipedia_data, description_source),
        "q4": build_climate_html(climate_text, preferred_season, climate_source, weather_context),
        "q5": build_planting_html(planting_text, preferred_season, planting_source),
        "q6": build_watering_fertilizer_html(fertilizer_text, preferred_season, watering_source, weather_context),
        "q7": build_sowing_steps(crop_name, sowing_source_text, planting_source),
        "q8": build_pest_management_html(pest_text, pest_source),
        "q9": build_harvest_html(harvest_text, f"{almanac_data.get('overview', '')} {almanac_data.get('harvest', '')} {hint.get('harvest', '')}", harvest_source),
        "q10": build_yield_html(crop_name, market_details),
        "q11": build_post_harvest_html(storage_text, storage_source),
        "q12": build_soil_html(crop_name, soil_text, soil_source),
    }


# ── Flask app ──────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.config['SECRET_KEY'] = 'asndjasnd'
app.config['UPLOAD_FOLDER'] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "static",
    "uploads"
)
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


class UploadFileForm(FlaskForm):
    file = FileField("File", validators=[InputRequired()])
    submit = SubmitField("Upload")


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/crop-info', methods=['GET', 'POST'])
def crop_info():
    if request.method == "POST":
        query = clean_text(request.form.get('query'))
        preferred_season = clean_text(request.form.get('preferred_season')) or "Current season"
        location = clean_text(request.form.get('location'))

        if query:
            crop_details = get_crop_details(query, preferred_season, location)
            return render_template("main.html", name=query.title(), preferred_season=preferred_season, location=location, **crop_details)

    return render_template('test.html')


@app.route('/ai', methods=['GET', 'POST'])
def index():
    form = UploadFileForm()

    if form.validate_on_submit():
        file = form.file.data

        # FIX 2: check extension on the ORIGINAL filename, before prepending timestamp
        original_filename = secure_filename(file.filename)
        if not allowed_file(original_filename):
            flash("Please upload a PNG, JPG, JPEG, or GIF image.")
            return redirect(url_for('index'))

        # FIX 3: use MODEL_LOAD_ERROR (correct casing) instead of model_load_error
        if model is None:
            flash(f"Prediction model is unavailable. {MODEL_LOAD_ERROR or ''}".strip())
            return redirect(url_for('index'))

        filename = str(datetime.now().timestamp()) + "_" + original_filename
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_image_path)

        predicted_class, confidence = predict_disease(uploaded_image_path)
        plant_name, disease_name = format_prediction_details(predicted_class)

        return render_template(
            "result.html",
            plant_name=plant_name,
            disease_name=disease_name,
            confidence=round(confidence, 2),
            image_path=f"uploads/{filename}",
        )

    return render_template("ai.html", form=form)


if __name__ == '__main__':
    app.run(debug=True)
