#!/usr/bin/env python3
"""
Create a benchmark dataset for model evaluation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

def create_benchmark_questions() -> List[Dict[str, str]]:
    """Create a curated set of benchmark questions for Egypt tourism"""
    
    benchmark_qa = [
        {
            "question": "What are the top 3 must-visit attractions in Egypt?",
            "answer": "The top 3 must-visit attractions in Egypt are: 1) The Great Pyramids of Giza - the last surviving wonder of the ancient world, 2) The Egyptian Museum in Cairo - home to over 120,000 artifacts including Tutankhamun's treasures, and 3) The Valley of the Kings in Luxor - where pharaohs were buried with their treasures."
        },
        {
            "question": "Do I need a visa to visit Egypt as a tourist?",
            "answer": "Yes, most visitors need a visa to enter Egypt. You can obtain a tourist visa upon arrival at major airports for $25 USD, valid for 30 days. Alternatively, you can apply for an e-visa online before travel. Citizens of some countries may be exempt from visa requirements."
        },
        {
            "question": "What is the best time of year to visit Egypt?",
            "answer": "The best time to visit Egypt is during the cooler months from October to April. The peak tourist season is from December to February when temperatures are mild (15-25°C). Avoid visiting during summer (June-August) when temperatures can exceed 40°C, especially in southern Egypt."
        },
        {
            "question": "Is it safe to travel to Egypt right now?",
            "answer": "Egypt is generally safe for tourists, with millions visiting annually. However, it's important to stay informed about current travel advisories, avoid border areas, and follow local guidance. Tourist areas like Cairo, Luxor, and Aswan are well-protected and safe for visitors."
        },
        {
            "question": "What currency should I bring to Egypt?",
            "answer": "The official currency of Egypt is the Egyptian Pound (EGP). While some tourist areas accept US dollars and euros, it's best to carry Egyptian pounds for most transactions. You can exchange money at banks, hotels, or exchange bureaus. Credit cards are accepted in major hotels and restaurants."
        },
        {
            "question": "How much should I tip in Egypt?",
            "answer": "Tipping (baksheesh) is expected in Egypt. For restaurants, tip 10-15% of the bill. For hotel staff, tip 10-20 EGP per day for housekeeping. For tour guides, tip 50-100 EGP per day. For drivers, tip 20-50 EGP per day. Always tip in local currency."
        },
        {
            "question": "What should I wear when visiting religious sites in Egypt?",
            "answer": "When visiting mosques and religious sites, dress modestly. Women should cover their shoulders, arms, and legs (wear long skirts or pants). Men should wear long pants and shirts with sleeves. Remove shoes before entering mosques. Some sites may provide coverings if needed."
        },
        {
            "question": "How do I get around in Cairo?",
            "answer": "Cairo has several transportation options: 1) Metro - clean, cheap, and efficient for major routes, 2) Taxis - negotiate fares in advance or use ride-sharing apps, 3) Uber/Careem - convenient and safe, 4) Buses - cheap but crowded, 5) Walking - good for short distances in tourist areas."
        },
        {
            "question": "What are the best souvenirs to buy in Egypt?",
            "answer": "Popular Egyptian souvenirs include: 1) Papyrus paintings and scrolls, 2) Alabaster and stone carvings, 3) Gold and silver jewelry, 4) Spices and essential oils, 5) Cotton and linen textiles, 6) Replica artifacts, 7) Traditional perfumes, 8) Handmade carpets and rugs."
        },
        {
            "question": "Can I drink tap water in Egypt?",
            "answer": "No, it's not recommended to drink tap water in Egypt. Stick to bottled water, which is widely available and inexpensive. Also avoid ice in drinks and raw vegetables that may have been washed in tap water. Use bottled water for brushing teeth as well."
        },
        {
            "question": "What languages are spoken in Egypt?",
            "answer": "The official language of Egypt is Arabic (Egyptian Arabic dialect). English is widely spoken in tourist areas, hotels, and by tour guides. French is also spoken by some educated Egyptians. Learning basic Arabic phrases like 'hello' (marhaba) and 'thank you' (shukran) is appreciated."
        },
        {
            "question": "How much does a typical meal cost in Egypt?",
            "answer": "Food costs in Egypt vary: 1) Street food: 10-30 EGP, 2) Local restaurants: 50-150 EGP per person, 3) Mid-range restaurants: 150-300 EGP per person, 4) High-end restaurants: 300+ EGP per person. Prices are higher in tourist areas and luxury hotels."
        },
        {
            "question": "What are the main airports in Egypt?",
            "answer": "Egypt's main international airports are: 1) Cairo International Airport (CAI) - the largest and busiest, 2) Hurghada International Airport (HRG) - for Red Sea resorts, 3) Sharm El Sheikh International Airport (SSH) - for Sinai Peninsula, 4) Luxor International Airport (LXR) - for Nile Valley tourism."
        },
        {
            "question": "Is bargaining expected in Egyptian markets?",
            "answer": "Yes, bargaining (haggling) is expected and expected in Egyptian markets, especially in souks and bazaars. Start by offering 50-60% of the asking price and negotiate from there. Be polite but firm. Tourist areas may have fixed prices, but smaller shops usually expect bargaining."
        },
        {
            "question": "What are the most important cultural customs to know?",
            "answer": "Important Egyptian customs include: 1) Greet with 'As-salamu alaykum' (peace be upon you), 2) Use your right hand for eating and handshakes, 3) Remove shoes before entering homes, 4) Dress modestly, especially women, 5) Accept tea/coffee when offered, 6) Don't show the soles of your feet, 7) Be patient with bureaucracy."
        }
    ]
    
    return benchmark_qa

def main():
    """Create and save the benchmark dataset"""
    print("📊 Creating benchmark dataset...")
    
    # Create benchmark questions
    benchmark_qa = create_benchmark_questions()
    
    # Create output directory
    output_dir = Path("data/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save benchmark dataset
    benchmark_file = output_dir / "egypt_tourism_benchmark.json"
    
    benchmark_data = {
        "metadata": {
            "description": "Benchmark dataset for Egypt tourism Q&A evaluation",
            "num_questions": len(benchmark_qa),
            "created_date": "2024-07-25",
            "purpose": "Compare base model vs fine-tuned model performance"
        },
        "questions": benchmark_qa
    }
    
    with open(benchmark_file, 'w', encoding='utf-8') as f:
        json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Benchmark dataset created with {len(benchmark_qa)} questions")
    print(f"📁 Saved to: {benchmark_file}")
    
    # Print sample questions
    print("\n📋 Sample questions:")
    for i, qa in enumerate(benchmark_qa[:3], 1):
        print(f"{i}. {qa['question']}")

if __name__ == "__main__":
    main() 