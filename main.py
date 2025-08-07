import os
import csv
import json
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from langchain.tools import Tool


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"] 

@dataclass
class CompanyInfo:
    """Data class for company information"""
    company_name: str
    founding_date: str
    founders: List[str]

class CompanyExtraction(BaseModel):
    """Pydantic model for company extraction"""
    companies: List[Dict[str, Any]] = Field(
        description="List of companies with their details",
        default=[]
    )

class CompanyExtractor:
    """Main class for extracting company information from text"""
    
    def __init__(self, api_key: str = None):
        """Initialize the extractor with OpenAI API key"""
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0,
            max_tokens=2000
        )
        
        self.extraction_chain = self._create_extraction_chain()
        self.agent_executor = self._create_agent()
        
    def _create_extraction_chain(self):
        """Create LCEL chain for company information extraction"""
        
        # Define the extraction prompt
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting company information from text.
            
            Extract the following information for each company mentioned in the text:
            1. Company Name (exact name as mentioned)
            2. Founding Date (in YYYY-MM-DD format)
            3. Founders (list of founder names)
            
            Date formatting rules:
            - If only year is provided: use YYYY-01-01
            - If year and month are provided: use YYYY-MM-01
            - If full date is provided: use YYYY-MM-DD
            
            Return the information as a JSON object with this structure:
            {{
                "companies": [
                    {{
                        "company_name": "Company Name",
                        "founding_date": "YYYY-MM-DD",
                        "founders": ["Founder 1", "Founder 2"]
                    }}
                ]
            }}
            
            If no companies are found, return {{"companies": []}}.
            """),
            ("human", "Extract company information from this text:\n\n{text}")
        ])
        
        # Create output parser
        parser = JsonOutputParser(pydantic_object=CompanyExtraction)
        
        # Create the extraction chain using LCEL
        extraction_chain = (
            {"text": RunnablePassthrough()}
            | extraction_prompt
            | self.llm
            | parser
        )
        
        return extraction_chain
    
    def _normalize_date(self, date_str: str) -> str:
        """Normalize date string to YYYY-MM-DD format"""
        if not date_str:
            return ""
        
        # Remove common words and clean the string
        clean_date = re.sub(r'\b(in|on|during|established|founded|created)\b', '', date_str, flags=re.IGNORECASE).strip()
        
        # Try to extract date patterns
        patterns = [
            r'(\d{4})-(\d{1,2})-(\d{1,2})',  # YYYY-MM-DD or YYYY-M-D
            r'(\d{1,2})/(\d{1,2})/(\d{4})',  # MM/DD/YYYY or M/D/YYYY
            r'(\d{4})/(\d{1,2})/(\d{1,2})',  # YYYY/MM/DD or YYYY/M/D
            r'(\d{4})-(\d{1,2})',            # YYYY-MM or YYYY-M
            r'(\d{4})',                      # YYYY only
        ]
        
        for pattern in patterns:
            match = re.search(pattern, clean_date)
            if match:
                groups = match.groups()
                if len(groups) == 3:
                    if pattern.startswith(r'(\d{1,2})/(\d{1,2})/(\d{4})'):
                        # MM/DD/YYYY format
                        month, day, year = groups
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                    else:
                        # YYYY-MM-DD or YYYY/MM/DD format
                        year, month, day = groups
                        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
                elif len(groups) == 2:
                    # YYYY-MM format
                    year, month = groups
                    return f"{year}-{month.zfill(2)}-01"
                elif len(groups) == 1:
                    # YYYY only
                    year = groups[0]
                    return f"{year}-01-01"
        
        return date_str  # Return original if no pattern matches
    
    @tool
    def extract_companies_from_paragraph(self, paragraph: str) -> Dict[str, Any]:
        """Tool to extract company information from a single paragraph"""
        try:
            result = self.extraction_chain.invoke(paragraph)
            
            # Normalize dates in the result
            if isinstance(result, dict) and "companies" in result:
                for company in result["companies"]:
                    if "founding_date" in company:
                        company["founding_date"] = self._normalize_date(company["founding_date"])
            
            return result
        except Exception as e:
            print(f"Error extracting from paragraph: {e}")
            return {"companies": []}
    
    @tool
    def save_to_csv(self, companies_data: List[Dict[str, Any]], filename: str = "company_info.csv") -> str:
        """Tool to save company data to CSV file"""
        try:
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['company_name', 'founding_date', 'founders']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for company in companies_data:
                    # Convert founders list to comma-separated string
                    founders_str = ", ".join(company.get('founders', []))
                    
                    writer.writerow({
                        'company_name': company.get('company_name', ''),
                        'founding_date': company.get('founding_date', ''),
                        'founders': founders_str
                    })
            
            return f"Successfully saved {len(companies_data)} companies to {filename}"
        except Exception as e:
            return f"Error saving to CSV: {e}"
    
    def _create_agent(self):
        """Create an intelligent agent with tools"""
        tools = [
        Tool.from_function(
            func=self.extract_companies_from_paragraph,
            name="extract_companies_from_paragraph",
            description="Extracts company information from a paragraph of text"
        ),
        Tool.from_function(
            func=self.save_to_csv,
            name="save_to_csv",
            description="Saves extracted company info to CSV"
        )
    ]
    
    def process_text_direct(self, text: str) -> List[CompanyInfo]:
        """Direct processing without agent for simpler use cases"""
        all_companies = []
        
        # Split into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        for paragraph in paragraphs:
            try:
                result = self.extraction_chain.invoke(paragraph)
                
                if isinstance(result, dict) and "companies" in result:
                    for company_data in result["companies"]:
                        company = CompanyInfo(
                            company_name=company_data.get("company_name", ""),
                            founding_date=self._normalize_date(company_data.get("founding_date", "")),
                            founders=company_data.get("founders", [])
                        )
                        all_companies.append(company)
            except Exception as e:
                print(f"Error processing paragraph: {e}")
                continue
        
        # Save to CSV
        self._save_companies_to_csv(all_companies)
        return all_companies
    
    def _save_companies_to_csv(self, companies: List[CompanyInfo], filename: str = "company_info.csv"):
        """Save companies to CSV file"""
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['company_name', 'founding_date', 'founders']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for company in companies:
                writer.writerow({
                    'company_name': company.company_name,
                    'founding_date': company.founding_date,
                    'founders': ", ".join(company.founders)
                })

def main():
    """Main function to demonstrate the system"""
    
    # Sample essay text about companies
    sample_text = """
    In the ever-evolving landscape of global commerce, the origin stories of major corporations are not merely tales of personal ambition and entrepreneurial spirit but also reflections of broader socio-economic trends and technological revolutions that have reshaped industries. These narratives, which often begin with modest ambitions, unfold into chronicles of innovation and strategic foresight that define industries and set benchmarks for future enterprises.

    Early Foundations: Pioneers of Industry
    One of the earliest examples is The Coca-Cola Company, founded on May 8, 1886, by Dr. John Stith Pemberton in Atlanta, Georgia. Initially sold at Jacob's Pharmacy as a medicinal beverage, Coca-Cola would become one of the most recognized brands worldwide, revolutionizing the beverage industry.
    Similarly, Sony Corporation was established on May 7, 1946, by Masaru Ibuka and Akio Morita in Tokyo, Japan. Starting with repairing and building electrical equipment in post-war Japan, Sony would grow to pioneer electronics, entertainment, and technology.
    As the mid-20th century progressed, McDonald's Corporation emerged as a game-changer in the fast-food industry. Founded on April 15, 1955, in Des Plaines, Illinois, by Ray Kroc, McDonald's built upon the original concept of Richard and Maurice McDonald to standardize and scale fast-food service globally. Around the same period, Intel Corporation was established on July 18, 1968, by Robert Noyce and Gordon Moore in Mountain View, California

    driving advancements in semiconductors and microprocessors that became the backbone of modern computing.

    The Rise of Technology Titans
    Samsung Electronics Co., Ltd., founded on January 13, 1969, by Lee Byung-chul in Su-dong, South Korea, initially focused on producing electrical appliances like televisions and refrigerators. As Samsung expanded into semiconductors, telecommunications, and digital media, it
    grew into a global technology leader. Similarly, Microsoft Corporation was founded on April 4, 1975, by Bill Gates and Paul Allen in Albuquerque, New Mexico, with the vision of placing a computer on every desk and in every home.
    In Cupertino, California, Apple Inc. was born on April 1, 1976, founded by Steve Jobs, Steve Wozniak, and Ronald Wayne. Their mission to make personal computing accessible and elegant revolutionized technology and design. A few years later, Oracle Corporation was established on June 16, 1977, by Larry Ellison, Bob Miner, and Ed Oates in Santa Clara, California.
    Specializing in relational databases, Oracle would become a cornerstone of enterprise software and cloud computing.
    NVIDIA Corporation, founded on April 5, 1993, by Jensen Huang, Chris Malachowsky, and Curtis Priem in Santa Clara, California, began with a focus on graphics processing units (GPUs) for gaming. Today, NVIDIA is a leader in artificial intelligence, deep learning, and autonomous systems, showcasing the power of continuous innovation.

    E-Commerce and the Internet Revolution
    The 1990s witnessed a dramatic shift toward e-commerce and internet technologies. Amazon.com Inc. was founded on July 5, 1994, by Jeff Bezos in a garage in Bellevue, Washington, with the vision of becoming the world's largest online bookstore. This vision rapidly expanded to encompass
    e-commerce, cloud computing, and digital streaming. Similarly, Google LLC was founded on September 4, 1998, by Larry Page and Sergey Brin, PhD students at Stanford University, in a garage in Menlo Park, California.
    Google's mission to "organize the world's information" transformed how we search, learn, and connect.
    In Asia, Alibaba Group Holding Limited was founded on June 28, 1999, by Jack Ma and 18 colleagues in Hangzhou, China. Originally an e-commerce platform connecting manufacturers with buyers, Alibaba expanded into cloud

    computing, digital entertainment, and financial technology, becoming a global powerhouse.
    In Europe, SAP SE was founded on April 1, 1972, by Dietmar Hopp,
    Hans-Werner Hector, Hasso Plattner, Klaus Tschira, and Claus Wellenreuther in Weinheim, Germany. Specializing in enterprise resource planning (ERP) software, SAP revolutionized how businesses manage operations and data.

    Social Media and Digital Platforms
    The 2000s brought a wave of social media and digital platforms that reshaped communication and commerce. LinkedIn Corporation was founded on December 28, 2002, by Reid Hoffman and a team from PayPal and Socialnet.com in Mountain View, California, focusing on professional networking.
    Facebook, Inc. (now Meta Platforms, Inc.) was launched on February 4, 2004, by Mark Zuckerberg and his college roommates in Cambridge, Massachusetts, evolving into a global social networking behemoth.
    Another transformative platform, Twitter, Inc., was founded on March 21, 2006, by Jack Dorsey, Biz Stone, and Evan Williams in San Francisco, California. Starting as a microblogging service, Twitter became a critical tool for communication and social commentary. Spotify AB, founded on April 23, 2006, by Daniel Ek and Martin Lorentzon in Stockholm, Sweden, leveraged streaming technology to democratize music consumption, fundamentally altering the music industry.
    In the realm of video-sharing, YouTube LLC was founded on February 14, 2005, by Steve Chen, Chad Hurley, and Jawed Karim in San Mateo, California. YouTube became the leading platform for user-generated video content, influencing global culture and media consumption.

    Innovators in Modern Technology
    Tesla, Inc., founded on July 1, 2003, by a group including Elon Musk, Martin Eberhard, Marc Tarpenning, JB Straubel, and Ian Wright, in San Carlos, California, championed the transition to sustainable energy with its electric vehicles and energy solutions. Airbnb, Inc., founded in August 2008 by Brian Chesky, Joe Gebbia, and Nathan Blecharczyk in San Francisco, California, disrupted traditional hospitality with its peer-to-peer lodging platform.
    In the realm of fintech, PayPal Holdings, Inc. was established in December 1998 by Peter Thiel, Max Levchin, Luke Nosek, and Ken Howery in Palo Alto,

    California. Originally a cryptography company, PayPal became a global leader in online payments. Stripe, Inc., founded in 2010 by Patrick and John Collison in Palo Alto, California, followed suit, simplifying online payments and enabling digital commerce.
    Square, Inc. (now Block, Inc.), founded on February 20, 2009, by Jack Dorsey and Jim McKelvey in San Francisco, California, revolutionized mobile payment systems with its simple and accessible card readers.

    Recent Disruptors
    Zoom Video Communications, Inc. was founded on April 21, 2011, by Eric Yuan in San Jose, California. Initially designed for video conferencing, Zoom became essential during the COVID-19 pandemic, transforming remote work and communication. Slack Technologies, LLC, founded in 2009 by Stewart Butterfield, Eric Costello, Cal Henderson, and Serguei Mourachov in Vancouver, Canada, redefined workplace communication with its innovative messaging platform.
    Rivian Automotive, Inc., founded on June 23, 2009, by RJ Scaringe in Plymouth, Michigan, entered the electric vehicle market with a focus on adventure and sustainability. SpaceX, established on March 14, 2002, by Elon Musk in Hawthorne, California, revolutionized aerospace with reusable rockets and ambitious plans for Mars exploration.
    TikTok, developed by ByteDance and launched in September 2016 by Zhang Yiming in Beijing, China, revolutionized short-form video content, becoming a cultural phenomenon worldwide.

    Conclusion
    These corporations, with their diverse beginnings and visionary founders, exemplify the interplay of innovation, timing, and strategic foresight that shapes industries and transforms markets. From repairing electronics in post-war Japan to building global e-commerce empires and redefining space exploration, their stories are milestones in the narrative of global economic transformation. Each reflects not only the aspirations of their founders but also the technological advancements and socio-economic trends of their time, serving as inspirations for future innovators. 
    """
    
    print("=== Company Information Extraction System ===\n")
    
    # Initialize the extractor (replace with your actual API key)
    extractor = CompanyExtractor(api_key=OPENAI_API_KEY)

    companies = extractor.process_text_direct(sample_text)
    
    print(f"Found {len(companies)} companies:")
    for i, company in enumerate(companies, 1):
        print(f"{i}. {company.company_name}")
        print(f"   Founded: {company.founding_date}")
        print(f"   Founders: {', '.join(company.founders)}")
        print()
    
    print("CSV file 'company_info.csv' has been generated!")

if __name__ == "__main__":
    main()