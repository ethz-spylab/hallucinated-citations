#!/usr/bin/env python3

import os
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import re

def extract_arxiv_id(identifier):
    """Extract arxiv ID from the oai identifier or URL"""
    if 'arXiv.org:' in identifier:
        return identifier.split('arXiv.org:')[1]
    elif 'arxiv.org/abs/' in identifier:
        return identifier.split('arxiv.org/abs/')[1]
    return None

def parse_xml_content(xml_content):
    """Parse XML content and extract paper information"""
    papers = []
    multi_versioned = 0
    
    try:
        root = ET.fromstring(xml_content)
        
        # Find all record elements with proper namespace
        for record in root.findall('.//{http://www.openarchives.org/OAI/2.0/}record'):
            paper = {}
            
            # Extract arxiv identifier from header
            header = record.find('.//{http://www.openarchives.org/OAI/2.0/}identifier')
            if header is not None:
                arxiv_id = extract_arxiv_id(header.text)
                if arxiv_id:
                    paper['arxiv_id'] = arxiv_id
            
            # Extract metadata
            metadata = record.find('.//{http://www.openarchives.org/OAI/2.0/}metadata')
            if metadata is not None:
                dc_elem = metadata.find('.//{http://www.openarchives.org/OAI/2.0/oai_dc/}dc')
                if dc_elem is not None:
                    # Extract title
                    title_elem = dc_elem.find('.//{http://purl.org/dc/elements/1.1/}title')
                    if title_elem is not None:
                        paper['title'] = title_elem.text.strip()
                    
                    # Extract authors
                    author_elems = dc_elem.findall('.//{http://purl.org/dc/elements/1.1/}creator')
                    authors = []
                    for author_elem in author_elems:
                        if author_elem.text:
                            authors.append(author_elem.text.strip())
                    paper['authors'] = authors
            
            # Extract date
            dates = dc_elem.findall('.//{http://purl.org/dc/elements/1.1/}date')
            if len(dates) > 1:
                multi_versioned += 1

            # Only add paper if we have at least title and arxiv_id
            if 'title' in paper and 'arxiv_id' in paper:
                papers.append(paper)

    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
    
    return papers, multi_versioned

def process_zst_file(file_path):
    """Process a single .zst file and extract papers"""
    print(f"Processing {file_path.name}...")
    
    try:
        # Use system zstd command to decompress
        result = subprocess.run(['zstd', '-dc', str(file_path)], 
                              capture_output=True, text=True, check=True)
        xml_content = result.stdout
        return parse_xml_content(xml_content)
    except subprocess.CalledProcessError as e:
        print(f"Error decompressing {file_path}: {e}")
        return [], 0
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return [], 0

def load_checkpoint():
    """Load the last processed file from checkpoint"""
    checkpoint_file = Path("last_processing.json")
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            return json.load(f).get('last_processed_file', None)
    return None

def save_checkpoint(filename):
    """Save the last processed file to checkpoint"""
    checkpoint_file = Path("/data/local/home/ftramer/citations/last_processing.json")
    with open(checkpoint_file, 'r') as f:
        d = json.load(f)
        d["last_processed_file"] = filename

    with open(checkpoint_file, 'w') as f:
        json.dump(d, f)

def load_existing_papers():
    """Load existing papers from arxiv_papers.json"""
    output_file = Path("/data/local/home/ftramer/citations/arxiv_papers.json")
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def main():
    arxiv_dir = Path("/data/projects/arxiv/I29haV9kYyNodHRwOi8vZXhwb3J0LmFyeGl2Lm9yZy9vYWky")
    output_file = Path("/data/local/home/ftramer/citations/arxiv_papers.json")
    
    # Load existing papers
    all_papers = load_existing_papers()
    all_multi_versioned = 0
    
    # Get checkpoint
    last_processed = load_checkpoint()
    
    # Get all .xml.zst files
    zst_files = list(arxiv_dir.glob("*.xml.zst"))
    zst_files.sort()
    
    # Skip files up to the last processed one
    start_index = 0
    if last_processed:
        for i, zst_file in enumerate(zst_files):
            if zst_file.name == last_processed:
                start_index = i + 1
                break
        print(f"Resuming from file {start_index + 1}/{len(zst_files)} (last processed: {last_processed})")
    
    print(f"Found {len(zst_files)} .xml.zst files, starting from index {start_index}")
    print(f"Already have {len(all_papers)} papers from previous runs")
    
    for i, zst_file in enumerate(zst_files[start_index:], start_index):
        papers, multi_versioned = process_zst_file(zst_file)
        all_papers.extend(papers)
        all_multi_versioned += multi_versioned
        
        # Save checkpoint after each file
        save_checkpoint(zst_file.name)
        
        # Save progress to file every 10 files
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(zst_files)} files, found {len(all_papers)} papers so far")
            print(f"{all_multi_versioned} papers have multiple versions")
            
            # Save intermediate results
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_papers, f, indent=2, ensure_ascii=False)
            print(f"Intermediate results saved to {output_file}")
    
    print(f"Total papers extracted: {len(all_papers)}")
    print(f"Total papers with multiple versions: {all_multi_versioned}")
    
    # Save final results to JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False)
    
    print(f"Papers saved to {output_file}")

if __name__ == "__main__":
    main()
