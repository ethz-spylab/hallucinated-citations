#!/usr/bin/env python3

import os
import json
import re
import unicodedata
from unidecode import unidecode
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
import editdistance
from datetime import datetime

def normalize_text(text):
    """Normalize text for soft matching by lowercasing and removing non-ASCII chars"""
    if not text:
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Remove accents and normalize unicode
    #text = unicodedata.normalize('NFKD', text)
    #text = ''.join(c for c in text if ord(c) < 128)
    text = unidecode(text)
    
    # Remove extra whitespace and common punctuation that might vary
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def extract_last_names(authors):
    """Extract last names from author list, limit to 10 authors"""
    if not authors:
        return [], []
    
    # Limit to 10 authors
    authors = authors[:10]
    
    last_names = []
    first_names = []
    for author in authors:
        if isinstance(author, dict):
            # JSON format: {"family": "Smith", "given": "John"}
            family = author.get("family", author.get("given", ""))
            if family and family != "al":
                last_names.append(normalize_text(family))
                first_names.append(None)
        elif isinstance(author, str):
            # String format: "Smith, John" or "John Smith"
            if ',' in author:
                # Format: "Last, First"
                parts = author.split(',')
                last_names.append(normalize_text(parts[0]))
                first_names.append(normalize_text(parts[1]))
            else:
                # Format: "First Last" - take last word
                parts = author.split()
                if parts:
                    last_names.append(normalize_text(parts[-1]))
                    first_names.append(normalize_text(parts[0]))
    
    def keep_main(l):
        if l is None:
            return l
        val = ""
        length = 0
        for v in l.split():
            if len(v) > length:
                length = len(v)
                val = v
        return val

    assert len(first_names) == len(last_names)
    return [keep_main(ln) for ln in last_names], [keep_main(ln) for ln in first_names]

def extract_arxiv_id(reference):
    """Extract arXiv ID from reference URL or note fields"""
    arxiv_patterns = [
        r'arxiv[\.:](\d{4}\.\d{4,5})',
        r'arXiv:(\d{4}\.\d{4,5})',
        r'(?:\D|^)+(\d{4}\.\d{4,5})(?:\D|$)+',
        r'arxiv\.org/abs/(\d{4}\.\d{4,5})',
        r'arxiv\.org/pdf/(\d{4}\.\d{4,5})',
        #r'(\w+-\w+/\d{7})',  # Old format like "cs-LG/0001001"
    ]
    
    # Check URL field
    urls = reference.get("url", [])
    if isinstance(urls, str):
        urls = [urls]
    
    for url in urls:
        for pattern in arxiv_patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                return match.group(1)
    
    # Check note field
    notes = reference.get("note", [])
    if isinstance(notes, str):
        notes = [notes]
    
    for note in notes:
        for pattern in arxiv_patterns:
            match = re.search(pattern, note, re.IGNORECASE)
            if match:
                return match.group(1)
    
    return None

def get_reference_title(reference):
    """Extract title from reference, ignore if multiple titles (processing failed)"""
    titles = reference.get("title", [])
    if isinstance(titles, list):
        if len(titles) == 1:
            return titles[0]
        elif len(titles) > 1:
            # Multiple titles usually means processing failed, ignore
            return ""
        else:
            return ""
    elif isinstance(titles, str):
        return titles
    return ""

def get_reference_authors(reference):
    """Extract authors from reference"""
    authors = reference.get("author", [])
    if isinstance(authors, list):
        return authors
    return []

def similarity_ratio(s1, s2):

    if s1 == s2:
        return 1.0

    if s1 in s2 or s2 in s1:
        return 1.0

    """Calculate similarity ratio between two strings"""
    return SequenceMatcher(None, s1, s2).ratio()

class ArxivDatabase:
    def __init__(self, arxiv_papers_file):
        print("Loading arXiv papers database...")
        with open(arxiv_papers_file, 'r', encoding='utf-8') as f:
            self.papers = json.load(f)
        
        print(f"Loaded {len(self.papers)} arXiv papers")
        
        # Create indexes for fast lookup
        self.by_arxiv_id = {}
        self.by_normalized_title = defaultdict(list)
        self.title_duplicates = defaultdict(list)
        
        print("Building search indexes...")
        for paper in self.papers:
            arxiv_id = paper['arxiv_id']
            title = paper['title']
            authors = paper['authors']
            
            # Index by arXiv ID
            self.by_arxiv_id[arxiv_id] = paper
            
            # Index by normalized title
            norm_title = normalize_text(title)
            self.by_normalized_title[norm_title].append(paper)
            
            # Track title duplicates
            if len(self.by_normalized_title[norm_title]) > 1:
                self.title_duplicates[norm_title] = self.by_normalized_title[norm_title]
        
        print(f"Found {len(self.title_duplicates)} duplicate titles after normalization")
        
        # Remove papers with identical titles but different authors to avoid false positives
        titles_to_remove = []
        for norm_title, papers in self.title_duplicates.items():
            # Check if papers have different author sets
            author_sets = []
            for paper in papers:
                author_set = set(extract_last_names(paper['authors'])[0])
                author_sets.append(author_set)
            
            # Check if all papers have identical author sets (last names)
            has_identical_authors = True
            if len(author_sets) > 1:
                first_set = author_sets[0]
                for author_set in author_sets[1:]:
                    if author_set != first_set:
                        has_identical_authors = False
                        break
            
            # Remove duplicates unless author lists are identical
            if not has_identical_authors:
                titles_to_remove.append(norm_title)
        
        # Remove ambiguous titles
        removed_papers = 0
        for norm_title in titles_to_remove:
            papers_to_remove = self.by_normalized_title[norm_title]
            removed_papers += len(papers_to_remove)
            del self.by_normalized_title[norm_title]
            del self.title_duplicates[norm_title]
        
        print(f"Removed {removed_papers} papers with ambiguous titles (same title, different authors)")
        
        # Print some examples of remaining duplicate titles
        if self.title_duplicates:
            print(f"Kept {len(self.title_duplicates)} titles with multiple papers (same authors)")
            print("\nExamples of kept duplicate titles:")
            for i, (norm_title, papers) in enumerate(list(self.title_duplicates.items())[:3]):
                print(f"  Title: '{norm_title}'")
                for paper in papers[:2]:  # Show first 2
                    print(f"    - {paper['arxiv_id']}: {paper['title']}")
                    print(f"      Authors: {paper['authors'][:3]}")
                print()
    
    def lookup_by_arxiv_id(self, arxiv_id):
        """Lookup paper by arXiv ID"""
        return self.by_arxiv_id.get(arxiv_id)
    
    def search_by_title(self, title):
        """Search for papers by exact title match only"""
        norm_title = normalize_text(title)
        
        # Only exact match - no fuzzy matching for speed
        if norm_title in self.by_normalized_title:
            return self.by_normalized_title[norm_title]
        
        return []
    
def check_author_match(ref_authors, paper_authors, threshold=0.45, last=False):
    """Check if author lists roughly match"""
    ref_last_names, _ = extract_last_names(ref_authors)
    paper_last_names, paper_last_names_alt = extract_last_names(paper_authors)
    
    if not ref_last_names or not paper_last_names:
        return False, 0.0

    # crop if one list is shorter (eg different cropping for et al.)
    l = min(len(ref_last_names), len(paper_last_names), 5)
    ref_last_names = ref_last_names[:l]
    paper_last_names = paper_last_names[:l]
    paper_last_names_alt = paper_last_names_alt[:l]

    def near_match(a1, a2, a2_alt):
        if a1 == a2:
            return True

        if a1 == a2_alt:
            return True

        d = editdistance.eval(a1, a2)
        if d <= 2:
            return True
        
        if a2_alt is not None and editdistance.eval(a1, a2_alt) <= 2:
            return True
        
        return False

    score = 0
    for a1 in ref_last_names:
        found = False
        for a2, a2_alt in zip(paper_last_names, paper_last_names_alt):
            if not found and near_match(a1, a2, a2_alt):
                score += 1
                found = True

    score /= len(ref_last_names)

    if score < threshold and not last:
        l = len(ref_authors)
        _, score2 = check_author_match(ref_authors, paper_authors[-l:], threshold=threshold, last=True)
        score = max(score, score2)
    return score >= threshold, score

def get_latest_processed_date():
    """Get the latest processed date (YYMM) from last_processing.json"""
    try:
        with open('last_processing.json', 'r') as f:
            data = json.load(f)
            last_dir = data.get('last_processed_directory', 0)
            return str(last_dir) if last_dir >= 2500 else "2506"  # Default to 2506 if no valid date
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return "2506"  # Default starting point

def save_checkpoint(directory_index):
    """Save the last processed directory index to checkpoint"""
    checkpoint_file = "last_processing.json"
    data = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)

    data['last_processed_directory'] = int(directory_index)

    with open(checkpoint_file, 'w') as f:
        json.dump(data, f)

def extract_date_from_filename(filename):
    """Extract YYMM from filename like '2506.24062.json'"""
    parts = filename.split('.')
    if len(parts) >= 2 and parts[0].isdigit() and len(parts[0]) == 4:
        return parts[0]
    return None

def get_directory_date_range(directory_path):
    """Get the date range (min, max YYMM) of files in a directory"""
    json_files = list(Path(directory_path).glob("*.json"))
    dates = []
    
    for json_file in json_files:
        date = extract_date_from_filename(json_file.name)
        if date:
            dates.append(date)
    
    if dates:
        return min(dates), max(dates)
    return None, None

def find_unprocessed_directories(last_processed_yymm):
    """Find directories containing files with dates later than last processed YYMM"""
    DIR = "refs++"
    done_dirs = [f"{DIR}/{d}" for d in os.listdir(DIR) if d.endswith('_done') and os.path.isdir(f"{DIR}/{d}")]
    done_dirs.sort()
    
    unprocessed_dirs = []
    for done_dir in done_dirs:
        min_date, max_date = get_directory_date_range(done_dir)
        if max_date and max_date > last_processed_yymm:
            unprocessed_dirs.append(done_dir)
            print(f"Found unprocessed directory: {done_dir} (date range: {min_date}-{max_date})")
    
    return unprocessed_dirs

def process_reference_files(arxiv_db, target_dirs=None):
    """Process reference files and check matches"""
    results = {
        'total_references': 0,
        'arxiv_id_matches': 0,
        'arxiv_id_correct': 0,
        'title_matches': 0,
        'title_with_author_matches': 0,
        'no_matches': 0,
        'details': []
    }
    
    # Use target_dirs if provided, otherwise process all directories
    if target_dirs is None:
        DIR = "refs++"
        done_dirs = [f"{DIR}/{d}" for d in os.listdir(DIR) if d.endswith('_done') and os.path.isdir(f"{DIR}/{d}")]
        done_dirs.sort()
    else:
        done_dirs = target_dirs
    
    print(f"Found {len(done_dirs)} reference directories")
    
    for i, done_dir in enumerate(done_dirs):
        print(f"Processing directory {i+1}/{len(done_dirs)}: {done_dir}")
        
        json_files = list(Path(done_dir).glob("*.json"))
        
        for j, json_file in enumerate(json_files):
            if j % 1 == 0:
                print(f"  Processing file {j+1}/{len(json_files)}: {json_file.name}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    references = json.load(f)
                
                for ref in references:
                    results['total_references'] += 1
                    
                    # Extract reference info
                    ref_arxiv_id = extract_arxiv_id(ref)
                    ref_title = get_reference_title(ref)
                    ref_authors = get_reference_authors(ref)
                    
                    match_found = False
                    match_details = {
                        'source': json_file.stem,
                        'ref_title': ref_title,
                        'ref_arxiv_id': ref_arxiv_id,
                        'ref_authors': ref_authors,
                        'match_type': None,
                        'matched_paper': None,
                        'title_match': False,
                        'author_match': False,
                        'author_score': 0.0
                    }

                    has_title = len(ref_title.split(" ")) > 4

                    def valid_author(a):
                        return len(a) > 2 and "'given'" not in a and "'literal'" not in a 

                    ref_last_names, _ = extract_last_names(ref_authors)
                    has_authors = len(ref_last_names) > 0 and all([valid_author(a) for a in ref_last_names])
                    skip = not has_title or not has_authors

                    # Check arXiv ID first if available
                    """
                    if not skip and ref_arxiv_id and ref_arxiv_id != match_details["source"]:
                        
                        paper = arxiv_db.lookup_by_arxiv_id(ref_arxiv_id)
                        if paper:
                            results['arxiv_id_matches'] += 1
                            match_found = True
                            match_details['match_type'] = 'arxiv_id'
                            match_details['matched_paper'] = {
                                'arxiv_id': paper['arxiv_id'],
                                'title': paper['title'],
                                'authors': paper['authors']
                            }
                            
                            # Check if title and authors match
                            title_score = similarity_ratio(
                                normalize_text(ref_title), 
                                normalize_text(paper['title'])
                            ) 
                            title_similar = title_score >= 0.7
                            
                            author_match, author_score = check_author_match(
                                ref_authors, paper['authors']
                            )
                            
                            match_details['title_match'] = title_similar
                            match_details['title_score'] = title_score
                            match_details['author_match'] = author_match
                            match_details['author_score'] = author_score
                            
                            if title_similar and author_match:
                                results['arxiv_id_correct'] += 1

                            if not title_similar and not author_match:
                                print("potential arxiv ID mismatch")
                                print(match_details)
                    """
                    # If no arXiv ID match, try title matching
                    if not skip and not match_found and ref_title:
                        candidates = arxiv_db.search_by_title(ref_title)
                        if candidates:
                            results['title_matches'] += 1
                            match_found = True
                            
                            # Find best author match among candidates
                            best_paper = None
                            best_score = 0.0
                            
                            for paper in candidates[:5]:  # Check top 5 candidates
                                author_match, author_score = check_author_match(
                                    ref_authors, paper['authors']
                                )
                                if author_score > best_score:
                                    best_score = author_score
                                    best_paper = paper
                            
                            match_details['match_type'] = 'title'
                            match_details['title_match'] = True
                            if best_paper:
                                match_details['matched_paper'] = {
                                    'arxiv_id': best_paper['arxiv_id'],
                                    'title': best_paper['title'],
                                    'authors': best_paper['authors']
                                }
                            else:
                                match_details['matched_paper'] = {
                                    'arxiv_id': candidates[0]['arxiv_id'],
                                    'title': candidates[0]['title'],
                                    'authors': candidates[0]['authors']
                                }
                            
                            if best_paper:
                                match_details['author_match'] = best_score >= 0.5
                                match_details['author_score'] = best_score
                                
                                if match_details['author_match']:
                                    results['title_with_author_matches'] += 1
                                elif len(match_details['matched_paper']["authors"]) <= 10:
                                    print("potential title mismatch")
                                    print(match_details)
                    
                    if not match_found:
                        results['no_matches'] += 1
                        match_details = {
                            'source': str(json_file).split("/")[2][:-5],
                            'ref_title': ref_title
                        }
                        results['details'].append(match_details)
                    else: 
                        # Store detailed results
                        results['details'].append(match_details)
                        
            except json.decoder.JSONDecodeError as e:
                #print(f"Error processing {json_file}: {e}")
                continue
    
    return results

def save_monthly_results(results_by_month):
    """Save results organized by month to months/yymm.json files"""
    os.makedirs('months', exist_ok=True)
    
    max_month = 0
    for yymm, month_results in results_by_month.items():
        month_file = f'months/{yymm}.json'
        
        # Load existing results if file exists
        existing_results = {'details': []}
        if os.path.exists(month_file):
            try:
                with open(month_file, 'r', encoding='utf-8') as f:
                    existing_results = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_results = {'details': []}
        
        # Merge with new results
        existing_results['details'].extend(month_results['details'])
        
        # Update summary statistics
        existing_results.update({
            'total_references': existing_results.get('total_references', 0) + month_results['total_references'],
            'arxiv_id_matches': existing_results.get('arxiv_id_matches', 0) + month_results['arxiv_id_matches'],
            'arxiv_id_correct': existing_results.get('arxiv_id_correct', 0) + month_results['arxiv_id_correct'],
            'title_matches': existing_results.get('title_matches', 0) + month_results['title_matches'],
            'title_with_author_matches': existing_results.get('title_with_author_matches', 0) + month_results['title_with_author_matches'],
            'no_matches': existing_results.get('no_matches', 0) + month_results['no_matches']
        })
        
        # Save updated results
        with open(month_file, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
        
        print(f"Updated results for {yymm} saved to {month_file}")
        max_month = max(int(yymm), max_month)

    return max_month

def main():
    # Get latest processed date
    last_processed_yymm = get_latest_processed_date()
    print(f"Last processed date (YYMM): {last_processed_yymm}")
    
    # Find unprocessed directories
    unprocessed_dirs = find_unprocessed_directories(last_processed_yymm)
    
    if not unprocessed_dirs:
        print("No unprocessed directories found.")
        return
    
    print(f"Found {len(unprocessed_dirs)} unprocessed directories")
    
    # Load arXiv database
    arxiv_db = ArxivDatabase('arxiv_papers.json')
    
    # Process unprocessed reference files
    print("\nProcessing unprocessed reference files...")
    results = process_reference_files(arxiv_db, unprocessed_dirs)
    
    # Organize results by month
    results_by_month = defaultdict(lambda: {
        'total_references': 0,
        'arxiv_id_matches': 0,
        'arxiv_id_correct': 0,
        'title_matches': 0,
        'title_with_author_matches': 0,
        'no_matches': 0,
        'details': []
    })
    
    # Group results by month based on source filename
    for detail in results['details']:
        # Extract date from source (assuming source is like '2506' from '2506.24062')
        source_parts = detail.get('source', '').split('.')
        if source_parts and len(source_parts[0]) == 4 and source_parts[0].isdigit():
            yymm = source_parts[0]
            results_by_month[yymm]['details'].append(detail)
            
            # Update counters
            results_by_month[yymm]['total_references'] += 1
            if detail.get('match_type') == 'arxiv_id':
                results_by_month[yymm]['arxiv_id_matches'] += 1
                if detail.get('title_match') and detail.get('author_match'):
                    results_by_month[yymm]['arxiv_id_correct'] += 1
            elif detail.get('match_type') == 'title':
                results_by_month[yymm]['title_matches'] += 1
                if detail.get('author_match'):
                    results_by_month[yymm]['title_with_author_matches'] += 1
            elif not detail.get('match_type'):
                results_by_month[yymm]['no_matches'] += 1
    
    # Save results by month
    max_month = save_monthly_results(results_by_month)
    save_checkpoint(max_month)
    
    # Print summary
    print("\n" + "="*60)
    print("REFERENCE CHECKING RESULTS")
    print("="*60)
    print(f"Total references processed: {results['total_references']:,}")
    print(f"References with arXiv ID matches: {results['arxiv_id_matches']:,} ({100*results['arxiv_id_matches']/max(1,results['total_references']):.1f}%)")
    print(f"ArXiv ID matches that are correct: {results['arxiv_id_correct']:,} ({100*results['arxiv_id_correct']/max(1,results['arxiv_id_matches']):.1f}% of ID matches)")
    print(f"References matched by title: {results['title_matches']:,} ({100*results['title_matches']/max(1,results['total_references']):.1f}%)")
    print(f"Title matches with author agreement: {results['title_with_author_matches']:,} ({100*results['title_with_author_matches']/max(1,results['title_matches']):.1f}% of title matches)")
    print(f"No matches found: {results['no_matches']:,} ({100*results['no_matches']/max(1,results['total_references']):.1f}%)")
    print(f"\nProcessed {len(results_by_month)} months of data")

if __name__ == "__main__":
    main()
