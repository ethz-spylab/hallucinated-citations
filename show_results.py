#!/usr/bin/env python3
"""
Fast script to plot hallucination rates per month.
Efficiently loads data from months/yymm.json and months/yymm_filtered.json files.
"""

import json
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import re
from collections import Counter, defaultdict
from typing import Dict, List
from unidecode import unidecode

def normalize_text(text):
    """Normalize text by lowercasing and converting to ASCII."""
    if not text:
        return ""
    return unidecode(text.lower())

def extract_last_name(author_text):
    """Extract and normalize last name from author text."""
    if not author_text:
        return ""
    
    normalized = normalize_text(author_text)
    
    # Handle "Last, First" format
    if ',' in normalized:
        return normalized.split(',')[0].strip()
    
    # Handle "First Last" or "F. Last" format - take the last word
    words = normalized.split()
    if words:
        return words[-1].strip()
    
    return normalized

def load_total_references_per_month(months_dir: str = "months") -> Dict[str, int]:
    """Load total reference counts from unfiltered month files or cached stats."""
    print("Loading total reference counts...")
    
    # Find all unfiltered files to determine date range
    unfiltered_files = glob.glob(os.path.join(months_dir, "[0-9][0-9][0-9][0-9].json"))
    if not unfiltered_files:
        print("No unfiltered month files found")
        return {}
    
    # Extract YYMM values and find range
    yymm_values = []
    for file_path in unfiltered_files:
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d{4})\.json$', filename)
        if match:
            yymm_values.append(match.group(1))
    
    if not yymm_values:
        print("No valid YYMM files found")
        return {}
    
    yymm_values.sort()
    yymm_start = yymm_values[0]
    yymm_end = yymm_values[-1]
    print(yymm_start, yymm_end)
    
    # Check for cached stats file
    #stats_file = os.path.join(months_dir, f"stats_{yymm_start}_{yymm_end}.json")
    stats_file = os.path.join(months_dir, f"stats.json")
    
    if os.path.exists(stats_file):
        print(f"Loading cached stats from {stats_file}")
        try:
            with open(stats_file, 'r', encoding='utf-8') as f:
                cached_stats = json.load(f)
            
            total_refs = cached_stats.get('total_references_per_month', {})
            print(f"Loaded cached reference counts for {len(total_refs)} months")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading cached stats: {e}, falling back to loading individual files")

        stats_end = int(cached_stats["date_range"][-4:])
        remaining_files = []
        for file_path in unfiltered_files:
            filename = os.path.basename(file_path)
            match = re.match(r'^(\d{4})\.json$', filename)
            yymm = int(match.group(1))
            if yymm > stats_end:
                remaining_files.append(file_path)

    else:
        # Load individual files if no cache available
        print(f"No cached stats found, loading {len(unfiltered_files)} individual files...")
        total_refs = {}
        remaining_files = unfiltered_files

    print(f"{len(remaining_files)} files to process")
    
    for i, file_path in enumerate(sorted(remaining_files)):
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d{4})\.json$', filename)
        if not match:
            continue
        
        yymm = match.group(1)
        
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(remaining_files)}: {yymm}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Count total references - could be in 'total_references' field or length of 'details'
            if isinstance(data, list):
                total_refs[yymm] = len(data)
            else:
                total_refs[yymm] = len(data["details"])
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
            total_refs[yymm] = 0
    
    # Save cached stats
    print(f"Saving cached stats to {stats_file}")
    try:
        cached_data = {
            'date_range': f"{yymm_start}_{yymm_end}",
            'total_files': len(unfiltered_files),
            'total_references_per_month': total_refs,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(cached_data, f, indent=2, ensure_ascii=False)
        
        print(f"Cached stats saved successfully")
    except Exception as e:
        print(f"Warning: Could not save cached stats: {e}")
    
    print(f"Loaded reference counts for {len(total_refs)} months")
    return total_refs

def load_all_filtered_data(months_dir: str = "months") -> Dict[str, List[dict]]:
    """Load all filtered data into memory."""
    print("Loading all filtered data...")
    
    filtered_data = {}
    filtered_files = glob.glob(os.path.join(months_dir, "[0-9][0-9][0-9][0-9]_filtered.json"))
    
    for i, file_path in enumerate(sorted(filtered_files)):
        filename = os.path.basename(file_path)
        match = re.match(r'^(\d{4})_filtered\.json$', filename)
        if not match:
            continue
        
        yymm = match.group(1)
        
        if i % 10 == 0:
            print(f"  Processing {i+1}/{len(filtered_files)}: {yymm}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            filtered_data[yymm] = data.get('results', [])
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error loading {file_path}: {e}")
            filtered_data[yymm] = []
    
    print(f"Loaded filtered data for {len(filtered_data)} months")
    return filtered_data

def apply_corrections_and_count(filtered_data: Dict[str, List[dict]]) -> Dict[str, Dict[str, int]]:
    """Apply corrections for frequent misclassifications and count results."""
    print("Applying corrections and counting...")
    
    # First pass: count how often each normalized title is classified as HALLUCINATED
    title_hallucination_count = Counter()
    
    for yymm, results in filtered_data.items():
        for result in results:
            if result.get('classification') == 'HALLUCINATED':
                title = result.get('title', '')
                normalized_title = normalize_text(title)
                title_hallucination_count[normalized_title] += 1
    
    # Find frequently misclassified titles (3+ times)
    frequently_misclassified = {title for title, count in title_hallucination_count.items() if count >= 10}
    
    print(f"Found {len(frequently_misclassified)} normalized titles classified as hallucinated 3+ times (will be corrected)")
    
    # Second pass: count with corrections
    monthly_counts = {}
    
    for yymm, results in filtered_data.items():
        counts = {
            'HALLUCINATED': 0,
            'NOT_HALLUCINATED': 0,
            'ERROR': 0,
            'FAILED': 0
        }
        
        for result in results:
            classification = result.get('classification', 'FAILED')
            title = result.get('title', '')
            normalized_title = normalize_text(title)
            
            # Apply correction for frequently misclassified titles
            if classification == 'HALLUCINATED' and normalized_title in frequently_misclassified:
                classification = 'NOT_HALLUCINATED'
            
            if classification in counts:
                counts[classification] += 1
            else:
                counts['FAILED'] += 1
        
        monthly_counts[yymm] = counts
    
    return monthly_counts

def find_author_hallucinations(filtered_data: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Find hallucinated references where author is in arXiv but NOT in the hallucinated reference."""
    print("Finding author hallucinations...")
    
    # First get the correction set
    title_hallucination_count = Counter()
    for yymm, results in filtered_data.items():
        for result in results:
            if result.get('classification') == 'HALLUCINATED':
                title = result.get('title', '')
                normalized_title = normalize_text(title)
                title_hallucination_count[normalized_title] += 1
    
    frequently_misclassified = {title for title, count in title_hallucination_count.items() if count >= 3}
    
    # Group by normalized author names - only count cases where author is in arXiv but NOT in reference
    author_hallucinations = defaultdict(list)
    
    for yymm, results in filtered_data.items():
        for result in results:
            classification = result.get('classification', 'FAILED')
            title = result.get('title', '')
            normalized_title = normalize_text(title)
            
            # Apply correction
            if classification == 'HALLUCINATED' and normalized_title in frequently_misclassified:
                classification = 'NOT_HALLUCINATED'
            
            if classification == 'HALLUCINATED':
                # Get authors from both the reference and the matched arXiv paper
                ref_authors = result.get('ref_authors', [])
                matched_authors = result.get('matched_authors', [])
                
                # Extract last names from reference authors
                ref_last_names = set()
                for ref_author in ref_authors:
                    if isinstance(ref_author, dict):
                        # Handle {"family": "Smith", "given": "John"} format
                        family = ref_author.get('family', '')
                        given = ref_author.get('given', '')
                        full_name = f"{given} {family}".strip()
                        last_name = extract_last_name(full_name)
                        if last_name:
                            ref_last_names.add(last_name)
                    elif isinstance(ref_author, str):
                        last_name = extract_last_name(ref_author)
                        if last_name:
                            ref_last_names.add(last_name)
                
                # Check each matched author (full name) - count if their last name is NOT in reference
                for author in matched_authors:
                    author_last_name = extract_last_name(author)
                    normalized_full_name = normalize_text(author)
                    
                    # Count as hallucination if:
                    # 1. We have the full name from arXiv
                    # 2. But their last name doesn't appear anywhere in the reference
                    if author_last_name and author_last_name not in ref_last_names:
                        author_hallucinations[normalized_full_name].append({
                            'original_author': author,
                            'source': result.get('source', ''),
                            'title': title,
                            'ref_authors': ref_authors,
                            'matched_authors': matched_authors,
                            'month': yymm
                        })
    
    return dict(author_hallucinations)

def create_monthly_dataframe(total_refs: Dict[str, int], monthly_counts: Dict[str, Dict[str, int]]) -> pd.DataFrame:
    """Create a pandas DataFrame with monthly statistics."""
    
    monthly_stats = []
    
    # Get all months that have both total refs and filtered data
    common_months = set(total_refs.keys()).intersection(set(monthly_counts.keys()))
    
    for yymm in sorted(common_months):
        total = total_refs[yymm]
        counts = monthly_counts[yymm]
        
        hallucinated = counts['HALLUCINATED']
        not_hallucinated = counts['NOT_HALLUCINATED']
        errors = counts['ERROR']
        failed = counts['FAILED']
        
        candidate_hallucinations = hallucinated + not_hallucinated + errors + failed
        
        # Convert YYMM to date
        month_date = f"20{yymm[:2]}-{yymm[2:]}"
        
        monthly_stats.append({
            'month': month_date,
            'yymm': yymm,
            'total_references': total,
            'candidate_hallucinations': candidate_hallucinations,
            'true_hallucinations': hallucinated,
            'not_hallucinated': not_hallucinated,
            'errors': errors,
            'failed': failed,
            'hallucination_fraction': hallucinated / total if total > 0 else 0,
            'candidate_fraction': candidate_hallucinations / total if total > 0 else 0,
            'true_positive_rate': hallucinated / candidate_hallucinations if candidate_hallucinations > 0 else 0
        })
    
    return pd.DataFrame(monthly_stats)

def plot_hallucination_trends(df: pd.DataFrame, save_path: str = "hallucination_analysis.png"):
    """Create a simple plot of hallucination trends."""
    
    # Convert to datetime and sort
    df['month_cleaned'] = df['month'].str.replace('-00', '-01')
    df['date'] = pd.to_datetime(df['month_cleaned'], format='%Y-%m')
    df = df.sort_values('date')
    
    # Create plot
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    
    # Plot hallucination fraction
    ax.plot(df['date'], df['hallucination_fraction'] * 100, 'b-', marker='o', linewidth=2, markersize=3)
    
    # Add important dates
    chatgpt_launch = pd.to_datetime('2022-11-30')
    dr_launch = pd.to_datetime('2025-02-03')
    
    ax.axvline(x=chatgpt_launch, color='red', linestyle='--', linewidth=2, alpha=0.7, label='ChatGPT Launch')
    ax.axvline(x=dr_launch, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Deep Research Launch')
    
    ax.set_title('Hallucination Rate Over Time', fontsize=14)
    ax.set_xlabel('Month', fontsize=12)
    ax.set_ylabel('Hallucinations / Total References (%)', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='x', rotation=45)
    
    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_minor_locator(mdates.MonthLocator((1, 7)))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()

def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    
    total_refs = df['total_references'].sum()
    total_candidates = df['candidate_hallucinations'].sum()
    total_hallucinated = df['true_hallucinations'].sum()
    total_not_hallucinated = df['not_hallucinated'].sum()
    total_errors = df['errors'].sum()
    total_failed = df['failed'].sum()
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Total references: {total_refs:,}")
    print(f"Candidate hallucinations: {total_candidates:,}")
    print(f"True hallucinations: {total_hallucinated:,}")
    print(f"Not hallucinated: {total_not_hallucinated:,}")
    print(f"Errors: {total_errors:,}")
    print(f"Failed: {total_failed:,}")
    print()
    print(f"Overall hallucination rate: {total_hallucinated/total_refs*100:.3f}%")
    print(f"Overall candidate rate: {total_candidates/total_refs*100:.3f}%")
    if total_candidates > 0:
        print(f"True positive rate: {total_hallucinated/total_candidates*100:.1f}%")
    
    print(f"\nMonthly averages:")
    print(f"  Hallucination rate: {df['hallucination_fraction'].mean()*100:.3f}% ± {df['hallucination_fraction'].std()*100:.3f}%")
    
    # Show some example months
    print(f"\nHighest hallucination rates:")
    top_months = df.nlargest(5, 'hallucination_fraction')[['month', 'total_references', 'true_hallucinations', 'hallucination_fraction']]
    for _, row in top_months.iterrows():
        print(f"  {row['month']}: {row['hallucination_fraction']*100:.3f}% ({row['true_hallucinations']}/{row['total_references']})")

def find_papers_with_most_hallucinated_refs(filtered_data: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Find arXiv papers that contain the most hallucinated references in their bibliography."""
    print("Finding papers with most hallucinated references in their bibliography...")
    
    # First get the correction set
    title_hallucination_count = Counter()
    for yymm, results in filtered_data.items():
        for result in results:
            if result.get('classification') == 'HALLUCINATED':
                title = result.get('title', '')
                normalized_title = normalize_text(title)
                title_hallucination_count[normalized_title] += 1
    
    frequently_misclassified = {title for title, count in title_hallucination_count.items() if count >= 10}
    
    # Group by the source paper (the paper that contains the hallucinated reference)
    source_paper_hallucinations = defaultdict(list)
    
    for yymm, results in filtered_data.items():
        for result in results:
            classification = result.get('classification', 'FAILED')
            title = result.get('title', '')
            normalized_title = normalize_text(title)
            
            # Apply correction
            if classification == 'HALLUCINATED' and normalized_title in frequently_misclassified:
                classification = 'NOT_HALLUCINATED'
            
            if classification == 'HALLUCINATED':
                source_paper = result.get('source', '')
                source_paper_hallucinations[source_paper].append({
                    'hallucinated_title': title,
                    'ref_authors': result.get('ref_authors', []),
                    'matched_authors': result.get('matched_authors', []),
                    'month': yymm
                })
    
    return dict(source_paper_hallucinations)

def print_top_papers_with_hallucinated_refs(source_paper_hallucinations: Dict[str, List[dict]]):
    """Print top 10 arXiv papers that contain the most hallucinated references."""
    
    # Get top 10 by count
    top_papers = sorted(source_paper_hallucinations.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    print("\n" + "="*80)
    print("TOP 10 ARXIV PAPERS WITH MOST HALLUCINATED REFERENCES IN THEIR BIBLIOGRAPHY")
    print("="*80)
    
    for i, (source_paper, hallucinations) in enumerate(top_papers):
        month_date = f"20{hallucinations[0]['month'][:2]}-{hallucinations[0]['month'][2:]}"
        print(f"\n{i+1:2d}. {source_paper} [{month_date}]")
        print(f"     Contains {len(hallucinations)} hallucinated references")
        print("     Hallucinated references:")
        
        for j, hall in enumerate(hallucinations):
            # Format reference authors
            ref_authors_str = []
            for ref_author in hall['ref_authors']:
                if isinstance(ref_author, dict):
                    family = ref_author.get('family', '')
                    given = ref_author.get('given', '')
                    ref_authors_str.append(f"{given} {family}".strip())
                elif isinstance(ref_author, str):
                    ref_authors_str.append(ref_author)
            
            ref_authors_display = "; ".join(ref_authors_str) if ref_authors_str else "No authors"
            matched_authors_display = "; ".join(hall['matched_authors']) if hall['matched_authors'] else "No authors"
            
            print(f"       {j+1:2d}. {hall['hallucinated_title'][:60]}{'...' if len(hall['hallucinated_title']) > 60 else ''}")
            print(f"           Hallucinated authors: {ref_authors_display[:70]}{'...' if len(ref_authors_display) > 70 else ''}")
            print(f"           Real arXiv authors: {matched_authors_display[:70]}{'...' if len(matched_authors_display) > 70 else ''}")
            print()

def print_author_results(author_hallucinations: Dict[str, List[dict]]):
    """Print results for top 10 authors plus specific requested authors."""
    
    # Get top 10 by count
    top_authors = sorted(author_hallucinations.items(), key=lambda x: len(x[1]), reverse=True)[:10]
    
    # Add specific requested authors if not already in top 10
    requested_authors = ['tramer, florian', 'schmidhuber, jurgen']
    top_author_names = {name for name, _ in top_authors}
    
    for requested in requested_authors:
        if requested not in top_author_names:
            if requested in author_hallucinations:
                top_authors.append((requested, author_hallucinations[requested]))
    
    print("\n" + "="*80)
    print("AUTHORS MISSING FROM HALLUCINATED REFERENCES")
    print("="*80)
    print("(Cases where author is in the real arXiv paper but NOT in the hallucinated reference)")
    
    for i, (normalized_author, hallucinations) in enumerate(top_authors):
        if i < 10:
            print(f"\n{i+1:2d}. {hallucinations[0]['original_author']} ({len(hallucinations)} times missing from hallucinated references)")
        else:
            print(f"\nRequested: {hallucinations[0]['original_author']} ({len(hallucinations)} times missing from hallucinated references)")
        
        print("     All cases where this author was omitted:")
        for j, hall in enumerate(hallucinations):
            month_date = f"20{hall['month'][:2]}-{hall['month'][2:]}"
            
            # Format reference authors
            ref_authors_str = []
            for ref_author in hall['ref_authors']:
                if isinstance(ref_author, dict):
                    family = ref_author.get('family', '')
                    given = ref_author.get('given', '')
                    ref_authors_str.append(f"{given} {family}".strip())
                elif isinstance(ref_author, str):
                    ref_authors_str.append(ref_author)
            
            ref_authors_display = "; ".join(ref_authors_str) if ref_authors_str else "No authors"
            matched_authors_display = "; ".join(hall['matched_authors']) if hall['matched_authors'] else "No authors"
            
            print(f"       {j+1:2d}. [{month_date}] {hall['source']}:")
            print(f"           Title: {hall['title'][:80]}{'...' if len(hall['title']) > 80 else ''}")
            print(f"           Reference authors: {ref_authors_display[:80]}{'...' if len(ref_authors_display) > 80 else ''}")
            print(f"           ArXiv authors: {matched_authors_display[:80]}{'...' if len(matched_authors_display) > 80 else ''}")
            print()

def main():
    months_dir = "months"
    
    if not os.path.exists(months_dir):
        print(f"Error: Directory {months_dir} not found")
        return
    
    # Step 1: Load total reference counts
    total_refs = load_total_references_per_month(months_dir)
    
    # Step 2: Load all filtered data
    filtered_data = load_all_filtered_data(months_dir)
    
    # Step 3: Apply corrections and count
    monthly_counts = apply_corrections_and_count(filtered_data)
    
    # Step 4: Find author hallucinations
    author_hallucinations = find_author_hallucinations(filtered_data)
    
    # Step 5: Find papers with most hallucinated references
    source_paper_hallucinations = find_papers_with_most_hallucinated_refs(filtered_data)
    
    # Step 6: Create DataFrame
    df = create_monthly_dataframe(total_refs, monthly_counts)
    
    if df.empty:
        print("No data to plot")
        return
    
    print(f"Created dataset with {len(df)} months")
    
    # Step 7: Print results
    print_summary_stats(df)
    print_top_papers_with_hallucinated_refs(source_paper_hallucinations)
    print_author_results(author_hallucinations)
    plot_hallucination_trends(df)

if __name__ == "__main__":
    main()
