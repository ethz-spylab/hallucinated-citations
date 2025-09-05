#!/usr/bin/env python3
"""
Script to analyze hallucinated references using Claude Sonnet 4 Batch API.
Reads from monthly_reference_stats.json and checks each candidate hallucinated reference.
"""

import json
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from anthropic.types.messages import MessageBatch
import os
import sys
import uuid
import re
import traceback


@dataclass
class Reference:
    source: str
    ref_title: str
    ref_authors: List[Dict[str, str]]
    matched_authors: List[str]

class HallucinationChecker:
    def __init__(self, api_key: Optional[str] = None):
        self.client = anthropic.Anthropic(
            api_key=api_key or os.getenv('ANTHROPIC_API_KEY')
        )
        self.logger = self._setup_logger()
        
    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _create_prompt(self, reference: Reference) -> str:
        """Create the prompt for Claude analysis."""
        ref_authors_str = str(reference.ref_authors)
        matched_authors_str = str(reference.matched_authors)
        
        prompt = f"""I have parsed a reference from a paper and I want to figure out if it was hallucinated by an LLM or not.
I found an arxiv paper with that title, but the author lists don't quite match.
I want to figure out why that is.

Here are the possible cases:
- the parsing somehow failed, and so some names are merged together or there's some stuff in there that isn't a name. In such a case output ERROR
- due to a parsing error, the parsed authors got cut, truncated or merged with another paper, and only correspond (roughly) to the first or last authors of the arxiv paper. In such a case output ERROR
- there are author discrepancies, but these are minor enough that they can likely be explained by human error (e.g., one name misspelled or slightly incorrect). In such a case output NOT_HALLUCINATED
- the author lists differ by 1 or 2 additions or removals of authors. This could be due to diverging arxiv versions. But as long as the first and last authors match, even if the added/removed authors are arbitrary, output NOT_HALLUCINATED
- the title of the article is generic enough that there could likely be multiple articles, or a different book with that title and different authors. In such a case, always output NOT_HALLUCINATED

Otherwise, if it seems clear that the author list is mostly wrong, output HALLUCINATED. One exception to this is if the paper is clearly written by a huge collaboration. Then, sometimes the reference lists individual authors while the arxiv version just lists the name of the collaboration, or vice-versa. In such a case, output NOT_HALLUCINATED.
Always output your final response as the last word of your response.

Title: {reference.ref_title}
Parsed authors from reference: {ref_authors_str}
Authors found on arxiv: {matched_authors_str}"""

        return prompt

    def _parse_response(self, response_text: str) -> Optional[str]:
        """Parse Claude's response to extract the final classification."""
        response_text = response_text.strip()
        
        # Look for the final word/classification
        words = response_text.split()
        if not words:
            return None
            
        last_word = words[-1].upper().strip('.,!?')
        
        valid_responses = {'ERROR', 'NOT_HALLUCINATED', 'HALLUCINATED'}
        if last_word in valid_responses:
            return last_word
        
        return None

    def create_batch_requests(self, references: List[Reference]) -> List[Request]:
        """Create batch requests for all references."""
        batch_requests = []
        
        for i, reference in enumerate(references):
            prompt = self._create_prompt(reference)
            
            request = Request(
                custom_id=f"ref_{i}_{reference.source.replace('.', '_')}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1000,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            )
            batch_requests.append(request)
        
        return batch_requests

    def submit_batch(self, requests: List[Request]) -> dict:
        """Submit batch requests and return batch ID."""
        self.logger.info(f"Submitting batch with {len(requests)} requests")

        message_batch = self.client.messages.batches.create(requests=requests)
        return message_batch

    def wait_for_batch_completion(self, batch_info: dict, max_wait_time: int = 3600) -> bool:
        """Wait for batch to complete, checking every 30 seconds."""
        batch_id = batch_info.id
        self.logger.info(f"Waiting for batch {batch_id} to complete...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            batch = self.client.messages.batches.retrieve(batch_id)
            status = batch.processing_status
            
            self.logger.info(f"Batch status: {status}")
            
            if status == "ended":
                return True
            elif status in ["failed", "expired", "cancelled"]:
                self.logger.error(f"Batch failed with status: {status}")
                return False
            else:
                self.logger.info(f"Batch status: {status}")
            
            time.sleep(30)  # Check every 30 seconds
        
        self.logger.error(f"Batch timed out after {max_wait_time} seconds")
        return False

    def process_batch_results(self, batch_info: dict, references: List[Reference]) -> List[Dict]:
        """Process the results from a completed batch."""
        batch_id = batch_info.id
        
        all_results = {}
        for result in self.client.messages.batches.results(batch_id):
            all_results[result.custom_id] = result.result

        # Match results back to references
        processed_results = []
        for i, reference in enumerate(references):
            custom_id = f"ref_{i}_{reference.source.replace('.', '_')}"
            
            if custom_id in all_results:
                batch_result = all_results[custom_id]
                
                if batch_result.type == "succeeded":
                    response_text = batch_result.message.content[0].text
                    classification = self._parse_response(response_text)
                        
                    result = {
                        'source': reference.source,
                        'title': reference.ref_title,
                        'ref_authors': reference.ref_authors,
                        'matched_authors': reference.matched_authors,
                        'classification': classification,
                        'response': response_text,
                    }
                else:
                    result = {
                        'source': reference.source,
                        'title': reference.ref_title,
                        'ref_authors': reference.ref_authors,
                        'matched_authors': reference.matched_authors,
                        'classification': None,
                        'response': batch_result.type,
                        'error': batch_result.error.message
                    }
            else:
                result = {
                    'source': reference.source,
                    'title': reference.ref_title,
                    'ref_authors': reference.ref_authors,
                    'matched_authors': reference.matched_authors,
                    'classification': None,
                    'response': "Result not found in batch output",
                }
            
            processed_results.append(result)
        
        return processed_results

    def load_references_from_month_file(self, json_file: str) -> List[Reference]:
        """Load references from a single month JSON file."""
        self.logger.info(f"Loading references from {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"File not found: {json_file}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return []
        
        references = []
        
        # Extract potentially hallucinated references from the details
        details = data.get('details', [])
        for detail in details:
            if self.is_potentially_hallucinated(detail):
                ref = Reference(
                    source=detail.get('source', ''),
                    ref_title=detail.get('ref_title', ''),
                    ref_authors=detail.get('ref_authors', []),
                    matched_authors=detail.get('matched_paper', {}).get('authors', [])
                )
                references.append(ref)
        
        self.logger.info(f"Found {len(references)} potentially hallucinated references")
        return references
    
    def is_collaboration_paper(self, authors: List[Any]) -> bool:
        """Check if paper contains collaboration keyword in author names"""
        if not authors:
            return False
        
        collaboration_keywords = ['collaboration', 'consortium', 'group', 'team']
        
        for author in authors:
            if isinstance(author, dict):
                author_text = f"{author.get('family', '')} {author.get('given', '')}".lower()
            elif isinstance(author, str):
                author_text = author.lower()
            else:
                continue
                
            if any(keyword in author_text for keyword in collaboration_keywords):
                return True
        
        return False
    
    def is_potentially_hallucinated(self, entry: Dict[str, Any]) -> bool:
        """Check if a reference is potentially hallucinated based on criteria"""
        if entry.get('match_type', '') != 'title':
            return False
        
        if entry.get('author_match', True):  # If author_match is True or missing, not hallucinated
            return False
        
        # Check reference authors
        ref_authors = entry.get('ref_authors', [])
        if len(ref_authors) >= 10 or self.is_collaboration_paper(ref_authors):
            return False
        
        # Check matched paper authors
        matched_paper = entry.get('matched_paper', {})
        matched_authors = matched_paper.get('authors', [])
        if len(matched_authors) >= 10 or self.is_collaboration_paper(matched_authors):
            return False
        
        return True
    
    def load_references(self, json_file: str) -> List[Reference]:
        """Load references from the JSON file (legacy method for compatibility)."""
        self.logger.info(f"Loading references from {json_file}")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            self.logger.error(f"File not found: {json_file}")
            return []
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {e}")
            return []
        
        references = []
        total_refs = 0
        
        for month_key, month_data in data.items():
            if 'hallucinated_references' in month_data:
                for ref_data in month_data['hallucinated_references']:
                    try:
                        ref = Reference(
                            source=ref_data.get('source', ''),
                            ref_title=ref_data.get('ref_title', ''),
                            ref_authors=ref_data.get('ref_authors', []),
                            matched_authors=ref_data.get('matched_authors', [])
                        )
                        references.append(ref)
                        total_refs += 1
                    except Exception as e:
                        self.logger.warning(f"Error parsing reference: {e}")
        
        self.logger.info(f"Loaded {total_refs} references from {len(data)} months")
        return references

    def process_month_references(self, json_file: str, output_file: str = None, max_wait_time: int = 3600, batch_info: dict = None) -> Dict[str, int]:
        """Process all references using batch API and return summary statistics."""
        references = self.load_references(json_file)

        if not references:
            self.logger.error("No references loaded")
            return {}

        if batch_info is None:
           
            self.logger.info(f"Processing {len(references)} references using batch API")
            
            # Create batch requests
            batch_requests = self.create_batch_requests(references)
            
            # Submit batch
            try:
                batch_info = self.submit_batch(batch_requests)
            except Exception as e:
                self.logger.error(f"Failed to submit batch: {e}")
                print(traceback.format_exc())
                return {'FAILED': len(references)}
        
        # Wait for completion
        if not self.wait_for_batch_completion(batch_info, max_wait_time):
            self.logger.error("Batch processing failed or timed out")
            return {'FAILED': len(references)}
        
        # Process results
        try:
            processed_results = self.process_batch_results(batch_info, references)
        except Exception as e:
            self.logger.error(f"Failed to process batch results: {e}")
            return {'FAILED': len(references)}
        
        # Calculate summary statistics
        results = {
            'ERROR': 0,
            'NOT_HALLUCINATED': 0,
            'HALLUCINATED': 0,
            'FAILED': 0
        }
        
        for result in processed_results:
            classification = result.get('classification')
            if classification:
                results[classification] += 1
            else:
                results['FAILED'] += 1
        
        # Save results if output file specified
        if output_file:
            self.logger.info(f"Saving results to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': results,
                    'processed_at': time.time(),
                    'total_processed': len(references),
                    'batch_id':batch_info.id,
                    'results': processed_results
                }, f, indent=2, ensure_ascii=False)
        
        return results

def find_next_yymm_to_process() -> Optional[str]:
    """Find the next YYMM that needs processing based on existing files"""
    months_dir = "months"
    if not os.path.exists(months_dir):
        return None
    
    # Get all YYMM.json files
    month_files = []
    for filename in os.listdir(months_dir):
        if re.match(r'^\d{4}\.json$', filename):
            yymm = filename[:-5]  # Remove .json extension
            month_files.append(yymm)
    
    if not month_files:
        return None
    
    month_files.sort()
    
    # Check which ones already have _filtered.json versions
    for yymm in month_files:
        filtered_file = os.path.join(months_dir, f"{yymm}_filtered.json")
        if not os.path.exists(filtered_file):
            return yymm
    
    return None

def main():
    max_wait_time = 7200
    
    # Check for API key
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        print("Please set it with: export ANTHROPIC_API_KEY=your_api_key_here")
        sys.exit(1)
    
    # Find next YYMM to process
    next_yymm = find_next_yymm_to_process()
    if not next_yymm:
        print("No unprocessed months found or months directory doesn't exist")
        return

    input_file = f"months/{next_yymm}.json"
    output_file = f"months/{next_yymm}_filtered.json"
    
    print(f"Processing month: {next_yymm}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    checker = HallucinationChecker()
    
    # Check if we should resume from a batch ID
    if len(sys.argv) > 1:
        #batch_info = {"id": sys.argv[1]}
        batch_info = MessageBatch(sys.argv[1])
    else:
        batch_info = None

    try:
        # Load references from the month file
        references = checker.load_references_from_month_file(input_file)
        
        if not references:
            print(f"No potentially hallucinated references found in {input_file}")
            # Create empty filtered file to mark as processed
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': {'ERROR': 0, 'NOT_HALLUCINATED': 0, 'HALLUCINATED': 0, 'FAILED': 0},
                    'processed_at': time.time(),
                    'total_processed': 0,
                    'month': next_yymm,
                    'results': []
                }, f, indent=2, ensure_ascii=False)
            return
        
        print(f"Found {len(references)} potentially hallucinated references")
        print(f"Processing references using batch API...")
        print(f"Maximum wait time: {max_wait_time} seconds")
        
        # Process references
        if batch_info is None:
            # Create batch requests
            batch_requests = checker.create_batch_requests(references)
            
            # Submit batch
            try:
                batch_info = checker.submit_batch(batch_requests)
                print(f"Batch submitted with ID: {batch_info.id}")
            except Exception as e:
                print(f"Failed to submit batch: {e}")
                print(traceback.format_exc())
                return
        else:
            print(f"Resuming from batch ID: {batch_info.id}")
        
        # Wait for completion
        if not checker.wait_for_batch_completion(batch_info, max_wait_time):
            print("Batch processing failed or timed out")
            print(f"You can resume later with: python {sys.argv[0]} {batch_info.id}")
            return
        
        # Process results
        try:
            processed_results = checker.process_batch_results(batch_info, references)
        except Exception as e:
            print(f"Failed to process batch results: {e}")
            return
        
        # Calculate summary statistics
        results = {
            'ERROR': 0,
            'NOT_HALLUCINATED': 0,
            'HALLUCINATED': 0,
            'FAILED': 0
        }
        
        for result in processed_results:
            classification = result.get('classification')
            if classification:
                results[classification] += 1
            else:
                results['FAILED'] += 1
        
        # Save results
        print(f"Saving results to {output_file}")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': results,
                'processed_at': time.time(),
                'total_processed': len(references),
                'batch_id': batch_info.id,
                'month': next_yymm,
                'results': processed_results
            }, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*50)
        print(f"SUMMARY RESULTS FOR {next_yymm}:")
        print("="*50)
        for category, count in results.items():
            print(f"{category:15}: {count}")
        
        total = sum(results.values())
        print(f"{'TOTAL':15}: {total}")
        
        if total > 0:
            print("\nPERCENTAGES:")
            for category, count in results.items():
                percentage = (count / total) * 100
                print(f"{category:15}: {percentage:.1f}%")
    
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
        if 'batch_info' in locals() and batch_info:
            print(f"You can resume later with: python {sys.argv[0]} {batch_info.id}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        print(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
