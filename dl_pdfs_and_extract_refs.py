import xml.etree.ElementTree as ET
import os
import shutil
import json
from multiprocessing import Pool


manifest = "arXiv_pdf_manifest.xml"
os.system(f"aws s3 cp s3://arxiv/pdf/arXiv_pdf_manifest.xml . --request-payer requester")

# read and parse the manifest
with open(manifest, 'r') as f:
    manifest = f.read()

# parse the manifest
root = ET.fromstring(manifest)

# get the list of files
files = root.findall('file')
print(len(files))

def load_checkpoint():
    """Load the last processed directory index from checkpoint"""
    checkpoint_file = "last_processing.json"
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
            if 'last_pdf_directory' in data:
                return data.get('last_pdf_directory') +1
    return 2886 # start in 2000

def save_checkpoint(directory_index):
    """Save the last processed directory index to checkpoint"""
    checkpoint_file = "last_processing.json"
    data = {}
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            data = json.load(f)
    
    data['last_pdf_directory'] = int(directory_index)
    
    with open(checkpoint_file, 'w') as f:
        json.dump(data, f)

start = load_checkpoint()
print(f"Starting from directory index: {start}")

def extract_refs(file):
    arxiv_id = file[:-4]
    #os.system(f"anystyle find {dir_name}/{file} > {dir_name}/{arxiv_id}.json")
    os.system(f"timeout 15s pdftotext++ {dir_name}/{file} {dir_name}/{arxiv_id}.txt")
    os.system(f"anystyle find {dir_name}/{arxiv_id}.txt > {dir_name}/{arxiv_id}.json")
    os.system(f"rm {dir_name}/{file} {dir_name}/{arxiv_id}.txt")


for i in range(start, len(files)):
    date = files[i].find('yymm').text
    year = date[:2]

    # ignore the 19XXs
    if int(year) > 50:
        continue

    fn = files[i].find('filename').text.split('/')[-1]
    print(i, fn)
    os.system(f"aws s3 cp s3://arxiv/pdf/{fn} . --request-payer requester")
    
    dir_name = fn.split('_')[2]
    os.system(f"tar -xvf {fn}")
    os.system(f"rm {fn}")
    print(f"extracted {fn}")

    # delete all files in the tar folder that are not .gz
    for file in os.listdir(dir_name):
        if not file.endswith('.pdf'):
            os.remove(os.path.join(dir_name, file))

    # list all files in the tar folder
    pdf_files = sorted(os.listdir(dir_name))
    #for file in pdf_files:
    #    arxiv_id = file[:-4]
    #    os.system(f"anystyle find {dir_name}/{file} > {dir_name}/{arxiv_id}.json")
    #    os.system(f"rm {dir_name}/{file}")
    pool = Pool(128) 
    pool.map(extract_refs, pdf_files)
    pool.close()
    pool.join()

    # rename dir
    os.system(f"mv {dir_name} refs++/{i}_done")
    
    # Save checkpoint after completing each directory
    save_checkpoint(i)
    print(f"Checkpoint saved: completed directory {i}")

