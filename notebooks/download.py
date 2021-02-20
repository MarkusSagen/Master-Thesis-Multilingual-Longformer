import argparse
import os
import wget
import zipfile


parser = argparse.ArgumentParser(description="Download files with wget")

parser.add_argument("--url", type=str, required=True, default=None, help="Url for the file to download, just like 'wget'")
parser.add_argument("--dest", type=str, required=False, default=None, help="Path where to store downloaded file. If no arg passed, default is current location")
parser.add_argument("--extract", type=bool, required=False, default=False, help="Define if downloaded file should be extracted (True/False)")
parser.add_argument("--delete", type=bool, required=False, default=False, help="Delete original file after download")


args = parser.parse_args()


# Download file
try:
    out_dir = args.dest
    filename = wget.download(args.url, out=out_dir)
    filepath = filename
except:
    print('Incorrect url...')

# Get full file path
if out_dir is not None:
    filepath = out_dir + filename if out_dir[-1] == '/' else out_dir + "/" + filename
    
# Extract zip
if args.extract:
    try:
        with zipfile.ZipFile(filepath, 'r') as f:
            f.extractall(out_dir)
    except:
        print('Tried to unpack zipfile, but failed \nAre you sure this is a .zip file?')
        
# Delete source file if argument supplied
if args.delete:
    if os.path.exists(filepath):
        os.remove(filepath)