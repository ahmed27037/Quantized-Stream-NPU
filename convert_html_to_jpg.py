#!/usr/bin/env python3
"""
Convert HTML diagram files to JPG images using playwright
"""
import os
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Error: playwright not installed. Installing now...")
    os.system(f"{sys.executable} -m pip install playwright")
    os.system(f"{sys.executable} -m playwright install chromium")
    from playwright.sync_api import sync_playwright

def convert_html_to_jpg(html_path, jpg_path, width=1920, height=1080):
    """Convert an HTML file to a JPG image"""
    html_path = Path(html_path).resolve()
    jpg_path = Path(jpg_path).resolve()
    
    # Ensure output directory exists
    jpg_path.parent.mkdir(parents=True, exist_ok=True)
    
    with sync_playwright() as p:
        # Launch browser
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={'width': width, 'height': height})
        
        # Load the HTML file
        file_url = f"file://{html_path}"
        page.goto(file_url)
        
        # Wait for page to fully load
        page.wait_for_load_state('networkidle')
        
        # Take screenshot
        page.screenshot(path=str(jpg_path), type='jpeg', full_page=True)
        
        browser.close()
    
    print(f"Converted: {html_path.name} -> {jpg_path.name}")

def main():
    # Get the diagrams directory
    diagrams_dir = Path('diagrams')
    
    if not diagrams_dir.exists():
        print(f"Error: {diagrams_dir} directory not found")
        sys.exit(1)
    
    # Find all HTML files in diagrams directory
    html_files = list(diagrams_dir.glob('*.html'))
    
    if not html_files:
        print("No HTML files found in diagrams directory")
        sys.exit(1)
    
    print(f"Found {len(html_files)} HTML file(s) to convert...")
    
    # Convert each HTML file to JPG
    for html_file in html_files:
        jpg_file = html_file.with_suffix('.jpg')
        print(f"Converting {html_file.name}...")
        try:
            convert_html_to_jpg(html_file, jpg_file, width=1920, height=2000)
        except Exception as e:
            print(f"Error converting {html_file.name}: {e}")
            sys.exit(1)
    
    print("All conversions completed successfully!")

if __name__ == '__main__':
    main()

