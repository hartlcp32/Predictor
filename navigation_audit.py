#!/usr/bin/env python3
"""
Navigation and link audit for Stock Predictor pages
"""

import os
import re
from pathlib import Path

def extract_nav_links(filepath):
    """Extract navigation links from HTML file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Find all nav links
        nav_pattern = r'<a href="([^"]+)"[^>]*class="nav-link[^"]*"[^>]*>([^<]+)</a>'
        links = re.findall(nav_pattern, content)

        # Also check for other important links
        all_href_pattern = r'href="([^"]+\.html)"'
        all_hrefs = set(re.findall(all_href_pattern, content))

        return links, all_hrefs
    except Exception as e:
        return [], set()

def check_file_exists(base_path, link):
    """Check if linked file exists"""
    # Handle relative paths
    if link.startswith('../'):
        # Go up one directory
        check_path = base_path.parent / link[3:]
    elif link.startswith('./'):
        check_path = base_path.parent / link[2:]
    elif '/' not in link and base_path.parent.name == 'docs':
        # Relative link from docs folder
        check_path = base_path.parent / link
    else:
        # Assume relative to project root
        check_path = Path('C:/Projects/Predictor') / link

    return check_path.exists()

def audit_navigation():
    """Perform comprehensive navigation audit"""
    project_root = Path('C:/Projects/Predictor')

    # Get all HTML files
    html_files = list(project_root.glob('*.html')) + list((project_root / 'docs').glob('*.html'))

    report = {
        'files_analyzed': [],
        'navigation_links': {},
        'broken_links': [],
        'missing_from_nav': [],
        'inconsistent_nav': []
    }

    # Analyze each file
    for html_file in html_files:
        relative_path = html_file.relative_to(project_root)
        nav_links, all_hrefs = extract_nav_links(html_file)

        report['files_analyzed'].append(str(relative_path))
        report['navigation_links'][str(relative_path)] = nav_links

        # Check for broken links
        for href, _ in nav_links:
            if not check_file_exists(html_file, href):
                report['broken_links'].append({
                    'file': str(relative_path),
                    'broken_link': href
                })

        # Check all hrefs for broken links
        for href in all_hrefs:
            if not check_file_exists(html_file, href):
                report['broken_links'].append({
                    'file': str(relative_path),
                    'broken_link': href
                })

    # Check for navigation consistency
    nav_counts = {}
    for file, links in report['navigation_links'].items():
        nav_count = len(links)
        if nav_count > 0:
            nav_counts[file] = nav_count

    if nav_counts:
        max_nav = max(nav_counts.values())
        for file, count in nav_counts.items():
            if count < max_nav - 1:  # Allow for some variation
                report['inconsistent_nav'].append({
                    'file': file,
                    'nav_count': count,
                    'expected': max_nav
                })

    # List of pages that should be in navigation
    expected_pages = [
        'HOME', 'PREDICTIONS', 'ANALYTICS', 'STRATEGIES',
        'ENTRY_TRACKER', 'LIVE_POSITIONS', 'TRADE_HISTORY',
        'HISTORIC', 'PERFORMANCE', 'ABOUT', 'MONTHLY'
    ]

    # Check which pages are missing from main navigation
    main_nav = report['navigation_links'].get('index.html', [])
    main_nav_texts = [text.upper() for _, text in main_nav]

    for expected in expected_pages:
        if expected not in main_nav_texts and not any(expected in text.upper() for text in main_nav_texts):
            report['missing_from_nav'].append(expected)

    return report

def print_report(report):
    """Print formatted audit report"""
    print("=" * 60)
    print("NAVIGATION AND LINK AUDIT REPORT")
    print("=" * 60)

    print(f"\n📁 Files Analyzed: {len(report['files_analyzed'])}")
    for file in sorted(report['files_analyzed']):
        nav_count = len(report['navigation_links'].get(file, []))
        print(f"  - {file}: {nav_count} nav links")

    print(f"\n❌ Broken Links: {len(report['broken_links'])}")
    if report['broken_links']:
        for item in report['broken_links']:
            print(f"  - {item['file']}: {item['broken_link']}")
    else:
        print("  ✅ No broken links found")

    print(f"\n⚠️ Missing from Main Navigation: {len(report['missing_from_nav'])}")
    if report['missing_from_nav']:
        for missing in report['missing_from_nav']:
            print(f"  - {missing}")

    print(f"\n🔄 Inconsistent Navigation: {len(report['inconsistent_nav'])}")
    if report['inconsistent_nav']:
        for item in report['inconsistent_nav']:
            print(f"  - {item['file']}: {item['nav_count']} links (expected ~{item['expected']})")

    # Additional checks
    print("\n📊 Additional Findings:")

    # Check for duplicate navigation patterns
    nav_patterns = {}
    for file, links in report['navigation_links'].items():
        pattern = tuple(text for _, text in links)
        if pattern:
            if pattern not in nav_patterns:
                nav_patterns[pattern] = []
            nav_patterns[pattern].append(file)

    print(f"\n  Navigation Patterns Found: {len(nav_patterns)}")
    if len(nav_patterns) > 2:
        print("  ⚠️ Multiple different navigation patterns detected")
        for i, (pattern, files) in enumerate(nav_patterns.items(), 1):
            print(f"\n  Pattern {i} ({len(files)} files):")
            print(f"    Links: {' | '.join(pattern[:5])}{'...' if len(pattern) > 5 else ''}")
            print(f"    Files: {', '.join(files[:3])}{'...' if len(files) > 3 else ''}")

if __name__ == "__main__":
    report = audit_navigation()
    print_report(report)

    # Write detailed report to file
    with open('navigation_audit_report.txt', 'w') as f:
        import json
        json.dump(report, f, indent=2)

    print("\n📝 Detailed report saved to navigation_audit_report.txt")