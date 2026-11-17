with open('docs/v5/research_story_src.html', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f, 1):
        if i >= 19840:
            if '<div class="chapter"' in line:
                print(f"Chapter: Line {i}: {line.strip()[:100]}")
            elif '<div class="ctab-panel"' in line:
                print(f"  Panel: Line {i}: {line.strip()[:100]}")
            elif '<button class="ctab-btn"' in line:
                print(f"  Button: Line {i}: {line.strip()[:100]}")
