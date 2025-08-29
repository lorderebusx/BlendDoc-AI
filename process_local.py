import os
from bs4 import BeautifulSoup

def processLocalFiles(rootFolder, outputFile):
    with open(outputFile, 'w', encoding='utf-8') as f:
        f.write("")

    fileCount = 0
    warningCount = 0
    for dirpath, _, filenames in os.walk(rootFolder):
        for filename in filenames:
            if filename.endswith('.html'):
                fullPath = os.path.join(dirpath, filename)

                try:
                    with open(fullPath, 'r', encoding='utf-8') as f:
                        content = f.read()

                    soup = BeautifulSoup(content, 'html.parser')
                    mainContent = soup.find(id='main-content')

                    if not mainContent:
                        mainContent = soup.find(attrs={'role': 'main'})

                    if mainContent:
                        print(f"Processing: {filename}")
                        text = mainContent.get_text(separator=' ', strip=True)
                        
                        with open(outputFile, 'a', encoding='utf-8') as out:
                            out.write(text + '\n\n')
                        fileCount += 1
                    else:
                        print(f"  -> WARNING: Could not find content in {filename}")
                        warningCount += 1

                except Exception as e:
                    print(f"  -> ERROR processing file {filename}: {e}")

    print("\n--- Processing complete ---")
    print(f"Successfully processed {fileCount} files.")
    if warningCount > 0:
        print(f"Encountered {warningCount} files where content could not be found.")
    print(f"All extracted text has been combined into '{outputFile}'")

if __name__ == "__main__":
    manualRootFolder = "E:/BlendDoc-AI/blender_manual_v450_en"
    outputFilename = "blender_docs.txt"
    processLocalFiles(manualRootFolder, outputFilename)