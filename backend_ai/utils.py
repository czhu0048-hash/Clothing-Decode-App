import pandas as pd
from pypdf import PdfReader
from pathlib import Path

def get_context_from_data():
    data_path = Path(__file__).parent / "data"
    all_context = ""

    for file in sorted(data_path.glob("*.*")):
        try:
            if file.suffix == ".csv":
                df = pd.read_csv(file)
                all_context += f"\nFILE {file.name}:\n{df.to_csv(index=False)}\n"
            elif file.suffix in [".xlsx", ".xls"]:
                df = pd.read_excel(file)
                all_context += f"\nFILE {file.name}:\n{df.to_csv(index=False)}\n"
            elif file.suffix == ".pdf":
                reader = PdfReader(file)
                text = "".join([page.extract_text() for page in reader.pages])
                all_context += f"\nFILE {file.name}:\n{text}\n"
        except Exception as e:
            print(f"Skipping {file.name}: {e}")

    if not all_context.strip():
        print("Warning: No data files found in /data folder.")

    return all_context