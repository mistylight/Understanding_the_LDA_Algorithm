"""
Preprocess the papers.csv from Kaggle: https://www.kaggle.com/benhamner/nips-papers/version/2?select=papers.csv
Input: `paperes.csv` with columns [id, year, title, event_type, pdf_name, abstract, paper_text]
Output: paper_text (without newlines) in `papers_raw.txt`, one paper per line.
"""
import csv
with open("papers.csv") as csvfile, open("papers_raw.txt", "w") as fout:
    # clean up null bytes: https://stackoverflow.com/questions/4166070/python-csv-error-line-contains-null-byte
    reader = csv.reader((line.replace('\0', '') for line in csvfile), delimiter=",")
    # only keep the paper text in the last column
    for i, row in enumerate(reader):
        if i == 0: continue
        assert len(row) == 7
        text = row[-1].replace("\n", " ")
        fout.write(text + "\n")
