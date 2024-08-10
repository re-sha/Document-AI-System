from datasets import Dataset, Features, Sequence, ClassLabel, Value, Array2D, Array3D
import fitz
import os

from transformers import pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
def extract_ner_tags(page_text):
    # Implement your NER extraction logic here
    return ["O"] * len(page_text.split())  # Placeholder implementation

def extract_image_from_pdf(pdf_file, page_num):
    ner_results = ner_pipeline(page_text)
    tags = ["O"] * len(page_text.split())
    
    for entity in ner_results:
        start = entity['start']
        end = entity['end']
        entity_text = entity['word']
        entity_type = entity['entity']
        
        # Find the corresponding words in page_text.split()
        for i, word in enumerate(page_text.split()):
            if page_text.find(word, start) == start:
                tags[i] = f"B-{entity_type}"
                for j in range(i+1, len(tags)):
                    if page_text.find(tags[j].split()[0], start) == start:
                        tags[j] = f"I-{entity_type}"
                    else:
                        break
                break
    
    return tags

pdf_dir = "path_to_your_pdf_directory"
pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

features = Features({
    'image': Array3D(dtype="uint8", shape=(3, 224, 224)),
    'words': Sequence(Value("string")),
    'bbox': Array2D(dtype="int64", shape=(None, 4)),
    'ner_tags': Sequence(ClassLabel(names=["O", "B-ENT", "I-ENT"]))
})

images = []
words = []
bboxes = []
ner_tags = []

for pdf_file in pdf_files:
    doc = fitz.open(pdf_file)
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        page_words = page_text.split()
        page_boxes = page.get_text_blocks()
        page_ner_tags = extract_ner_tags(page_text)
        images.append(extract_image_from_pdf(pdf_file, page_num))
        words.append(page_words)
        bboxes.append(page_boxes)
        ner_tags.append(page_ner_tags)

dataset = Dataset.from_dict({
    "image": images,
    "words": words,
    "bbox": bboxes,
    "ner_tags": ner_tags
}, features=features)