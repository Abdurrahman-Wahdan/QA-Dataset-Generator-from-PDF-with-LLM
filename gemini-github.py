import google.generativeai as genai
import PyPDF2
import csv
import os
import re
from tqdm import tqdm
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Configure the Google GenAI API
genai.configure(api_key="")


def load_pdf(file_path):
    if not os.path.isfile(file_path):
        raise ValueError(f"File path {file_path} is not a valid file.")

    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text()
    return text


def split_text_into_chunks(text, max_words=1024, overlap=128):
    # Split text into words while keeping newlines
    words = re.findall(r'\S+|\n', text)
    chunks = []
    start = 0

    while start < len(words):
        end = start + max_words
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += max_words - overlap

    return chunks


def extract_questions(text):
    # Define the regex pattern to match [s]... [e]
    pattern = r'\[s\](.*?)\[s\]'

    # Find all matches in the text
    matches = re.findall(pattern, text, re.DOTALL)

    # Strip leading and trailing whitespace from each match
    stripped_matches = [match.strip() for match in matches]

    return stripped_matches


def extract_qa(text):
    # Define the regex patterns for questions and answers
    q_pattern = r'\[s\](.*?)\[s\]'
    a_pattern = r'\[c\](.*?)\[c\]'

    # Find all matches in the text
    questions = re.findall(q_pattern, text, re.DOTALL)
    answers = re.findall(a_pattern, text, re.DOTALL)

    # Strip leading and trailing whitespace from each match
    questions = [q.strip() for q in questions]
    answers = [a.strip() for a in answers]

    # Pair up questions and answers
    qa_pairs = list(zip(questions, answers))

    return qa_pairs


def generate_questions_and_answers(text):
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Generate questions
    prompt_questions = f"{text}\n\nVerdiğim metin Türk eğitim sistemiyle ilgili bilgi içeriyor. Bu metnin içinde geçen bilgilerle cevaplanabilecek şekilde Türk eğitim sistemiyle ilgili genel sorular oluştur. Format: [s]soru_burada[s]"
    try:
        response_text = model.generate_content(prompt_questions,
                                               safety_settings={
                                                   HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                   HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                   HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                   HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                               })
        questions = extract_questions(response_text.text.strip())
    except ValueError as e:
        print(f"Error generating questions: {e}")
        questions = []

    qa_pairs = []
    for question in questions:
        if question.strip().endswith("?"):
            prompt_answer = f"İçerik:{text}\nSoru:{question}. Verdiğim içerik Türk eğitim sistemiyle ilgili bilgiler içeriyor. Sen bunu artık öğrendin ve benim sağladığımı unut. Resmi bir dille sen biliyormuşsun gibi cevapla. Format:[s]soru_burada[s][c]cevap_burada[c]"
            try:
                response_answer = model.generate_content(prompt_answer,
                                                         safety_settings={
                                                             HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                                                             HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                             HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                             HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                                                         })
                qa_pairs.extend(extract_qa(response_answer.text.strip()))
            except ValueError as e:
                print(
                    f"Error generating answer for question {question}\n\n{text}")

    return qa_pairs


def save_to_csv(qa_pairs, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(["Question", "Answer"])

        for question, answer in qa_pairs:
            csv_writer.writerow([question, answer])


def process_pdfs_in_directory(input_directory, output_directory):
    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # List all PDF files in the input directory
    pdf_files = [f for f in os.listdir(
        input_directory) if f.lower().endswith('.pdf')]

    for pdf_file in tqdm(pdf_files, desc="Processing PDFs", unit="file"):
        file_path = os.path.join(input_directory, pdf_file)
        text = load_pdf(file_path)
        text_chunks = split_text_into_chunks(text)

        all_qa_pairs = []
        for chunk in text_chunks:
            qa_pairs = generate_questions_and_answers(chunk)
            all_qa_pairs.extend(qa_pairs)

        output_file = os.path.join(
            output_directory, f"{os.path.splitext(pdf_file)[0]}_QA.csv")
        save_to_csv(all_qa_pairs, output_file)

        print(f"CSV file generated for {pdf_file}: {output_file}")


def main():
    input_directory = r"PDFs"
    output_directory = r"QAS"

    process_pdfs_in_directory(input_directory, output_directory)


if __name__ == "__main__":
    main()
