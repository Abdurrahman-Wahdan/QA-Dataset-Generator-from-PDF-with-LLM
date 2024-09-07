# QA Dataset Generator from PDF with LLM
>[!IMPORTANT]
>**Since the documents is in Turkish, the prompts used are also in Turkish.**

## Dividing the PDFs to Chunks
Using Gemini to create a Turkish dataset:
1. We first divide the content from the PDF into chunks, allowing Gemini to generate questions for each chunk.
2. One by one, we sent each question and the relevant chunk back to Gemini and asked it to answer that question. 

## Fixing the Response Format
Since LLMs do not always produce responses in the same format, we could not receive the questions and answers properly with a Python code. We solved this problem with a different approach. For example, a response format looks like this:

>Of course, I can produce questions and answers as you wish:
>
>
>Question1?
>
>Answer1
>
>Here is another question and answer:
>
>Question2?
>
>Answer2
>
>I hope it was useful.

**Another Response Format can be:**

>Of course I can make it in the format you want.
>
>Question1? Answer1
>
>Question2? Answer2



**As a solution to this problem, we changed the prompt so it displays question-answer pairs between certain labels. Using our format here's an example respose from Gemini downbelow:**

>Of course, I can produce questions and answers as you wish:
>
>[q] Question [/q]
>
>[a] The answer is [/a]

>[!NOTE]
>q stands for "Question"
>
>a stands for "Answer"

## Results
Thanks to our solution, now we can obtain all the questions and answers using `Regex` (Regular Expression) without missing on any data.
