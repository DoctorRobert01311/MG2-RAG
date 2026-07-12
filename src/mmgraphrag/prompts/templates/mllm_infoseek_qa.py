qa_system="""
Answer the encyclopedic question about the given image. Don’t mention the visual content of image in your output. Directly output the answer of the question according to the context.
If you need to answer questions about numbers or time, please output the corresponding numerical format directly. If the context does not contain the information required to answer the question, you should answer the question using internal model knowledge.
There is an example:
- Context:
The Dolomites, also known as the Dolomite Mountains, Dolomite Alps or Dolomitic Alps, are a mountain
range located in northeastern Italy. The Dolomites are located in the regions of Veneto, Trentino-Alto
Adige/Südtirol and Friuli Venezia Giulia, covering an area shared between the provinces of Belluno,
Vicenza, Verona, Trentino, South Tyrol, Udine and Pordenone.
- Question: Which city or region does this mountain locate in?
Just answer the questions , no explanations needed. Short answer is: Province of Belluno
"""


prompt_template = [
    {"role": "system", "content": qa_system},
    {"role": "user", "content": "{Query Image}\n -Context:${context_user}\n -Question:${question_user}\nJust answer the questions , no explanations needed. Short answer is:"},
]