from  transformers  import  AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def predict_seq_2_seq(model_name: str, question: str, context:str, max_length: int=384):
    """
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input = f"question: {question} context: {context}"
    encoded_input = tokenizer([input],
                            return_tensors='pt',
                            max_length=max_length,
                            truncation=True)
    output = model.generate(input_ids = encoded_input.input_ids,
                        attention_mask = encoded_input.attention_mask)
    output = tokenizer.decode(output[0], skip_special_tokens=True)
    print(output)


def predict_seq_2_seq_with_pipeline(model_name: str, question: str, context:str):
    text2text_generator = pipeline("text2text-generation",  model=model_name, device=0)
    result = text2text_generator(f"question:{question} context: {context}")

    print(result)

def extractive_with_pipeline(model_name: str, question: str, context:str):
    nlp = pipeline("question-answering",  model=model_name, device=0)

    result = question_answerer(question=question, context=context)
    print(
        f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
    )


if __name__ == "__main__":
    question = "When did Beyonce start becoming popular?"
    context = 'Beyoncé Giselle Knowles-Carter (/biːjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destinys Child. Managed by her father, Mathew Knowles, the group became one of the worlds best-selling girl groups of all time. Their hiatus saw the release of Beyoncés debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".'
    predict_seq_2_seq_with_pipeline("/home/oogundep/african_qa/models/google/mt5-base/checkpoint-80000", question, context)