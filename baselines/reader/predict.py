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

    result = nlp(question=question, context=context)
    print(
        f"Answer: '{result['answer']}', score: {round(result['score'], 4)}, start: {result['start']}, end: {result['end']}"
    )


if __name__ == "__main__":
    question = "What kind of organization is leading the country of Zambia?"
    context = 'UPND Youth League UPND Youth League The UPND Youth League is a youth organization of the United Party for National Development, a political party in Zambia. Formation. The Youth League was formed to help mobilise youths towards, the values, beliefs and principles of the UPND, the official opposition political party in Zambia. The Youth League was formed by the youth members of the party to spearhead the issues faced by young people in Zambia. Among them was the 4-month closure of the Copperbelt University which the Zambian Education minister had closed following student protests. UPND Youth League The UPND Youth League is'
    extractive_with_pipeline("deepset/bert-base-cased-squad2", question, context)