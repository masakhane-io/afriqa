from transformers import DPRReader, DPRReaderTokenizer

tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
encoded_inputs = tokenizer(
    questions=["What is the capital of France?"],
    texts=["Paris is the capital of France."],
    return_tensors="pt",
)
print(encoded_inputs)
outputs = model(**encoded_inputs)
start_logits = outputs.start_logits
end_logits = outputs.end_logits
relevance_logits = outputs.relevance_logits


predicted_spans = tokenizer.decode_best_spans(
                encoded_inputs,
                outputs,
                # self.batch_size * self.num_spans_per_passage,
                # self.max_answer_length,
                # self.num_spans_per_passage,
            )

print(predicted_spans)