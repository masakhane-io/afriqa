from transformers import DPRReader, DPRReaderTokenizer
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
import torch

tokenizer = AutoTokenizer.from_pretrained("ToluClassics/extractive_reader_nq_squad_v2")

model = AutoModelForQuestionAnswering.from_pretrained("ToluClassics/extractive_reader_nq_squad_v2")

question = "Who is the leader of the Shiites in Nigeria?"
context = "Ibraheem Yaqoub El-Zakzaky (alternately Ibraheem Zakzaky, Ibrahim Al-Zakzaky; born 5 May 1953) is a Nigerian religious leader. He is an imprisoned outspoken and prominent Shi'a Muslim leader in Nigeria.  He is the head of Nigeria's Islamic Movement, which he founded in the late 1970s, when a student at Ahmadu Bello University, and began propagating Shia Islam around 1979, at the time of the Iranian revolution\u2014which saw Iran's monarchy overthrown and replaced with an Islamic republic under Ayatollah Khomeini.  Zakzaky believed that the establishment of a republic along similar religious lines in Nigeria would be feasible. He has been detained several times due to accusations of civil disobedience or recalcitrance under military regimes in Nigeria during the 1980s and 1990s, and is still viewed with suspicion or as a threat by Nigerian authorities. In December 2015, the Nigerian Army raided his residence in Zaria, seriously injured him, and killed hundreds of his followers. Since then, he has remained under state detention in the nation's capital pending his release, which was ordered in late 2016.\nIn 2019, a court in Kaduna state granted him and his wife bail to seek treatment abroad but they returned from India after 3 days on the premises of unfair treatment and tough restrictions by security operatives deployed to the medical facility."

inputs = tokenizer.encode(question, context, add_special_tokens=True, return_tensors="pt")

tokens = tokenizer.convert_ids_to_tokens(inputs[0])

output = model(inputs)

answer_start = torch.argmax(output.start_logits)
answer_end = torch.argmax(output.end_logits)
if answer_end >= answer_start:
    answer = " ".join(tokens[answer_start:answer_end+1])

print(tokenizer.decode(inputs[0][answer_start:answer_end+1]))

# tokenizer = DPRReaderTokenizer.from_pretrained("facebook/dpr-reader-single-nq-base")
# model = DPRReader.from_pretrained("facebook/dpr-reader-single-nq-base")
# encoded_inputs = tokenizer(
#     questions=["What is the capital of France?"],
#     texts=["Paris is the capital of France."],
#     return_tensors="pt",
# )
# print(encoded_inputs)
# outputs = model(**encoded_inputs)
# start_logits = outputs.start_logits
# end_logits = outputs.end_logits
# relevance_logits = outputs.relevance_logits


# predicted_spans = tokenizer.decode_best_spans(
#                 encoded_inputs,
#                 outputs,
#                 # self.batch_size * self.num_spans_per_passage,
#                 # self.max_answer_length,
#                 # self.num_spans_per_passage,
#             )

# print(predicted_spans)