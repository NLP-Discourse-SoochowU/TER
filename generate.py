from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

""" I used this file to show the results for one specific cluster.
    For LLaMa, it already has one for CC summary.
"""

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")
model = AutoModelForSeq2SeqLM.from_pretrained("output_ent/checkpoint-1722", use_cache=False)  # flan-t5-xl/checkpoint-228
model = model.to("cuda")
model.eval()

while True:
    print("Longyin: ")
    cc = input("")
    if len(cc) == 0:
        break
    cc_ids = tokenizer(cc, return_tensors="pt").input_ids.to("cuda")
    cc_summary = model.generate(cc_ids, max_length=500)
    print("T5: \n", tokenizer.decode(cc_summary[0]))