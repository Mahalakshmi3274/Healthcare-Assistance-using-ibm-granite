from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "ibm-granite/granite-3.3-2b-instruct"
save_dir = "./models/ibm-granite-3.3-2b-instruct"

# Download and save
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)

print("✅ Model downloaded to", save_dir)