import json 
import random
import torch
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    PeftModel,
    PeftConfig,
)
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
import bitsandbytes as bnb
from peft.tuners.lora import LoraLayer
import math
from accelerate import init_empty_weights
import transformers

gradient_accumulation_steps = 1
per_device_train_batch_size = 4
warmup_proportion = 0.3

def construct_training_data(training_file, tokenizer):
    data = []
    with open(training_file, 'r') as f:
        for l in f:
            data.append(json.loads(l))
    sentences = []
    for one_data in data:
        message = one_data['message']
        input_sent = message[1]
        output_sent = message[2]
        sentence= input_sent + '\nAnswer:' + output_sent
        sentences.append(sentence)
    random.shuffle(sentences)

    def load_custom_dataset(data):
        train_encodings = tokenizer(data, truncation=True, padding=True)

        class InputDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings

            def __getitem__(self, idx):
                item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
                return item

            def __len__(self):
                return len(self.encodings["input_ids"])

        train_dataset = InputDataset(train_encodings)

        return train_dataset
    
    return load_custom_dataset(sentences)

############# training helper function #############
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def find_all_linear_names(model):
    cls = bnb.nn.Linear4bit
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)

def training_steps(training_dataset_length):
    num_gpus = torch.cuda.device_count()
    training_steps = int(
        math.ceil(
            training_dataset_length
            / (gradient_accumulation_steps * per_device_train_batch_size)
        )
    )
    warmup_steps = int(math.ceil(training_steps * warmup_proportion))

    return training_steps, warmup_steps

def load_4bit_model(model_name):
    config = AutoConfig.from_pretrained(model_name, torch_dtype=torch.float16,trust_remote_code=True)
    config.tie_word_embeddings = True

    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config,trust_remote_code=True)
        model.tie_weights()

    if 'pretrained_models/' not in model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            device_map="auto",
            offload_state_dict=True,
            trust_remote_code=True,
            cache_dir='pretrained_models/'
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=nf4_config,
            device_map="auto",
            offload_state_dict=True,
            trust_remote_code=True
        )

    if 'Mistral' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='pretrained_models/', trust_remote_code=True)
        tokenizer.pad_token_id = tokenizer.eos_token_id
    elif 'vicuna' in model_name:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='pretrained_models/', trust_remote_code=True)
        tokenizer.pad_token = '</s>'
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name,cache_dir='pretrained_models/', trust_remote_code=True)
        model.config.pad_token_id = tokenizer.pad_token_id = 0
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config_eos_token_id = tokenizer.eos_token_id

    return model, tokenizer

def finetune_opensource_model(training_file, model):
    print("loading model ...")
    model, tokenizer = load_4bit_model(model)

    try:
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
    except:
        print("gradient checkpointing not supported for model {}".format(model))

    modules = find_all_linear_names(model)
    config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module = module.to(torch.float32)
        if "norm" in name:
            module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if module.weight.dtype == torch.float32:
                    module = module.to(torch.float32)

    print("model loaded.")

    train_dataset = construct_training_data(training_file, tokenizer)

    training_steps, warmup_steps = training_steps(len(train_dataset))
    print(
        """
length of training dataset: {}
number of training steps: {}
number of warmup steps: {}
    """.format(
            len(train_dataset), training_steps, warmup_steps
        )
    )

    print("start training ...")
    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            max_steps=training_steps,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=True,
            logging_steps=10,
            output_dir="outputs",
            optim="paged_adamw_8bit",
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(
            tokenizer, mlm=False
        ),
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )
    trainer.train()

    adapter_model_output_dir = 'pretrained_models/safeagent_adapter' + '/' + model.split('/')[-1]
    model.save_pretrained(adapter_model_output_dir)
    print("adapter model saved.")

    base_model = AutoModelForCausalLM.from_pretrained(model, device_map="auto", cache_dir='pretrained_models', trust_remote_code=True)
    model = PeftModel.from_pretrained(base_model, adapter_model_output_dir)
    model = model.merge_and_unload()

    whole_model_output_dir = 'pretrained_models/safeagent_whole_model' + '/' + model.split('/')[-1]
    model.save_pretrained(whole_model_output_dir)
    tokenizer.save_pretrained(whole_model_output_dir)
    print("whole model and tokenizer saved.")

    return whole_model_output_dir