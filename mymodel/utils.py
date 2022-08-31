import torch
from mymodel.classes import Seq2Seq
from transformers import BertTokenizer
from mymodel import tokenizer, MAX_LEN, device


def tokenize(text):
    return tokenizer(text=text, add_special_tokens=True, padding='max_length',
                     max_length=MAX_LEN, truncation=True, return_tensors='pt').to(device)


def generate_answer(question, tokenizer: BertTokenizer, model: Seq2Seq, device, max_len=MAX_LEN):
    model.eval()
    question = tokenize(question)
    input_ids = question.input_ids
    src_mask = question.attention_mask
    token_type_ids = question.token_type_ids
    batch_size = input_ids.shape[0]

    with torch.no_grad():
        enc_src = model.encoder(input_ids, src_mask, token_type_ids).last_hidden_state

    src_mask = model.make_src_mask(question.input_ids)
    trg_indices = [[tokenizer.bos_token_id] for _ in range(batch_size)]
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indices).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)

        with torch.no_grad():
            output = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)

        # maybe beam search
        pred_token = [i.item() for i in output.argmax(2)[:, -1]]  # greedy

        for j in range(batch_size):
            trg_indices[j].append(pred_token[j])

        if pred_token == [tokenizer.eos_token_id for _ in range(batch_size)]:
            break
    output_indices = [indices[1:-1] for indices in trg_indices]
    washed_indices = []
    for indices in output_indices:
        this_indices = []
        for idx in indices:
            if idx == tokenizer.eos_token_id:
                break
            else:
                this_indices.append(idx)
        washed_indices.append(this_indices)
    answer = [tokenizer.decode(indices).strip() for indices in washed_indices]
    return answer
