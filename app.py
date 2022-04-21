import json

from flask import Flask, redirect, request, render_template
# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
tasks = json.load(open('tasks.json'))

# tokenizer = AutoTokenizer.from_pretrained(
#     "Grossmend/rudialogpt3_medium_based_on_gpt2")
# text_model = AutoModelForCausalLM.from_pretrained(
#     "Grossmend/rudialogpt3_medium_based_on_gpt2")

def get_length_param(text: str) -> str:
    tokens_count = len(tokenizer.encode(text))
    if tokens_count <= 15:
        len_param = '1'
    elif tokens_count <= 50:
        len_param = '2'
    elif tokens_count <= 256:
        len_param = '3'
    else:
        len_param = '-'
    return '1'


@app.route('/')
def hello_world():  # put application's code here
    return redirect('/test/0')


@app.route('/test/<int:id>', methods=['GET', 'POST'])
def test(id):  # put application's code here
    question = tasks[id]['q']
    if id == 0:
        placeholder = 'возраст'
    elif id == len(tasks):
        placeholder = 'реакция'
    else:
        placeholder = 'ответ'
    if request.method == 'POST':
        answer = request.form.get('answer')
        input_user = question + ' ' + answer
        new_user_input_ids = tokenizer.encode(
            f"|0|{get_length_param(input_user)}|" + input_user + tokenizer.eos_token + "|1|1|",
            return_tensors="pt")
        bot_input_ids = new_user_input_ids
        chat_history_ids = text_model.generate(
            bot_input_ids,
            num_return_sequences=1,
            max_length=512,
            no_repeat_ngram_size=3,
            do_sample=True,
            top_k=50,
            top_p=0.9,
            temperature=0.9,
            mask_token_id=tokenizer.mask_token_id,
            eos_token_id=tokenizer.eos_token_id,
            unk_token_id=tokenizer.unk_token_id,
            pad_token_id=tokenizer.pad_token_id,
            device='cuda',
        )
        bot_answer = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        is_form = False
    else:
        answer = ''
        bot_answer = 'test test test'
        is_form = True
    args = {
        'placeholder': placeholder,
        'number': id,
        'question': question,
        'answer': answer,
        'bot_answer': bot_answer,
        'is_form': is_form,
        'next_id': id+1
    }
    return render_template('login.html', **args)


if __name__ == '__main__':
    app.run()
