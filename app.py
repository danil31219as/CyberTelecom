import json

from flask import Flask, redirect, request, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything

app = Flask(__name__)
tasks = json.load(open('tasks.json'))

text_tokenizer = AutoTokenizer.from_pretrained(
    "Grossmend/rudialogpt3_medium_based_on_gpt2")
text_model = AutoModelForCausalLM.from_pretrained(
    "Grossmend/rudialogpt3_medium_based_on_gpt2")

device = 'cuda'
dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
dalle_tokenizer = get_tokenizer()
vae = get_vae(dwt=True).to(device)

# pipeline utils:
realesrgan = get_realesrgan('x2', device=device)
clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
clip_predictor = ruclip.Predictor(clip, processor, device, bs=8)

def get_length_param(text: str) -> str:
    tokens_count = len(text_tokenizer.encode(text))
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
        new_user_input_ids = text_tokenizer.encode(
            f"|0|{get_length_param(input_user)}|" + input_user + text_tokenizer.eos_token + "|1|1|",
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
            mask_token_id=text_tokenizer.mask_token_id,
            eos_token_id=text_tokenizer.eos_token_id,
            unk_token_id=text_tokenizer.unk_token_id,
            pad_token_id=text_tokenizer.pad_token_id,
            device='cuda',
        )
        bot_answer = text_tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        is_form = False
    else:
        answer = ''
        bot_answer = ''
        is_form = True
        pil_images = []
        scores = []
        for top_k, top_p, images_num in [
            (2048, 0.995, 8),
        ]:
            _pil_images, _scores = generate_images(question, dalle_tokenizer, dalle, vae,
                                                   top_k=top_k,
                                                   images_num=images_num, bs=8,
                                                   top_p=top_p)
            pil_images += _pil_images
            scores += _scores
        top_images, clip_scores = cherry_pick_by_ruclip(pil_images, question,
                                                        clip_predictor, count=1)
        sr_images = super_resolution(top_images, realesrgan)
        show(sr_images, 1, save_dir='static/img')

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
