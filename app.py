import json

from flask import Flask, redirect, request, render_template
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_ruclip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan
from rudalle.utils import seed_everything
from transformers import T5ForConditionalGeneration, T5Tokenizer


app = Flask(__name__)
tasks = json.load(open('tasks.json'))
img_list = [f'img_{i}.png' for i in range(17) if i != 12] + ['image3.jpeg']
text_tokenizer = AutoTokenizer.from_pretrained(
    "Grossmend/rudialogpt3_medium_based_on_gpt2")
text_model = AutoModelForCausalLM.from_pretrained(
    "Grossmend/rudialogpt3_medium_based_on_gpt2")
#
# device = 'cuda'
# dalle = get_rudalle_model('Malevich', pretrained=True, fp16=True, device=device)
# dalle_tokenizer = get_tokenizer()
# vae = get_vae(dwt=True).to(device)
#
# # pipeline utils:
# realesrgan = get_realesrgan('x2', device=device)
# clip, processor = ruclip.load('ruclip-vit-base-patch32-384', device=device)
# clip_predictor = ruclip.Predictor(clip, processor, device, bs=1)

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
def paraphrase(model, tokenizer, text, beams=5, grams=4, do_sample=False):
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams, num_beams=beams, max_length=max_size, do_sample=do_sample)
    return tokenizer.decode(out[0], skip_special_tokens=True)

@app.route('/')
def hello_world():  # put application's code here
    return redirect('/test/0')

@app.route('/admin')
def admin():  # put application's code here
    return render_template('index.html')


@app.route('/test/<int:id>', methods=['GET', 'POST'])
def test(id):  # put application's code here
    question = tasks[id]['q']
    score = request.args.get('score')
    if score is None:
        score = '0'
    score = int(score)
    if id == 0:
        placeholder = '??????????????'
    elif id == len(tasks) - 1:
        placeholder = '??????????????'
        if score > 10:
            question = f'??????! {score} ???????????? ???? ????????! ???? ??????????????????!'
        else:
            question = f'??????????????, ??????????????, ?? ???????????? {score} ???????? ??????????????'
        para_model = T5ForConditionalGeneration.from_pretrained('cointegrated/rut5-base-paraphraser')
        para_tokenizer = T5Tokenizer.from_pretrained('cointegrated/rut5-base-paraphraser')
        para_model.cuda()
        para_model.eval()
        question = paraphrase(para_model, para_tokenizer, question)
        score = 0
    else:
        placeholder = '??????????'

    if request.method == 'POST':
        answer = request.form.get('answer')
        true_answer = tasks[id]['a']
        if true_answer.lower() == answer.lower():
            score += 1
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
        #bot_answer = '??????\n??????'
        is_form = False
    else:
        answer = ''
        bot_answer = ''
        is_form = True
        # pil_images = []
        # scores = []
        # for top_k, top_p, images_num in [
        #     (2048, 0.995, 8),
        # ]:
        #     _pil_images, _scores = generate_images(question, dalle_tokenizer, dalle, vae,
        #                                            top_k=top_k,
        #                                            images_num=images_num, bs=1,
        #                                            top_p=top_p)
        #     pil_images += _pil_images
        #     scores += _scores
        # top_images, clip_scores = cherry_pick_by_ruclip(pil_images, question,
        #                                                 clip_predictor, count=1)
        # sr_images = super_resolution(top_images, realesrgan)
        # show(sr_images, 1, save_dir='static/img')

    args = {
        'placeholder': placeholder,
        'number': id,
        'question': question,
        'answer': answer,
        'bot_answer': bot_answer,
        'is_form': is_form,
        'next_id': (id+1) % 17,
        'score': score,
        'img_path': img_list[id]
    }
    return render_template('login.html', **args)


if __name__ == '__main__':
    app.run()
