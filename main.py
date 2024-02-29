import spacy
import os
from tika import parser
from deep_translator import GoogleTranslator
import re
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# --------------------- #

# text_data = []
# folder = os.listdir('resumes')
# for f in folder:
#     path = 'resumes/' + f
#     parsed = parser.from_file(path)
#     text_data.append(parsed)

# --------------------- #

model_path = 'model'

def predict_block(block, model, tokenizer, device):
    model.eval()

    encoded = tokenizer.encode_plus(
        block,
        add_special_tokens=True,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
    )

    input_id = torch.tensor([encoded["input_ids"]]).to(device)
    attention_mask = torch.tensor([encoded["attention_mask"]]).to(device)

    with torch.no_grad():
        outputs = model(input_id, attention_mask=attention_mask)

    logits = outputs[0].detach().cpu().numpy()
    res = logits.argmax(axis=-1)[0]

    return res

tok = 'tokenizer_config.json'
voc = 'vocab.txt'

fast_tokenizer = DistilBertTokenizer(tokenizer_file=tok, vocab_file=voc)

the_model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=4,
    output_attentions=False,
    output_hidden_states=False,
)

the_model.load_state_dict(torch.load('blocks.h5', map_location=torch.device('cpu')))

# ------------------------- #

a = '''
1) загрузить резюме
2) предикт каждого блока + выявление сущностей
3) заполнение json
'''

jsoni = {
    "resume_id": "",
    "first_name": "",
    "last_name": "",
    "middle_name": "",
    "birth_date": "",
    "birth_date_year_only": "",
    "country": "",
    "city": "",
    "about": "",
    "key_skills": "",
    "salary_expectations_amount": "",
    "salary_expectations_currency": "",
    "photo_path": "",
    "gender": "",
    "language": "",
    "resume_name": "",
    "source_link": "",
    "contactItems": [
        {
        "resume_contact_item_id": "",
        "value": "",
        "comment": "",
        "contact_type": ""
        },
        {
        "resume_contact_item_id": "",
        "value": "",
        "comment": "",
        "contact_type": ""
        }
    ]
    ,
    "educationItems": [
        {
        "resume_education_item_id": "",
        "year": "",
        "organization": "",
        "faculty": "",
        "specialty": "",
        "result": "",
        "education_type": "",
        "education_level": ""
        }
    ],
    "experienceItems": [
        {
        "resume_experience_item_id": "",
        "starts": "",
        "ends": "",
        "employer": "",
        "city": "",
        "url": "",
        "position": "",
        "description": "",
        "order": ""
        }
    ],
    "languageItems": [
        {
        "resume_language_item_id": "",
        "language": "",
        "language_level": ""
        }
    ]
}

res_block = '''Experience in Selenium (Webdriver), C#.
Experience in SQL (MS SQL).
Experience in Waterfall and Agile (Scrum, Kanban) methodologies.
Experience in preparing and maintaining testing/technical documentation.
Strong communication and self-organizing skills, troubleshooting and problem-solving skills.
Strong work ethics, passionate and detail oriented, focused on result and quality, creative thinking, initiative and committed to project goals. Team player.
'''

i = 0
cur_file = parser.from_file(r'resumes\Alex Morozko.pdf')

labels = ['Personal info', 'Work Experience', 'Education', 'Skills']

result = re.sub(r'(\s*\n\s*){3,}', '\n\n', cur_file['content'])
blocks = []
emstr = ''
for r in result.split('\n\n'):
    cnt = 0
    for ch in r:
        if ch == '\n':
            cnt += 1
    if cnt < 3:
        emstr += r + '\n'
    else:
        blocks.append(r)
        if len(emstr) > 0:
            blocks.append(emstr)
        emstr = ''

if len(emstr) > 0:
    blocks.append(emstr)

model = spacy.load(model_path)

e_name    = []
e_contact = []
e_skills  = []
e_about   = []
e_edu     = []

phone_num, email_address = None, None
for b in blocks:
    phone_regex = r'((\+(\d+)|8)\s*\-*\(?(\d{3})\)?\s*\-*\d{3}\s*\-*\d{2}\s*\-*\d{2})'
    print(b)
    phone_matches = re.findall(phone_regex, b.replace('\n', ' '))
    email_regex = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    email_matches = re.findall(email_regex, b.replace('\n', ' '))

    if len(phone_matches) > 0:
        phone_num = phone_matches[0][0]

    if len(email_matches) > 0:
        email_address = email_matches[0]

    predicted_sentiment = predict_block(b, the_model, fast_tokenizer, torch.device('cpu'))
    cb = labels[predicted_sentiment]

    resume = model(b)
    jsoni["resume_id"] = i

    for ent in resume.ents:
        data = ent.text.replace('\n', '')
        # print(data, ent.label_)

        if cb == 'Personal info':
            if ent.label_ == 'Name':
                e_name.append(data)

        if cb == 'Skills':
            e_skills.append(data)

        if cb == 'Work Experience':
            e_about.append(data)

        if cb == 'Education':
            e_edu.append(data)


fn = cur_file['metadata']['resourceName']
jsoni["resume_name"] = fn[2:-1]

fio = fn.replace('   ', ' ').replace('  ', ' ').split(' ')
jsoni["first_name"] = fio[0][2:]
jsoni["last_name"] = fio[1].split('.')[0]

if phone_num:
    jsoni["contactItems"][0]['resume_contact_item_id'] = 0
    jsoni["contactItems"][0]['value'] = phone_num
    jsoni["contactItems"][0]['contact_type'] = 'Phone'

if email_address:
    jsoni["contactItems"][1]['resume_contact_item_id'] = 1
    jsoni["contactItems"][1]['value'] = email_address
    jsoni["contactItems"][1]['contact_type'] = 'Email'

jsoni["about"] = e_about
jsoni["key_skills"] = list(set(e_skills))

print('\nOutput:')
print(jsoni)

import json
with open(f"json\\{fio[0][2:]} {fio[1].split('.')[0]}.json", 'w') as json_file:
    json.dump(jsoni, json_file, indent=4)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# the_model.to(device)

# ----------------------------------- #

# def sort_resume(text):
#     r = re.compile(r"^[А-ЯЁ][а-яё]+")
#     if r.findall(text) is not None:
#         return 'ru'
#     else:
#         return 'eng'

# GoogleTranslator(source='ru', target='en').translate(...)

# ----------------------------------- #
