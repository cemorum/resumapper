# resumapper
AI service for structuring unstructrured resumes in doxc, doc and pdf format

Веса для модели по классификации блоков - https://drive.google.com/file/d/1-q0Qa1QuZjfXMM3AyHDk2UG4lqJxsFHy/view?usp=sharing

Версия Python: 3.6

Шаг 1 - Установить зависимости из файла - requirements.txt

Шаг 2 - Установить библиотеку SpaCy через .whl файл

Шаг 3 - Установить PyTorch

Шаг 4 - Установить transformers

Шаг 5 - Указать путь к резюме в main.py (cur_file)

Шаг 6 - Запустить программу

#TODO:
* Сделать запуск программы через sys.argv
* Дообучить NER сегментатор

