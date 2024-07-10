import pandas as pd
import spacy
import re
import argparse
import os
from tqdm import tqdm
from openpyxl import load_workbook

# Lista actualizada de conjugaciones de "haber"
haber_conjugations = [
    "hemos", "ha", "hay", "han", "había", "habría", "hubo", "habían",
    "haya", "hayan", "habido", "haber", "habrá", "he", "hayamos",
    "hubiéramos", "hubieran", "habíamos", "hubiese", "habrán",
    "habremos", "hubiera", "habríamos", "has", "hubiésemos",
    "habrían", "hubieron", "habiendo", "habéis", "habías",
    "hubiesen", "hayas", "habrás", "hayáis", "habré", "habemos",
    "haberes", "hube", "habidos", "habes", "habí", "haiga", "haigan"
]

# Función para mapear timestamps con tokens
def map_timestamps_to_tokens(text):
    timestamp_pattern = re.compile(r'\[(\d{1,2}):(\d{2}):(\d{2})\]')
    parts = timestamp_pattern.split(text)
    timestamps = []
    texts = []
    for i in range(0, len(parts), 4):
        texts.append(parts[i])
        if i + 3 < len(parts):
            timestamps.append((int(parts[i+1]), int(parts[i+2]), int(parts[i+3])))
    return texts, timestamps

# Función para preprocesar el texto eliminando timestamps y convirtiendo a minúsculas
def preprocess_text(text):
    text_no_timestamps = re.sub(r'\[(\d{1,2}):(\d{2}):(\d{2})\]', '', text)
    return text_no_timestamps.lower()

# Función para extraer contextos de "haber" con timestamps y clasificar normativa
def extract_haber_contexts(text, row, nlp):
    original_texts, timestamps = map_timestamps_to_tokens(text)
    preprocessed_text = preprocess_text(text)
    words = preprocessed_text.split()
    original_text = ' '.join(original_texts)
    original_words = original_text.split()

    # Crear una lista de tuplas (palabra, timestamp)
    word_timestamp_pairs = []
    timestamp_index = 0

    for segment in original_texts:
        segment_words = segment.split()
        for word in segment_words:
            timestamp = timestamps[timestamp_index] if timestamp_index < len(timestamps) else (0, 0, 0)
            word_timestamp_pairs.append((word, timestamp))
        timestamp_index += 1

    contexts = []
    seen_contexts = set()

    for i, word in enumerate(words):
        if word in haber_conjugations:
            original_word_indices = [index for index, (w, _) in enumerate(word_timestamp_pairs) if w and w.lower() == word]
            for original_word_index in original_word_indices:
                timestamp = word_timestamp_pairs[original_word_index][1]
                hours, minutes, seconds = timestamp
                time_seconds = hours * 3600 + minutes * 60 + seconds
                before = ' '.join(words[max(0, i - 10):i])
                after = ' '.join(words[i + 1:i + 11])
                match = f"{before} {word} {after}"

                # Check if this context has been seen before
                context_key = (before, word, after)
                if context_key not in seen_contexts:
                    context = {
                        'Before': before,
                        'Haber': word,
                        'After': after,
                        'Match': match,
                        'Exact_Timestamp': time_seconds,
                        'Video ID': row['Video ID'],
                    }
                    spacy_results = process_sentence(match, word, nlp)
                    context.update(spacy_results)
                    classification = classify_row(context, nlp)
                    context['Classification'] = classification
                    contexts.append(context)
                    seen_contexts.add(context_key)
                
                # Remove the processed word from word_timestamp_pairs to avoid duplicate processing
                word_timestamp_pairs[original_word_index] = (None, timestamp)
                break  # Consider only the first matching timestamp for this instance of "haber"
    return contexts

# Crear una función para generar la oración combinada
def create_sentence(row):
    before_words = row['Before'].split()[-5:]
    after_words = row['After'].split()[:5]
    sentence = ' '.join(before_words + [row['Haber']] + after_words)
    return sentence

# Procesar las oraciones y extraer los atributos
def process_sentence(sentence, haber, nlp):
    doc = nlp(sentence)
    results = {
        'haber.pos': None,
        'haber.dep': None,
        'haber.morph': None,
        'haber.child.text': None,
        'haber.child.lemma': None,
        'haber.child.pos': None,
        'haber.child.dep': None,
        'haber.child.morph': None,
        'haber.head.text': None,
        'haber.head.lemma': None,
        'haber.head.pos': None,
        'haber.head.dep': None,
        'haber.head.morph': None,
    }

    for token in doc:
        if (token.text == haber) or (token.lemma_ == haber):
            results['haber.pos'] = str(token.pos_)
            results['haber.dep'] = str(token.dep_)
            results['haber.morph'] = str(token.morph)

            # Procesar hijos
            if token.children:
                children = list(token.children)
                if children:
                    results['haber.child.text'] = str(children[0].text)
                    results['haber.child.lemma'] = str(children[0].lemma_)
                    results['haber.child.pos'] = str(children[0].pos_)
                    results['haber.child.dep'] = str(children[0].dep_)
                    results['haber.child.morph'] = str(children[0].morph)

            # Procesar cabeza
            head = token.head
            results['haber.head.text'] = str(head.text)
            results['haber.head.lemma'] = str(head.lemma_)
            results['haber.head.pos'] = str(head.pos_)
            results['haber.head.dep'] = str(head.dep_)
            results['haber.head.morph'] = str(head.morph)
            break

    return results

# Función para clasificar la fila
def classify_row(row, nlp):
    if pd.isna(row['After']):
        return 'sing'
    doc_after = nlp(str(row['After']))
    doc_before = nlp(str(row['Before']))
    if len(doc_after) > 0:
        first_token_after = doc_after[0]
        if first_token_after.text.lower() == 'habido':
            if len(doc_after) > 1:
                second_token_after = doc_after[1]
                if (second_token_after.pos_ in ['PRON', 'PROPN', 'NOUN', 'ADJ', 'DET', 'NUM']) and (second_token_after.morph.get('Number') == ['Plur']):
                    return 'plur'
                if second_token_after.pos_ == 'ADV' and len(doc_after) > 2:
                    third_token_after = doc_after[2]
                    if (third_token_after.pos_ in ['PRON', 'PROPN', 'NOUN', 'ADJ', 'DET', 'NUM']) and (third_token_after.morph.get('Number') == ['Plur']):
                        return 'plur'
                if (second_token_after.pos_ == 'NUM') and (second_token_after.morph.get('NumForm') == ['Digit']):
                    if second_token_after.text not in ['0', '1']:
                        return 'plur'
        elif (first_token_after.pos_ in ['PRON', 'PROPN', 'NOUN', 'ADJ', 'DET', 'NUM']) and (first_token_after.morph.get('Number') == ['Plur']):
            return 'plur'
        elif (first_token_after.pos_ == 'NUM') and (first_token_after.morph.get('NumForm') == ['Digit']):
            if first_token_after.text not in ['0', '1']:
                return 'plur'
        elif len(doc_after) > 1:
            second_token_after = doc_after[1]
            if (first_token_after.pos_ == 'ADV') and (second_token_after.pos_ in ['PRON', 'PROPN', 'NOUN', 'ADJ', 'DET', 'NUM']) and (second_token_after.morph.get('Number') == ['Plur']):
                return 'plur'

    # New conditions
    if len(doc_before) > 0 and len(doc_after) > 0:
        last_token_before = doc_before[-1]
        first_token_after = doc_after[0]
        if (last_token_before.pos_ in ['PRON', 'PROPN', 'NOUN', 'ADJ', 'DET', 'NUM']) and (last_token_before.morph.get('Number') == ['Plur']) and (first_token_after.pos_ == 'ADP'):
            return 'plur'
        if (last_token_before.pos_ == 'NUM') and (last_token_before.morph.get('NumForm') == ['Digit']) and (last_token_before.text not in ['0', '1']) and (first_token_after.pos_ == 'ADP'):
            return 'plur'

    return 'sing'

def filter_conditions(df, nlp):
    df['After'] = df['After'].fillna('')
    
    # Aplicar la primera condición (sin tener en cuenta si Haber es "hay")
    cond1 = (df['haber.dep'] == 'aux') & (~df['After'].str.startswith('habido')) & (df['Haber'] != 'hay')
    haber_no_ex_1 = df[cond1].copy()
    haber_no_ex_1['Condicion'] = 'Condicion 1'
    df_remaining_1 = df[~cond1]

    # Aplicar la segunda condición
    cond2 = df_remaining_1['After'].str.startswith('que ')
    haber_no_ex_2 = df_remaining_1[cond2].copy()
    haber_no_ex_2['Condicion'] = 'Condicion 2'
    df_remaining_2 = df_remaining_1[~cond2]

    # Aplicar la tercera condición (sin tener en cuenta si Haber es "hay")
    cond3 = df_remaining_2['After'].apply(lambda text: is_first_word_participle_not_habido(text, nlp)) & (df_remaining_2['Haber'] != 'hay')
    haber_no_ex_3 = df_remaining_2[cond3].copy()
    haber_no_ex_3['Condicion'] = 'Condicion 3'
    df_remaining_3 = df_remaining_2[~cond3]

    # Aplicar la cuarta condición
    cond4 = (df_remaining_3['Haber'].isin(['ha', 'han'])) & (~df_remaining_3['After'].str.startswith('habido'))
    haber_no_ex_4 = df_remaining_3[cond4].copy()
    haber_no_ex_4['Condicion'] = 'Condicion 4'
    df_remaining_4 = df_remaining_3[~cond4]

    # Combinar los resultados que cumplen con alguna condición
    haber_no_ex = pd.concat([haber_no_ex_1, haber_no_ex_2, haber_no_ex_3, haber_no_ex_4])

    # Filas que no cumplen con ninguna condición
    haber_ex = df_remaining_4

    return haber_no_ex, haber_ex

def process_ex_parquet(parquet_path, output_xlsx_path, nlp):
    df_ex = pd.read_parquet(parquet_path)

    def is_last_words_in_list(text, words_list):
        if not text:
            return False
        words = text.split()
        return any(' '.join(words[i:]).lower() in words_list for i in range(len(words)))

    df_ex['Before'] = df_ex['Before'].fillna('')
    df_ex['After'] = df_ex['After'].fillna('')

    conditions = {
        'Condicion 1': (df_ex['haber.morph'].str.contains('Number=Sing')) & (df_ex['haber.child.dep'] == 'obj') &
                       (df_ex['haber.child.morph'].str.contains('Number=Sing') | df_ex['haber.child.morph'].str.contains('Number=Plur')),
        'Condicion 2': (df_ex['haber.morph'].str.contains('Number=Plur')) & (df_ex['haber.child.dep'] == 'obj') &
                       (df_ex['haber.child.morph'].str.contains('Number=Plur')),
        'Condicion 3': df_ex['Before'].apply(lambda x: is_last_words_in_list(x, ['han', 'habían', 'habrían', 'hayan', 'habrán', 'hubiesen', 'hubieran'])) &
                       (df_ex['Haber'].str.lower() == 'habido'),
        'Condicion 4': df_ex['Before'].apply(lambda x: is_last_words_in_list(x, ['ha', 'había', 'habría', 'haya', 'habrá', 'hubiese', 'hubiera', 'haber'])) &
                       (df_ex['Haber'].str.lower() == 'habido'),
        'Condicion 5': df_ex['Haber'].str.lower().isin(['había', 'habrá', 'habría', 'hubiera', 'haya', 'haiga', 'hubo', 'ha', 'habrá']),
        'Condicion 6': df_ex['Haber'].str.lower().isin(['habían', 'habrán', 'habrían', 'hubieran', 'hayan', 'haigan', 'hubieron', 'han']),
        'Condicion 7': df_ex['Haber'].str.lower().isin(['hubiese', 'habiendo', 'hay']),
        'Condicion 8': df_ex['Haber'].str.lower().isin(['hubiesen', 'habemos', 'habíamos', 'hayamos', 'habidos']),
        'Condicion 9': df_ex['Before'].apply(lambda x: is_last_words_in_list(x, [
            'pueden', 'podían', 'podrían', 'deben', 'deberían', 'deben de', 'deberían de', 'van a', 'van', 'tienen que', 'tienen',
            'llegaron a', 'empiezan a', 'han de', 'siguen', 'suelen', 'solían', 'debían', 'debían de', 'puedan', 'pudieran',
            'podrán', 'vayan a', 'vayan', 'iban a', 'iban', 'hayan podido', 'tengan que', 'tuvieran que', 'tuvieron que',
            'tenían que', 'tendrían que'
        ])) & (df_ex['Haber'].str.lower() == 'haber'),
        'Condicion 10': df_ex['Before'].apply(lambda x: is_last_words_in_list(x, [
            'puede', 'podía', 'podría', 'debe', 'debería', 'debe de', 'debería de', 'va a', 'va', 'tiene que', 'tiene',
            'llegar a', 'llegara a', 'empieza a', 'ha de', 'sigue', 'suele', 'solía', 'debía', 'debía de', 'debió', 'pueda',
            'pudiera', 'podrá', 'vaya a', 'vaya', 'iba a', 'iba', 'haya podido', 'tenga que', 'tuviera que', 'tenido que',
            'tenía que', 'tendría que'
        ])) & (df_ex['Haber'].str.lower() == 'haber'),
        'Condicion 11': df_ex['Before'].apply(lambda x: is_last_words_in_list(x, ['están'])) & (df_ex['Haber'].str.lower() == 'habiendo'),
        'Condicion 12': df_ex['Before'].apply(lambda x: is_last_words_in_list(x, ['está'])) & (df_ex['Haber'].str.lower() == 'habiendo'),
    }

    frames_normativa = ['Condicion 1', 'Condicion 4', 'Condicion 5', 'Condicion 7', 'Condicion 10', 'Condicion 12']
    frames_pluralizacion = ['Condicion 2', 'Condicion 3', 'Condicion 6', 'Condicion 8', 'Condicion 9', 'Condicion 11']

    haber_normativa = pd.concat([df_ex[conditions[cond]].copy().assign(Condicion=cond) for cond in frames_normativa])
    haber_pluralizacion = pd.concat([df_ex[conditions[cond]].copy().assign(Condicion=cond) for cond in frames_pluralizacion])

    # Clasificar haber_normativa en normativa_plur y normativa_sing
    haber_normativa_plur = haber_normativa[haber_normativa.apply(lambda row: classify_row(row, nlp) == 'plur', axis=1)]
    haber_normativa_sing = haber_normativa[haber_normativa.apply(lambda row: classify_row(row, nlp) == 'sing', axis=1)]

    cond_total = pd.concat([df_ex[conditions[cond]] for cond in conditions.keys()]).index
    haber_no_clasificado = df_ex[~df_ex.index.isin(cond_total)].copy()
    haber_no_clasificado['Condicion'] = 'sin_class'

    with pd.ExcelWriter(output_xlsx_path, engine='openpyxl') as writer:
        haber_normativa_plur.to_excel(writer, sheet_name='normativa_plur', index=False)
        haber_normativa_sing.to_excel(writer, sheet_name='normativa_sing', index=False)
        haber_pluralizacion.to_excel(writer, sheet_name='pluralización', index=False)
        haber_no_clasificado.to_excel(writer, sheet_name='sin_class', index=False)

    print(f'Total de casos en normativa_plur: {len(haber_normativa_plur)}')
    print(f'Total de casos en normativa_sing: {len(haber_normativa_sing)}')
    print(f'Total de casos en pluralización: {len(haber_pluralizacion)}')
    print(f'Total de casos no clasificados: {len(haber_no_clasificado)}')

def main(parquet_folder, output_folder, spacy_model):
    # Instalar los requerimientos necesarios
    os.system("pip install pandas spacy fastparquet tqdm openpyxl")

    # Instalar el modelo de spaCy necesario
    if spacy_model == "trf":
        os.system("pip install https://huggingface.co/spacy/es_dep_news_trf/resolve/main/es_dep_news_trf-any-py3-none-any.whl")
        model_name = "es_dep_news_trf"
    else:
        model_name = f"es_core_news_{spacy_model}"
        os.system(f"python -m spacy download {model_name}")

    # Cargar el modelo de spaCy
    nlp = spacy.load(model_name)

    # Leer todos los archivos .parquet en la carpeta y combinarlos en un DataFrame
    all_files = [os.path.join(parquet_folder, f) for f in os.listdir(parquet_folder) if f.endswith('.parquet')]
    df_list = [pd.read_parquet(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # Crear el directorio de salida si no existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    batch_size = 100  # Procesar 100 filas a la vez
    results = []

    for start_idx in tqdm(range(0, df.shape[0], batch_size), desc="Procesando transcripciones por lotes"):
        end_idx = min(start_idx + batch_size, df.shape[0])
        batch_df = df[start_idx:end_idx]
        batch_results = []
        for idx, row in batch_df.iterrows():
            transcription = row['Transcription']
            contexts = extract_haber_contexts(transcription, row, nlp)
            for context in contexts:
                batch_results.append(context)
        results.extend(batch_results)

    combined_df = pd.DataFrame(results)

    # Filtrar las condiciones y separar los datos
    haber_no_ex, haber_ex = filter_conditions(combined_df, nlp)

    # Guardar los resultados en archivos Parquet separados
    output_parquet_no_ex_path = os.path.join(output_folder, "corpus_haber_yt_parsed_no_ex.parquet")
    output_parquet_ex_path = os.path.join(output_folder, "corpus_haber_yt_parsed_ex.parquet")

    haber_no_ex.to_parquet(output_parquet_no_ex_path, index=False)
    haber_ex.to_parquet(output_parquet_ex_path, index=False)

    print(f"DataFrames saved as Parquet at: {output_parquet_no_ex_path} and {output_parquet_ex_path}")

    # Procesar el archivo corpus_haber_yt_parsed_ex.parquet para crear el Excel
    output_xlsx_path = os.path.join(output_folder, "haber_ex_norm_plur_yt.xlsx")
    process_ex_parquet(output_parquet_ex_path, output_xlsx_path, nlp)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process transcripts and extract contexts of 'haber'.")
    parser.add_argument("--parquet", required=True, help="Path to the input folder containing Parquet files.")
    parser.add_argument("--output", required=True, help="Path to the output folder.")
    parser.add_argument("--model", required=True, choices=["sm", "md", "lg", "trf"], help="spaCy model size.")
    args = parser.parse_args()

    main(args.parquet, args.output, args.model)
