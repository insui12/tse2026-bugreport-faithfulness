import pandas as pd
import os
import re
import textstat
import stanza
import numpy as np

# 🚨 워커마다 독립적으로 nlp 로드 (데드락 방지)
_NLP_INSTANCE = None

def get_nlp():
    global _NLP_INSTANCE
    if _NLP_INSTANCE is None:
        _NLP_INSTANCE = stanza.Pipeline(
            'en',
            processors='tokenize,pos,lemma,depparse',
            download_method=None,
            use_gpu=False,   # GPU 사용 금지
            verbose=False,
        )
    return _NLP_INSTANCE

# Load words from CSV files
def load_word_list(filename):
    file_path = os.path.join(filename)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        return set(df['word'].dropna().str.lower())
    return set()

USER_DIRECT_VERBS = load_word_list("final_unique_behaviours.csv")
INTERFACE_ELEMENT_WORDS = load_word_list("interface_element_words.csv") or load_word_list("interactive_elements.csv")
SYSTEM_DEFECT_WORDS = load_word_list("_negative_words.csv")

ENVIRONMENT_WORDS = {"android", "ios", "chrome", "firefox", "windows", "macos"}
SCREENSHOT_WORDS = {"screenshot", "attached image", "see attachment"}
SCREENSHOT_GUIDELINE_WORDS = {"please see", "as shown in the image", "attachment"}
if not INTERFACE_ELEMENT_WORDS:
    INTERFACE_ELEMENT_WORDS = {"adapter", "button", "page", "menu", "dialog", "tab", "screen"}
if not SYSTEM_DEFECT_WORDS:
    SYSTEM_DEFECT_WORDS = {"crash", "down", "flashback", "overlap", "too big", "too small", "freeze"}
if not USER_DIRECT_VERBS:
    USER_DIRECT_VERBS = {"login", "register", "logout"}

# ============= MORPHOLOGICAL INDICATORS =============
# 순수 텍스트 연산 (빠름)
def check_RM1_size(report_text, min_tokens=50, max_tokens=300):
    tokens = report_text.split()
    if min_tokens <= len(tokens) <= max_tokens:
        return True, 1
    return False, 0

def check_RM2_readability(report_text, min_score=30, max_score=100):
    if not report_text.strip():
        return False, 0
    flesch_score = textstat.flesch_reading_ease(report_text)
    return min_score <= flesch_score <= max_score, 1 if min_score <= flesch_score <= max_score else 0

def check_RM3_punctuation(report_text):
    sentences = re.split(r'[.!?]', report_text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]
    for s in sentences:
        if not re.match(r'.+[.!?]$', s + "."):
            return False, 0
    return True, 1

def check_RM4_avg_sentence_length(doc, min_len=5, max_len=40):
    sents = list(doc.sentences)
    if not sents:
        return False, 0
    lengths = [len(sent.words) for sent in sents]
    avg_len = sum(lengths)/len(lengths)
    if min_len <= avg_len <= max_len:
        return True, 1
    return False, 0

# ============= RELATIONAL INDICATORS =============
# 순수 텍스트 연산 (빠름)
def check_RR1_itemization(report_text):
    if re.search(r'^(\d+\)|-|\*)', report_text, flags=re.MULTILINE):
        return True, 1
    return False, 0

def check_RR2_itemization_symbol(report_text):
    if re.search(r'^(-|\*)', report_text, flags=re.MULTILINE):
        return True, 1
    return False, 0

def check_RR3_environment(report_text):
    text_lower = report_text.lower()
    for word in ENVIRONMENT_WORDS:
        if word in text_lower:
            return True, 2
    return False, 0

def check_RR4_screenshot(report_text):
    text_lower = report_text.lower()
    for w in SCREENSHOT_WORDS:
        if w in text_lower:
            return True, 1
    return False, 0

def check_RR5_screenshot_guideline(report_text):
    text_lower = report_text.lower()
    for w in SCREENSHOT_GUIDELINE_WORDS:
        if w in text_lower:
            return True, 1
    return False, 0

# ============= RELATIONAL ANALYSIS FUNCTIONS =============
def get_phrase_with_modifiers(sentence, word_id):
    word = sentence.words[word_id - 1]
    modifiers = [w.text for w in sentence.words if w.head == word_id]
    phrase = word.text + ' ' + ' '.join(modifiers)
    return phrase

def has_location_preposition(sentence, word_id):
    for w in sentence.words:
        if w.head == word_id and w.upos == 'ADP' and w.text.lower() in ['in', 'on', 'at']:
            return True
    return False

def has_time_preposition(sentence, word_id):
    for w in sentence.words:
        if w.head == word_id and w.upos == 'ADP' and w.text.lower() in ['during', 'after', 'before']:
            return True
    return False

# 🚨 미리 분석된 doc을 파라미터로 받아서 중복 연산을 피함
def check_RA1_interface_element(doc, text):
    text_lower = text.lower()
    if not any(w in text_lower for w in INTERFACE_ELEMENT_WORDS):
        return False, 0
    
    relation_mapping = {
        'ATT': ['amod', 'nmod', 'nummod', 'det', 'compound', 'case', 'mark'],
        'ADV': ['advmod', 'advcl', 'obl', 'neg'],
        'CMP': ['obj', 'iobj', 'xcomp', 'ccomp', 'acl'],
        'COO': ['conj', 'cc'],
        'LAD': ['acl:relcl', 'acl'],
        'RAD': ['appos', 'parataxis']
    }
    
    relation_lookup = {}
    for rel_type, ud_rels in relation_mapping.items():
        for ud_rel in ud_rels:
            relation_lookup[ud_rel] = rel_type
            
    interface_elements_found = False
    max_relation_score = 0
    
    for sentence in doc.sentences:
        for word in sentence.words:
            if word.lemma.lower() in INTERFACE_ELEMENT_WORDS:
                interface_elements_found = True
                relations = {'ATT': 0, 'ADV': 0, 'CMP': 0, 'COO': 0, 'LAD': 0, 'RAD': 0}
                
                for other_word in sentence.words:
                    if other_word.head == word.id:
                        rel_type = relation_lookup.get(other_word.deprel, None)
                        if rel_type: relations[rel_type] += 1
                
                if word.head != 0:
                    head_word = sentence.words[word.head - 1]
                    rel_type = relation_lookup.get(word.deprel, None)
                    if rel_type: relations[rel_type] += 1
                
                total_relation_count = sum(relations.values())
                relation_score = 2 if total_relation_count >= 4 else (1 if total_relation_count >= 2 else 0)
                max_relation_score = max(max_relation_score, relation_score)
                
    return interface_elements_found and max_relation_score > 0, max_relation_score

def check_RA2_user_behavior(doc, text):
    text_lower = text.lower()
    if any(verb in text_lower for verb in USER_DIRECT_VERBS):
        return True, 2
        
    for sentence in doc.sentences:
        predicates = [word for word in sentence.words if word.upos == 'VERB']
        for verb in predicates:
            obj_deps = [w for w in sentence.words if w.head == verb.id and w.deprel in ['obj', 'iobj']]
            for obj in obj_deps:
                obj_phrase = get_phrase_with_modifiers(sentence, obj.id)
                if any(element in obj_phrase.lower() for element in INTERFACE_ELEMENT_WORDS):
                    return True, 2
                    
        has_loc = any(w.deprel in ['obl', 'advmod'] and any(loc_word in w.text.lower() or has_location_preposition(sentence, w.id) for loc_word in ['in', 'on', 'at', 'page', 'screen', 'window']) for w in sentence.words)
        has_tmp = any(w.deprel in ['obl', 'advmod'] and any(tmp_word in w.text.lower() or has_time_preposition(sentence, w.id) for tmp_word in ['during', 'when', 'while', 'after', 'before']) for w in sentence.words)
        
        if has_loc and has_tmp: return True, 2
        elif has_loc or has_tmp: return True, 1
    return False, 0

def check_RA3_system_defect(doc, text):
    text_lower = text.lower()
    if any(defect in text_lower for defect in SYSTEM_DEFECT_WORDS):
        return True, 2
        
    for sentence in doc.sentences:
        predicates = [word for word in sentence.words if word.upos == 'VERB']
        for verb in predicates:
            has_negation = any(w.deprel == 'advmod' and w.text.lower() in ['not', "n't", 'never', 'no'] for w in sentence.words if w.head == verb.id)
            if has_negation:
                system_actions = ["load", "display", "show", "appear", "refresh", "update", "work", "function"]
                is_system_action = any(action in verb.lemma.lower() for action in system_actions)
                has_interface_object = any(w.deprel == 'obj' and any(elem in w.lemma.lower() for elem in INTERFACE_ELEMENT_WORDS) for w in sentence.words if w.head == verb.id)
                
                if is_system_action or has_interface_object: return True, 2
                else: return True, 1
    return False, 0

def check_RA4_defect_description_quality(doc, text):
    has_defect, _ = check_RA3_system_defect(doc, text)
    if not has_defect: return False, 0
    
    relation_mapping = {
        'ATT': ['amod', 'nmod', 'nummod', 'det', 'compound', 'case', 'mark'],
        'ADV': ['advmod', 'advcl', 'obl', 'neg'],
        'CMP': ['obj', 'iobj', 'xcomp', 'ccomp', 'acl'],
        'COO': ['conj', 'cc'],
        'LAD': ['acl:relcl', 'acl'],
        'RAD': ['appos', 'parataxis']
    }
    relation_lookup = {ud_rel: rel_type for rel_type, ud_rels in relation_mapping.items() for ud_rel in ud_rels}
    max_relation_score = 0
    
    for sentence in doc.sentences:
        defect_words = [word for word in sentence.words if word.lemma.lower() in SYSTEM_DEFECT_WORDS]
        for word in sentence.words:
            if word.upos == 'VERB' and any(w.deprel == 'advmod' and w.text.lower() in ['not', "n't", 'never', 'no'] for w in sentence.words if w.head == word.id):
                defect_words.append(word)
                
        if not defect_words: continue
        
        for defect_word in defect_words:
            relations = {'ATT': 0, 'ADV': 0, 'CMP': 0, 'COO': 0, 'LAD': 0, 'RAD': 0}
            for other_word in sentence.words:
                if other_word.head == defect_word.id:
                    rel_type = relation_lookup.get(other_word.deprel, None)
                    if rel_type: relations[rel_type] += 1
                    
            if defect_word.head != 0:
                rel_type = relation_lookup.get(defect_word.deprel, None)
                if rel_type: relations[rel_type] += 1
                
            total_relation_count = sum(relations.values())
            relation_score = 2 if total_relation_count >= 4 else (1 if total_relation_count >= 2 else 0)
            max_relation_score = max(max_relation_score, relation_score)
            
    return max_relation_score > 0, max_relation_score

# ============= MAIN EVALUATION =============
def evaluate_bug_report(text):
    """Evaluate the overall quality of a bug report."""
    # 무한루프 및 OOM 방지 (매우 긴 에러 로그 컷)
    if len(text) > 10000:
        text = text[:10000]

    # 🚨 가장 중요한 부분: nlp 객체를 1번만 안전하게 가져와서 1번만 분석합니다.
    nlp = get_nlp()
    doc = nlp(text)
    
    results = {
        "RM1_size": check_RM1_size(text),
        "RM2_readability": check_RM2_readability(text),
        "RM3_punctuation": check_RM3_punctuation(text),
        "RM4_sentence_length": check_RM4_avg_sentence_length(doc),
        
        "RR1_itemization": check_RR1_itemization(text),
        "RR2_itemization_symbol": check_RR2_itemization_symbol(text),
        "RR3_environment": check_RR3_environment(text),
        "RR4_screenshot": check_RR4_screenshot(text),
        "RR5_screenshot_guideline": check_RR5_screenshot_guideline(text),
        
        # doc을 파라미터로 넘겨주어 속도를 극대화합니다.
        "RA1_interface_element": check_RA1_interface_element(doc, text),
        "RA2_user_behavior": check_RA2_user_behavior(doc, text),
        "RA3_system_defect": check_RA3_system_defect(doc, text),
        "RA4_defect_description": check_RA4_defect_description_quality(doc, text)
    }
    
    total_score = sum(score for _, score in results.values())
    max_possible = 16
    
    return {
        "detail_scores": results,
        "total_score": total_score,
        "max_possible": max_possible
    }

def process_excel_file(excel_path, output_path=None,output_prefix="bug_report_scores"):
    try:
        df = pd.read_excel(excel_path)
        print(f"Number of rows read from Excel: {len(df)}") 
    except Exception as e:
        print(f"Error loading Excel file: {e}")
        return None
    
    text_column = next((col for col in ['4o_gpt Output'] if col in df.columns), df.columns[0] if len(df.columns) > 0 else None)
    if text_column is None:
        print("No suitable text column found in the Excel file")
        return None
    
    results = []
    file_count = 1
    for idx, row in df.iterrows():
        report_text = str(row[text_column])
        evaluation = evaluate_bug_report(report_text)
        
        result = dict(row)
        result['total_score'] = evaluation['total_score']
        result['max_possible'] = evaluation['max_possible']
        result['score_percentage'] = (evaluation['total_score'] / evaluation['max_possible']) * 100
        
        for rule, (passed, score) in evaluation['detail_scores'].items():
            result[f'{rule}_passed'] = passed
            result[f'{rule}_score'] = score

        if (idx + 1) % 1000 == 0 or idx == len(df) - 1:
            results_df = pd.DataFrame(results)
            output_dir = "Original_dataset"
            os.makedirs(output_dir, exist_ok=True)
            output_file = f"./Original_dataset/{output_prefix}_score_here_all_12k_{file_count}.xlsx"
            results_df.to_excel(output_file, index=False)
            results.clear()
            file_count += 1
        
        results.append(result)
    
    results_df = pd.DataFrame(results)
    if output_path:
        results_df.to_excel(output_path, index=False)
        print(f"Results saved to {output_path}")
    
    return results_df

if __name__ == "__main__":
    input_file = "/mnt/c/Users/selab/Ease_2025_AI_model/Evaluation/CTQRS_200_Score_test_llama_Lora.xlsx"
    output_file = "/mnt/c/Users/selab/Ease_2025_AI_model/Evaluation/TEst_CTQRS_llama_Lora_bug_report_scores.xlsx"
    
    result = process_excel_file(input_file, output_file)
        
    if result is not None:
        print(f"Processed {len(result)} bug reports.")
        print(f"Average score: {result['score_percentage'].mean():.2f}%")