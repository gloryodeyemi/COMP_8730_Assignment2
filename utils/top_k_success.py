import pytrec_eval as pt
from operator import itemgetter

# function to return top-k most probable list of tokens
def top_k_tokens(data_row, model, tokenized_corpus):
    cor_word = data_row[1]
    incor_word = data_row[0]
    sent_to_predict = data_row[2:]
    sent_to_predict = ' '.join(sent_to_predict).split('*')
    sent_to_predict = sent_to_predict[0]

    l_model_result = {'correct_word':cor_word, 'incorrect_word':incor_word}
    probs_dict = {}
    for sentence in tokenized_corpus:
        for token in sentence:             
            probability = model.score(token, sent_to_predict.split())
            probs_dict[token] = probability
        
    probs_dict = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
    for k in [1,5,10]:
        probs_dict_res = dict(list(probs_dict.items())[:k])
        l_model_result['top_' + str(k)] = probs_dict_res
    
    return l_model_result

# function to check for the success at k
def check_insert(k_list, correct_word):
    if correct_word in k_list:
        return 1
    else:
        return 0


# function to return the success at k for top-k most probable word returned
def success_at_k(top_k_result):
    suc_at_k_dict = {}
    result_dict = {}
    for item in top_k_result:
        correct_word, incorrect_word, top_1, top_5, top_10 = itemgetter('correct_word', 'incorrect_word',
                                                                        'top_1', 'top_5', 'top_10')(item)
    
        result_dict['success_at_1'] = check_insert(top_1, correct_word)
        result_dict['success_at_5'] = check_insert(top_5, correct_word)
        result_dict['success_at_10'] = check_insert(top_10, correct_word)

        suc_at_k_dict[incorrect_word] = result_dict
        result_dict = {}

    return suc_at_k_dict


# function calculate average success at k for k={1, 5, 10} using PyTrec_Eval_Terrier
def average_k(success_dict):
    average_dict = {}
    for k_success in success_dict[list(success_dict.keys())[0]].keys():
        average_dict[k_success] = pt.compute_aggregated_measure(
                                  k_success, [val[k_success] for val in success_dict.values()])
    return average_dict

