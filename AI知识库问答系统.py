from flask import Flask, request, jsonify, render_template
from transformers import AutoTokenizer, AutoModelWithLMHead, MT5ForConditionalGeneration
from elasticsearch import Elasticsearch
from bs4 import BeautifulSoup
from torch.nn.functional import softmax
import requests
import torch

es = Elasticsearch(hosts="http://localhost:9200")
tokenizer = AutoTokenizer.from_pretrained("/Users/administrator/nlpmodels/huggingface-T5QA")
model = MT5ForConditionalGeneration.from_pretrained("/Users/administrator/nlpmodels/huggingface-T5QA")


def get_response(input_text):
    query = {
        "query": {
            "bool": {
                "should": [
                    {
                        "multi_match": {
                            "query": input_text,
                            "fields": ["content", "caption"],
                            "boost": 2.0,
                            "fuzziness": "AUTO",
                            "analyzer": "custom_analyzer"
                        }
                    },
                    {
                        "match_phrase_prefix": {
                            "content": {
                                "query": input_text,
                                "boost": 1.5,
                                "analyzer": "custom_analyzer"
                            }
                        }
                    }
                ],
                "minimum_should_match": 1
            }
        },
        "size": 10,
        "_source": ["id", "caption", "content"],
        "highlight": {
            "fields": {
                "content": {}
            }
        }
    }

    res = es.search(index='cupdkb_index', body=query)
    es_results = []
    for hit in res['hits']['hits']:
        source = hit["_source"]
        document_id = hit["_id"]
        document_caption = source["caption"].replace("\n", "")
        highlight = hit.get("highlight", {})
        content_highlight = highlight.get("content", [])
        content_snippet = content_highlight[0] if content_highlight else source["content"]
        content_snippet = BeautifulSoup(content_snippet, "html.parser").get_text()

        es_results.append({
            "id": document_id,
            "caption": document_caption,
            "content": content_snippet
        })

    answers = []

    if es_results:
        for es_result in es_results:
            task_specific_input = f"question: {input_text} context: {es_result['content']}"
            inputs = tokenizer.encode_plus(task_specific_input, return_tensors='pt')
            outputs = model.generate(inputs['input_ids'], max_length=512, temperature=0.7, no_repeat_ngram_size=2,
                                     return_dict_in_generate=True, output_scores=True)
            decoded_output = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            answer = decoded_output.split('<eos>')[0]
            scores = outputs.scores[0]
            probabilities = softmax(scores, dim=-1)
            avg_probability = probabilities.mean().item()
            answers.append({"id": es_result["id"], "caption": es_result["caption"], "answer": answer,
                            "probability": avg_probability})

        answers.sort(key=lambda x: x['probability'], reverse=True)
        top_answers = answers[:3]

        ###############################
        print('----------es_results: Found-------------')
        print({
            'status': 'found',
            'answers': [{'id': answer['id'], 'caption': answer['caption'], 'answer': answer['answer']} for answer in
                        top_answers]
        })
        ###############################

        return {
            'status': 'found',
            'answers': [{'id': answer['id'], 'caption': answer['caption'], 'answer': answer['answer']} for answer in
                        top_answers]
        }
    else:
        inputs = tokenizer.encode(input_text, return_tensors='pt')
        outputs = model.generate(inputs, max_length=512, num_return_sequences=1, temperature=0.7,
                                 no_repeat_ngram_size=2)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        ###############################
        print('----------es_results: Not Found-------------')
        print({
            'status': 'not_found',
            'response': response
        })
        ###############################

        return {
            'status': 'not_found',
            'response': response
        }

    print('----------es_results: Error-------------')
    print({
        'status': 'error',
        'message': 'Unexpected error occurred.'
    })

    return {
        'status': 'error',
        'message': 'Unexpected error occurred.'
    }


app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/get_response', methods=['POST'])
def get_flask_response():
    data = request.get_json()
    user_input = data['input_text']
    try:
        response = get_response(user_input)
        return jsonify(response)
    except Exception as e:
        print(e)
        return jsonify({"status": "error", "message": "Unexpected error occurred."})


@app.route('/api/get_detail', methods=['POST'])
def get_detail():
    data = request.get_json()
    id = data['id']
    # 根据ID获取详细内容
    detail = get_detail_content(id)  # 自定义函数，根据ID获取详细内容

    if detail:
        return jsonify({'status': 'found', 'detail': detail})
    else:
        return jsonify({'status': 'not_found'})


@app.route('/detail')
def show_detail():
    id = request.args.get('id')
    detail = get_detail_content(id)  # 自定义函数，根据ID获取详细内容

    if detail:
        return render_template('detail.html', detail=detail)
    else:
        return render_template('detail.html', detail=None)


def get_detail_content(id):
    # 使用Elasticsearch查询根据ID获取详细内容
    query = {
        "query": {
            "ids": {
                "values": [id]
            }
        },
        "_source": ["id", "caption", "content"]
    }

    res = es.search(index='cupdkb_index', body=query)

    if res['hits']['total']['value'] > 0:
        hit = res['hits']['hits'][0]
        source = hit["_source"]
        detail = {
            "id": source["id"],
            "caption": source["caption"],
            "content": source["content"]
        }
        return detail
    else:
        return None


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9898)
