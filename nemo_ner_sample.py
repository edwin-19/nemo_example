from nemo.collections import nlp as nemo_nlp

if __name__ == "__main__":
    pretrained_ner_model = nemo_nlp.models.TokenClassificationModel.from_pretrained(model_name="ner_en_bert")
    
    # define the list of queries for inference
    queries = [
        'we bought four shirts from the nvidia gear store in santa clara.',
        'Nvidia is a company.',
        'The Adventures of Tom Sawyer by Mark Twain is an 1876 novel about a young boy growing '
        + 'up along the Mississippi River.',
    ]
    results = pretrained_ner_model.add_predictions(queries)

    for query, result in zip(queries, results):
        print()
        print(f'Query : {query}')
        print(f'Result: {result.strip()}\n')
        