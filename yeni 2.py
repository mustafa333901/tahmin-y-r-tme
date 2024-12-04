if not os.path.exists('model_data.json'):
    with open('model_data.json', 'w') as f:
        json.dump({"X": [[1], [2], [3], [4], [5]], "y": [2, 4, 6, 8, 10]}, f)
