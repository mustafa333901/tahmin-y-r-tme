with open('predictions.csv', 'a', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([input_value, prediction])
