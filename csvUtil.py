import csv

def writeRow(val, output = 'results.csv'):
    with open(output, 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(val)
