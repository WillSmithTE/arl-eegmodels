import csv

def writeRow(val):
    with open('results.csv', 'a', newline='') as csvfile:
        spamwriter = csv.writer(csvfile)
        spamwriter.writerow(val)
