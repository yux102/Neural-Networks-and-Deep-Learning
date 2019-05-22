
def load_data(filename="tempReview.txt"):
    data = list()
    with open(filename, 'r') as f:
        for review in f.readlines():
            data.append(review.split(" "))
    return data

print (len(load_data()))
