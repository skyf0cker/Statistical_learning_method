from naive_bayes import NaiveBayes

def test_init():
    n = NaiveBayes(2,8)
    return n

def test_fit(n):
    n.fit()
    print(n.mean)
    print(n.var)

def test_predict(n):
    n.predict([0.5, 0.7])

if __name__ == "__main__":
    n = test_init()
    test_fit(n)
    test_predict(n)