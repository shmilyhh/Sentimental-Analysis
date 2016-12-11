def negative_sequence(text):
    negation = False
    delims = "?.,!:;"
    result = []
    words = text.split()
    prev = None
    pprev = None
    for word in words:
        stripped = word.strip(delims).lower()
        negated = "not_" + stripped if negation else stripped
        result.append(negated)
        if prev:
            bigram = prev + " " + negated
            result.append(bigram)
            if pprev:
                trigram = pprev + " " + bigram
                result.append(trigram)
            pprev = prev
        prev = word
        
        if any(neg in word for neg in ["not", "n't", "no"]):
            negation = not negation
        
        if any(c in word for c in delims):
            negation = False
            
    return result

text = "I am not happy, and I am not feeling well."
print negative_sequence(text)