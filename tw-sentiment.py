from transformers import AutoTokenizer,AutoModelForSequenceClassification
from scipy.special import softmax

#tweet = "@Samik This is a test analysis on tweet @ home 😒 https://hello.com"
tweet =input("Enter a tweet:  ")
#@MIKXY
#preprocess tweet
tweet_words = []
print()
for word in tweet.split(' '):
    if word.startswith('@') and len(word) > 1:           #checking for user tags (checking if the @symbol is just a symbol or a user tag)
        word = '@user'
    elif word.startswith('http'):      #checking if the word is a link(starting with http)(!!!!!!check later for modifying)
        word = "http"
    tweet_words.append(word)
    
tweet_proc = " ".join(tweet_words)
print ("Extracting words present in the tweet: ")
print(tweet_words)
print()
print("Merging processed words in the tweet: ")
print(tweet_proc)
print()

#load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"        #importing roberta model from huggingface"https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)  #sequence classification
tokenizer = AutoTokenizer.from_pretrained(roberta)                   #
labels = ['Negative', 'Neutral', 'Positive']                         #types of emotion we can derive from tweets(Negetive,Neutral,Positive)


#sentiment analysis
#first we need to convert the tweet to pytorch tensors and then we pass that into the model

encoded_tweet = tokenizer(tweet_proc, return_tensors='pt')
print("Displaying the encoded tweet:")
print(encoded_tweet)
print()
output = model(encoded_tweet['input_ids'],encoded_tweet['attention_mask'])
#output = model(**encoded_tweet)
print("After converting the encoded tweet in tensor values:")
print(output)
print()
#convert the tensorvalues into probablities using softmax
scores = output[0][0].detach().numpy()
scores = softmax(scores)
print("After converting the tensor values into probablity scores using softmax:")
print(scores)
print()
#printing output in the corresponding labels
print("Mapping the scores returned according to the labels[NEGATIVE, NEUTRAL, POSITIVE]")
for i in range(len(scores)):
    l = labels[i]
    s = scores[i]
    print(l,s)
    
resultant_dict = {labels[i]: scores[i] for i in range(len(labels))}
print('Dictionary:', resultant_dict)

max_value = list(resultant_dict.values())
max_key= list(resultant_dict.keys())
print("The emotion of this tweet is: ",max_key[max_value.index(max(max_value))])