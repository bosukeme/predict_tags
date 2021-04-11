import pandas as pd
import re
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import operator
from sklearn.svm import SVC

df = pd.read_csv("mind_sharing1.csv")


df = df.reset_index(drop=True)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def clean_text(text):
    """
        text: a string
        return: modified initial string
    """
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = BAD_SYMBOLS_RE.sub('', text) # remove symbols which are in BAD_SYMBOLS_RE from text. substitute the matched string in BAD_SYMBOLS_RE with nothing. 
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # remove stopwors from text
    return text
df['body'] = df['body'].apply(clean_text)



#######################################################################################

def depression_predictions(df, new_text):
    
    #one hot encoding
    code = []
    for tag in df['tags']:
        if 'depression' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code

    # select the columns we need
    df = df[['body', 'code']]
    
    # split into test and train
    Y = df['code'].values
    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)

    # transform texts to arrays
    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)
    
    
    #build model
    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    # transform new text to array
    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    # get the prediction probability of the new text
    depression_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return depression_score

#######################################################################

def anxiety_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'anxiety' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code

    df = df[['body', 'code']]
    Y = df['code'].values
    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)


    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    anxiety_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return anxiety_score

#######################################################################

def grief_loss_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'grief & loss' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    grief_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return grief_score

###################################################################

def self_esteem_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'self esteem & confidence' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    self_esteem_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return self_esteem_score

##################################################################################

def stress_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'stress' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    stress_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return stress_score

####################################################################

def relationship_issues_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'relationship issues' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    relationship_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return relationship_score

############################################################

def work_professional_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'work/professional/career' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    work_score = list(classifier.predict_proba(transform_text)[:,0])[0]

    return work_score

#################################################################

def trauma_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'trauma' in tag:
            tag = 0
            code.append(tag)
        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    trama_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return trama_score

####################################################################

def abuse_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'abuse' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    abuse_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return abuse_score

###########################################################################3

def anger_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'anger' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    anger_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return anger_score

#################################################################################

def sexuality_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'sexuality' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    sexuality_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return sexuality_score

########################################################################

def eating_disorder_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'eating disorder' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    eating_disorder_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return eating_disorder_score

###############################################################

def addictions_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'addictions' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    addictions_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return addictions_score

#######################################################################

def sex_related_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'sex-related' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    sex_related_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return sex_related_score

#########################################################################################

def adhd_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'ADHD' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)


    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    adhd_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return adhd_score

    
##############################################################################

def sleeping_disorders_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'sleeping disorders' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)

    
    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    sleeping_disorders_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return sleeping_disorders_score


#######################################################################

def others_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'others' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)


    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    others_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return others_score

#######################################################################

def suicidal_predictions(df, new_text):
    code = []
    for tag in df['tags']:
        if 'suicidal' in tag:
            tag = 0
            code.append(tag)

        else:
            tag = 1
            code.append(tag)
    df['code'] = code


    df = df[['body', 'code']]
    
    
    Y = df['code'].values

    df_train, df_test, Ytrain, Ytest = train_test_split(df['body'], Y, test_size=0.33, random_state=0)


    tfidf = TfidfVectorizer(decode_error='ignore')
    Xtrain = tfidf.fit_transform(df_train)


    svm = SVC(random_state= 1, probability=True)
    classifier = OneVsRestClassifier(svm)
    classifier.fit(Xtrain, Ytrain)

    text_series = pd.Series([new_text])
    transform_text = tfidf.transform(text_series)

    
    suicidal_score = list(classifier.predict_proba(transform_text)[:,0])[0]
    
    return suicidal_score



def process_all_predictions(new_text):
    depression_score = depression_predictions(df, new_text)
    anxiety_score = anxiety_predictions(df, new_text)
    grief_score = grief_loss_predictions(df, new_text)
    self_esteem_score = self_esteem_predictions(df, new_text)
    stress_score = stress_predictions(df, new_text)
    relationship_score = relationship_issues_predictions(df, new_text)
    work_profession_score = work_professional_predictions(df, new_text)
    trauma_score = trauma_predictions(df, new_text)
    abuse_score = abuse_predictions(df, new_text)
    anger_score = anger_predictions(df, new_text)
    sexuality_score = sexuality_predictions(df, new_text)
    eating_disorder_score = eating_disorder_predictions(df, new_text)
    addictions_score = addictions_predictions(df, new_text)
    sex_related_score = sex_related_predictions(df, new_text)
    adhd_score = adhd_predictions(df, new_text)
    sleeping_disorder_score = sleeping_disorders_predictions(df, new_text)
    others_score = others_predictions(df, new_text)
    suicidal_score = suicidal_predictions(df,new_text)

    
    
    items_scores = {"depression_score": depression_score, "anxiety_score": anxiety_score,
                   "grief_score":grief_score, "self_esteem_score": self_esteem_score,
                    "stress_score":stress_score, "relationship_score": relationship_score,
                   "work_profession_score": work_profession_score, "trauma_score":trauma_score,
                   "abuse_score":abuse_score, "anger_score":anger_score, "sexuality_score": sexuality_score,
                   "eating_disorder_score":eating_disorder_score, "addictions_score":addictions_score,
                   "sex_related_score": sex_related_score, "adhd_score":adhd_score,
                    "sleeping_disorder_score":sleeping_disorder_score, "others_score":others_score,
                    "suicidal_score": suicidal_score}
    
    
    sorted_score = dict(sorted(items_scores.items(), key=operator.itemgetter(1),reverse=True))
    
    text_and_score = {new_text : [sorted_score]}
    
    
    return text_and_score
