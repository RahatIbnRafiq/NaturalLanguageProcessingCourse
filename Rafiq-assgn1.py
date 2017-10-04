outputFileName = "Rafiq-out-assgn1.txt"
testAnswersFileName = "testGoldenAnswers-2015.txt"
wordListFileName = "bigwordlist.txt"
hashtagsFileName = "hashtags-test-2015.txt"


def substCost(x,y):
    if x == y: 
        return 0
    else: 
        return 1
    
def  minEditDist(target, source):
    ''' Computes the min edit distance from target to source. Figure 3.25 '''
    
    n = len(target)
    m = len(source)

    distance = [[0 for i in range(m+1)] for j in range(n+1)]

    for i in range(1,n+1):
        distance[i][0] = distance[i-1][0] + 1

    for j in range(1,m+1):
        distance[0][j] = distance[0][j-1] + 1

    for i in range(1,n+1):
        for j in range(1,m+1):
           distance[i][j] = min(distance[i-1][j]+1,
                                distance[i][j-1]+1,
                                distance[i-1][j-1]+substCost(source[j-1],target[i-1]))
    return distance[n][m]


def computeAverageWERDistance(targetFile,sourceFile):
    f = open(targetFile,"r")
    g = open(sourceFile,"r")
    totalDistances = 0.0
    totalWordCount = 0

    for line1 in f:
        line1 = line1.strip().lower()
        line2 = g.readline().strip().lower()
        distance = minEditDist(line1,line2)
        words = line2.split()
        totalDistances = totalDistances + distance
        totalWordCount = totalWordCount + len(words)
    f.close()
    g.close()
    return (totalDistances/totalWordCount) # total edits made divided by total words count in the source


def loadWordsFromGoogleWordsList(filename):
    f = open("bigwordlist.txt","r")
    data = []
    for line in f:
        line = line.strip()
        wordFrequencyPair = line.split()
        data.append(wordFrequencyPair)
    f.close()
    return dict(data)



def matchWithWordList(word,wordList):
    for googleWord in wordList:
        if word == googleWord:
            return word
    return -1

def modifiedMatchWithWordList(word,wordList,length):
    for googleWord in wordList:
        if word == googleWord and len(word) <= length:
            return word
    return -1


def maxMatchHastag(hashtag,wordList):
    result = ""
    index1 = 0
    index2 = len(hashtag)
    hashtag = hashtag.strip()
    while index2 >= 0:
        subStr = hashtag[index1:index2]
        if len(subStr) == 0:
            break
        if matchWithWordList(subStr,wordList) == -1:
            index2 = index2-1
        else:
            if len(result) == 0:
                result = subStr
            else:
                result = result + " "+ subStr
            index1 = index2
            index2 = len(hashtag)
    if index1 < len(hashtag):
        result = result + " "+ hashtag[index1:]
    return result


def maximumMatchingStart(filename1,wordList,filename2):
    # starting the maximum matching algorithm
    f = open(filename1,"r")
    g = open(filename2,"w")
    for hashtag in f:
        hashtag = hashtag.strip()[1:]
        g.write(str(maxMatchHastag(hashtag,wordList))+"\n")
    f.close()
    g.close()


def showOutputs(filename1,filename2):
    f = open(filename1,"r")
    g = open(filename2,"r")

    for line1 in f:
        line1 = line1.strip()
        line2 = g.readline()
        line2 = line2.strip()
        print line1
        print line2
        print "____________________________________"


    f.close()
    g.close()
    




def alteringLexicon(wordList):
    #reference: mos common two and three letter words in english->http://scottbryce.com/cryptograms/stats.htm
    twoLetterWords = ["of", "to", "in", "it", "is", "be", "as", "ja","at"
                      , "so", "we", "he", "by", "or", "on", "do", "if", "me", "my", "up",
                      "go", "no", "us", "am","vs","u","a"]
    
    threeLetterWords = ["the", "and", "for", "are", "but", "not", "you", "all", "any", "can", "had", "her", "was", "one", "art","ski",
                        "our", "out", "day", "get", "has", "him", "his", "tha","how", "man","ipa","nbc","sag","may","pit",
                         "new", "now", "old", "see", "two", "way", "cnn","gop","bbc","who", "boy", "did", "its", "let", "put", "say", "she", "too", "use","atl"]
    
    newWordList  = []
    
    for key in wordList:
        if len(key) > 1:
            if len(key) == 2:
                if key in twoLetterWords:
                    newWordList.append([key,wordList[key]])
            elif len(key) == 3:
                if key in threeLetterWords:
                    newWordList.append([key,wordList[key]])
            else:
                newWordList.append([key,wordList[key]])
    newWordList = dict(newWordList)
    # most used prepositions and articles and word composites->https://www.englishclub.com/vocabulary/common-prepositions-25.htm
    composites = ["of","in","the","with","on","at","by","about","to","from","be","vs","is","real","for","i","won"]

    for w1 in composites:
        for w2 in composites:
            if w1 != w2:
                word = str(w1)+str(w2)
                try:
                    newWordList.pop(word)
                except Exception:
                    continue

    print "previous word list size : "+str(len(wordList))
    print "now word list size : "+str(len(newWordList))
    return newWordList
                
    


#part 1


        

def modifiedMaxMatchHastag1(hashtag,wordList,longest):
    allMatches = []
    i = 1
    allMatches1 = []
    while i <= longest:
        result = []
        result1 = ""
        index1 = 0
        index2 = len(hashtag)
        hashtag = hashtag.strip()
        while index2 >= 0:
            subStr = hashtag[index1:index2]
            if len(subStr) == 0:
                break
            if modifiedMatchWithWordList(subStr,wordList,i) == -1:
                index2 = index2-1
            else:
                result1 = result1+" "+subStr
                result.append(subStr)
                index1 = index2
                index2 = len(hashtag)
        if index1 < len(hashtag):
            result.append(hashtag[index1:])
            result1 = result1+" "+hashtag[index1:]
        i=i+1
        
        if result1 in allMatches1:
            continue
        allMatches1.append(result1)
        allMatches.append(result)
    return allMatches





def modifiedMaxMatchHastag(hashtag,wordList,longest):
    """
    we do the modified maximum matching algorithm twice. divide the hashtag into two pieces where
    the starting point starts from 0 to the length of the word and then find out the longest possible matches for the two
    paritions. if there are any extra words after that long word then add those to one token.
    then take the match that had totalTokenCount-totalTokenCount number lowest. 
    """
    length = len(hashtag)
    allMatches = []
    for index in range(0,length):
        matchedWords = []
        result1 = maxMatchHastag(hashtag[0:index],wordList)
        result2 = maxMatchHastag(hashtag[index:],wordList)
        result1 =  result1.split() 
        result2 =  result2.split()
        for word in result1:
            matchedWords.append(word)
        for word in result2:
            matchedWords.append(word)
        allMatches.append(matchedWords)
    return allMatches
    


def findDifferenceOfTokensLength(match):
    maxLength = 0
    minLength = 100
    for word in match:
        if len(word) > maxLength:
            maxLength = len(word)
        if len(word) < minLength:
            minLength = len(word)
    return (maxLength-minLength)

def compareMatches(allMatches,wordList):
    counts = []
    min = 100
    result = None
    minWordCount = 100
    for match in allMatches:
        totalCount = len(match)
        wordCount = 0
        for word in match:
            if matchWithWordList(word,wordList) != -1:
                wordCount = wordCount + 1
        if (totalCount-wordCount) < min:
            result = match
            min = totalCount-wordCount
            minWordCount = wordCount
        elif (totalCount-wordCount) == min:
            if wordCount < minWordCount:
                result = match
                min = totalCount-wordCount
                minWordCount = wordCount
            elif wordCount == minWordCount:
                report = checkWhenTie(result,match,wordList)
                if report == "change":
                    result = match
                    min = totalCount-wordCount
                    minWordCount = wordCount                    
                
                
    return result
                 

def checkWhenTie(previousMatch, recentMatch,wordlist):
    index = 0
    len1 = len(previousMatch)
    len2 = len(recentMatch)
    for word in previousMatch:
        if word.endswith("s") and index < len1-1:
            w1 = "s"+ previousMatch[index+1]
            if matchWithWordList(w1, wordlist):
                return "change"
        elif word.endswith("d") and index < len1-1:
            w1 = "d"+ previousMatch[index+1]
            if matchWithWordList(w1, wordlist):
                return "change"
        elif word.endswith("a") and index < len1-1:
            w1 = "a"+ previousMatch[index+1]
            if matchWithWordList(w1, wordlist):
                return "change"
        index = index + 1
    return "nothange"


def modifiedMaximumMatchingStart(filename1,wordList,filename2,longest):
    f = open(filename1,"r")
    g = open(filename2,"w")
    for hashtag in f:
        toBeWritten = ""
        hashtag = hashtag.strip()[1:].lower()
        allMatches =  modifiedMaxMatchHastag(hashtag,wordList,longest)
        result = compareMatches(allMatches,wordList)
        for word in result:
            toBeWritten = toBeWritten +" "+word
        toBeWritten = toBeWritten[1:].strip() 
        print toBeWritten 
        g.write(str(toBeWritten)+"\n")
    f.close()
    g.close()


print "##############################################################################################"
print "part 1 works have been started."
print "______________________________________________________________________________________________"

print "step1: loading the dictionary from google words."
print "loading google words"
googleWordList = loadWordsFromGoogleWordsList(wordListFileName)
print "loading of google words is done. size of the list is :"+str(len(googleWordList))
print "______________________________________________________________________________________________"

print "step2: applying traditional maximum matching algorithm.."
print "traditional maximum matching algorithm is started."
maximumMatchingStart(hashtagsFileName,googleWordList,outputFileName)
print "traditional maximum matching algorithm is finished."
print "______________________________________________________________________________________________"


print "______________________________________________________________________________________________"
print "step3. analyzing the behavior of the traditional maximum matching algorithm"
print "now analyzing the behavavior"
print "1.absense of some words like cuboulder and iphone6 in the dictionary"
print "2.there are some words in the dictionary that influnces the result like ue, isa, inn"
print "3.there are some words that comprises of two or more words like forthe-> for the, superbowl->super bowl"
print "4.sometimes the algorithm, in the process to match the maximum length words, in the tail leaves letters that couldn't be matched, like vsat l, ol l"
print "5. only two one letter words have been used. they are u and i. Other one letter words can be removed."
print "______________________________________________________________________________________________"


print "part 1 works have been finished."
print "##############################################################################################"







#part 2
print "##############################################################################################"
print "part 2 works have been started."
print "______________________________________________________________________________________________"
print "step1:computation of average WER by applying traditional maximum matching algorithm"
averageWERDistance =  computeAverageWERDistance(outputFileName,testAnswersFileName)
print "average WER distance for the traditional maximum matching algorithm is :" + str(averageWERDistance)
print "______________________________________________________________________________________________"
print "part 2 works have been finished."
print "##############################################################################################"



print "##############################################################################################"
print "______________________________________________________________________________________________"
print "part 3 has been started."

print "______________________________________________________________________________________________"
print "step 1. changing the lexicon."
newWordList = alteringLexicon(googleWordList)
print "______________________________________________________________________________________________"

longest = len(max(newWordList,key=len))
print "______________________________________________________________________________________________"
print "step 2. modifying the greedy strategy of the maximum matching algorithm."
modifiedMaximumMatchingStart(hashtagsFileName,newWordList,outputFileName,longest)

averageWERDistance =  computeAverageWERDistance(outputFileName,testAnswersFileName)
print "average WER distance after second improvement :" + str(averageWERDistance)
print "______________________________________________________________________________________________"








print "Assignment one has been finished. Thanks!"
