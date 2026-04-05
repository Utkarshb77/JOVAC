# Q1:
l = [2,4,6,7,7,8,7,6]
max1 = 0
max2 = 0
for i in l:
    if i>max1:
        max2 = max1
        max1 = i
print(max2)

# Q2:
l.sort()
unique = []
for i in l:
        if i not in unique:
            unique.append(i)
print(unique)

# Q3:
sum = 0
for i in l:
      sum=sum+i
print(sum)
avg = sum/len(l)
print(avg)

# Q4
def prime(n):
    if n <= 1:
        return False 
    for i in range(2, n):
        if n%i==0:
            return False
    return True

# Q5:
def counts(s):
    vowels = 'aeiouAEIOU'
    v = c = d = sc = 0
    for ch in s:
        if(ch in vowels):
            v+=1
        elif ch.isalpha():
            c+=1
        elif ch.isdigit():
            d+=1
        else:
            sc+=1
    return v,c,d,sc
# print(counts("hii! its me:Ub77"))

# Q6:

for i in range(1 , 101):
    if i % 2 == 0:
        print(i)

# Q7:
# num = int(input("enter a number:"))
def table(num):
    i = 1
    while i<=10:
        print(num*i)
        i+=1
# table(7)

# Q8:
ls = []
for i in range(1 , 101):
    if(i%3==0 and i%5==0):
        ls.append(i)
# print(ls)

# Q9:
# n = int(input("enter number:"))
def reverse(n):
    a =  0
    while(n > 0):
        b = n%10
        n = n//10
        a=a*10+b
    return a

# Q10:
dict = {}
s = "hello world"
for i in s:
    if i in dict:
        dict[i] +=1
    else:
        dict[i] = 1
# print(dict)

# Q11:
def allprimes(n):
    primes=[]
    for i in range(2,n):
        if(prime(i) == True):
            primes.append(i)
    return primes

# print(allprimes(100))

# Q12:
def ispalin(n):
    return str(n) == str(n)[::-1]

# Q13:
l = [1,2,7,7,2,4,5,5,7,7,7]
def count(l , tar):
    c=0
    for i in l:
        if i == tar:
            c+=1
    return c
# print(count(l , 7))

# Q14:
def allevensq(n):
    l=[]
    for i in range(1,n+1):
        if(i%2==0):
            l.append(i*i)
    return l
# print(allevensq(50))

# Q15:
l = [1,8,2,4,5,6,7,7,7,2,2,1,1,6,6,5]
def uni(l):
    ls=[]
    for i in l:
        if i not in ls:
            ls.append(i)
    return ls
# print(uni(l))

# Q16:
def oddeven(n):
    if(n%2==0):
        return 'even'
    else:
        return 'odd'
    
# Q17:
def summ(l):
    sum = 0
    for i in l:
      sum=sum+i
    return sum
l = [1,2,4,5,7]
# print(summ(l))

# Q18:
def fact(n):
    if n == 0 or n == 1:
        return 1
    return n * fact(n-1)
# print(fact(7))

# Q19:
def fib(n):
    l = []
    a=0
    b=1
    for _ in range(n):
        l.append(a)
        a,b=b,a+b 
    return l
print(fib(7))

# Q20:
def is_palindrome_string(s):
    return s == s[::-1]

# Q21:
def maxmin(l):
    return max(l), min(l)

# Q22:
def is_pangram(s):
    s = s.lower()
    count = 0
    for char in 'abcdefghijklmnopqrstuvwxyz':
        if char in s:
            count += 1
    return count == 26

# Q23:
def primesrange(s, e):
    primes = []
    for n in range(s,e+1):
        if n<2:
            continue
        for i in range(2, int(n**0.5)+1):
            if i % i == 0:
                break
        else:
            primes.append(n)
    return primes

# Q24:
def countcase(s):
    upper = lower = 0
    for ch in s:
        if ch.isupper():
            upper += 1
        elif ch.islower():
            lower += 1
    return upper, lower

# Q25:
def sumofdigits(n):
    total=0
    while n>0:
        total+=n%10
        n=n//10
    return total

# Q26:
def wordcount(s):
    word = s.split()
    return len(word)

# Q27:

def removepunctuation(s):
    result = ''
    for char in s:
        if char.isalnum() or char.isspace():
            result += char
    return result

# Q28:
def gcd(a, b):
    while b:
        temp = a
        a=b
        b = temp % b
    return a

# Q29:
def duplicates(l):
    seen = {}
    result = []
    for i in l:
        if i in seen:
            if seen[i] == 1: 
                result.append(i)
            seen[i] += 1
        else:
            seen[i] = 1
    return result

# Q30:
def largest(a, b, c):
    if a >= b and a >= c:
        return a
    elif b >= a and b >= c:
        return b
    else:
        return c
    
# Q31:
def sort(s):
    word = s.split()
    word.sort()
    return ' '.join(word)

# Q32:
def mergedicts(d1, d2):
    return {**d1, **d2}

# Q3:
def vowelinword(sentence):
    vowels = 'aeiouAEIOU'
    result = {}
    words = sentence.split()
    for word in words:
        count = 0
        for ch in word:
            if ch in vowels:
                count += 1
        result[word] = count
    return result

# Q34:
def listis(t):
    return list(t)

# Q35:
def removespace(s):
    return ''.join(s.split())

# Q36:
def dictionary(keys, values):
    result = {}
    for i in range(len(keys)):
        result[keys[i]] = values[i]
    return result

# Q37:
def maxkey(d):
    return max(d, key=d.get)

# Q38:
def wordfrequencies(sentence):
    words = sentence.split()
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq

# Q39:
def keyis(d, key):
    return key in d

# Q40:
def replacevowels(s):
    vowels='aeiouAEIOU'
    return ''.join('*' if ch in vowels else ch for ch in s)

# Q41:
def readfile(filename):
    with open(filename, 'r') as file:
        contents = file.read()
        print(contents)

# Q42:
def countwordsinafile(filename):
    with open(filename, 'r') as file:
        text = file.read()
        words = text.split()
        return len(words)
    
# Q43:
def copyfile(source, destination):
    with open(source, 'r') as s:
        data = s.read()
    with open(destination, 'w') as d:
        d.write(data)

# Q44:

def printlonglines(filename):
    with open(filename, 'r') as file:
        for line in file:
            if len(line.strip()) > 50:
                print(line.strip())

# Q45:
def writelisttofile(filename, lines):
    with open(filename, 'w') as file:
        for line in lines:
            file.write(line + '\n')

# Q46:
def dividenumbers():
    try:
        a = int(input("Enter numerator: "))
        b = int(input("Enter denominator: "))
        result = a / b
        print("Result:", result)
    except ZeroDivisionError:
        print("Error: Cannot divide by zero!")

# Q47:

def getinteger():
    try:
        num = int(input("Enter a number: "))
        print("You entered:", num)
    except ValueError:
        print("Invalid input! Please enter a valid integer.")

# Q48:

def open_file(filename):
    try:
        with open(filename, 'r') as file:
            print(file.read())
    except FileNotFoundError:
        print("Error: File does not exist.")

# Q49:

def access_list():
    numbers = [10, 20, 30]
    try:
        index = int(input("Enter index (0 to 2): "))
        print("Element:", numbers[index])
    except IndexError:
        print("Index out of range!")
    except ValueError:
        print("Invalid index input!")


# Q50:

def demo_finally():
    try:
        x = int(input("Enter a number: "))
        y = int(input("Enter another number: "))
        print("Result:", x / y)
    except ZeroDivisionError:
        print("You can't divide by zero!")
    finally:
        print("This is the finally block. It always runs.")