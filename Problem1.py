# Say "Hello, World!" With Python
if __name__ == '__main__':
    x = "Hello, World!"
    print(x)

# Python If-Else
if __name__ == '__main__':
    n = int(input())
    if n % 2 == 1:  # n is odd
        print("Weird")
    elif n % 2 == 0 and 2 <= n <= 5:  # n is even and in the inclusive range of 2 to 5
        print("Not Weird")
    elif n % 2 == 0 and 6 <= n <= 20:  # n is even and in the inclusive range of 6 to 20
        print("Weird")
    elif n % 2 == 0 and n > 20:  # n is even and greater than 20
        print("Not Weird")

# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a + b)  # addition
    print(a - b)  # subtraction
    print(a * b)  # multiplication

# Python: Division
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a // b)  # integer division
    print(a / b)  # float division

# Loops
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i ** 2)

# Write a function
def is_leap(year):
    leap = False
    if year % 4 == 0:  # leap year
        leap = True
        if year % 100 == 0:  # not leap year
            leap = False
            if year % 400 == 0:  # leap year
                leap = True
    return leap

# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range(1, n + 1):
        print(i, end="") 

# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    coordinates = []
    for i in range(x + 1):
        for j in range(y + 1):
            for k in range(z + 1):
                if i + j + k != n:
                    coordinates.append([i, j, k])

    print(coordinates)

# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arr = list(arr)
    arr.sort(reverse=True)
    for i in range(len(arr)):
        if arr[i] != arr[i + 1]:
            print(arr[i + 1])
            break

# Nested Lists
if __name__ == '__main__':
    students = []
    for _ in range(int(input())):
        name = input()
        score = float(input())
        students.append([name, score])
    students.sort(key=lambda x: x[1])  # sort by score
    second_lowest = students[0][1]
    for i in range(len(students)):
        if students[i][1] != second_lowest:
            second_lowest = students[i][1]
            break
    names = []
    for i in range(len(students)):
        if students[i][1] == second_lowest:
            names.append(students[i][0])
    names.sort()
    for i in range(len(names)):
        print(names[i])

# Lists
if __name__ == '__main__':
    N = int(input())
    my_list = []
    # Iterate through each command
    for _ in range(N):
        command = input().split()
        
        if command[0] == "insert":
            i, e = map(int, command[1:])
            my_list.insert(i, e)
        elif command[0] == "print":
            print(my_list)
        elif command[0] == "remove":
            e = int(command[1])
            my_list.remove(e)
        elif command[0] == "append":
            e = int(command[1])
            my_list.append(e)
        elif command[0] == "sort":
            my_list.sort()
        elif command[0] == "pop":
            my_list.pop()
        elif command[0] == "reverse":
            my_list.reverse()

# Tuples
if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())

    t = tuple(integer_list)

    result = hash(t)
    print(result)

# sWAP cASE
def swap_case(s):
    result = ""
    for i in range(len(s)):
        if s[i].islower():
            result += s[i].upper()
        elif s[i].isupper():
            result += s[i].lower()
        else:
            result += s[i]
    return result

# String Split and Join
def split_and_join(line):
    line = line.split(" ")
    line = "-".join(line)
    return line

# What's Your Name?
def print_full_name(first, last):
    print("Hello " + first + " " + last + "! You just delved into python.")

# Mutations
def mutate_string(string, position, character):
    string = string[:position] + character + string[position + 1:]
    return string

# Find a string
def count_substring(string, sub_string):
    count = 0
    for i in range(len(string)):
        if string[i:].startswith(sub_string):
            count += 1
    return count

# String Validators
if __name__ == '__main__':
    s = input()
    print(any([char.isalnum() for char in s]))
    print(any([char.isalpha() for char in s]))
    print(any([char.isdigit() for char in s]))
    print(any([char.islower() for char in s]))
    print(any([char.isupper() for char in s]))

# Text Alignment
#Replace all ______ with rjust, ljust or center.
thickness = int(input())  # This must be an odd number
c = 'H'
# Top Cone
for i in range(thickness):
    print((c * i).rjust(thickness - 1) + c + (c * i).ljust(thickness - 1))
# Top Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
# Middle Belt
for i in range((thickness + 1) // 2):
    print((c * thickness * 5).center(thickness * 6))
# Bottom Pillars
for i in range(thickness + 1):
    print((c * thickness).center(thickness * 2) + (c * thickness).center(thickness * 6))
# Bottom Cone
for i in range(thickness):
    print(((c * (thickness - i - 1)).rjust(thickness) + c + (c * (thickness - i - 1)).ljust(thickness)).rjust(
        thickness * 6))

# Text Wrap
def wrap(string, max_width):
    count=0
    res=""
    for i in string:
        if count==max_width:
            res+="\n"
            count=1
            res+=i
        else:
            res+=i 
            count+=1 
    return res

# Designer Door Mat
n, m = map(int, input().split())
for i in range(1, n, 2):
    print((".|." * i).center(m, "-"))
print("WELCOME".center(m, "-"))
for i in range(n - 2, -1, -2):
    print((".|." * i).center(m, "-"))

# String Formatting
def print_formatted(number):
    # Calculate the width of the binary representation of 'number'
    width = len(bin(number)) - 2

    # Loop from 1 to 'number' and print formatted values
    for i in range(1, number + 1):
        decimal = str(i)
        octal = oct(i)[2:]
        hexadecimal = hex(i)[2:].upper()
        binary = bin(i)[2:]
        # Use string formatting to pad and align the values
        print(f"{decimal.rjust(width)} {octal.rjust(width)} {hexadecimal.rjust(width)} {binary.rjust(width)}")

# Alphabet Rangoli
def print_rangoli(size):
    # your code goes here
    import string
    alpha = string.ascii_lowercase
    L = []
    for i in range(size):
        s = "-".join(alpha[i:size])
        L.append((s[::-1] + s[1:]).center(4 * size - 3, "-"))
    print('\n'.join(L[:0:-1] + L))

# Introduction to Sets
def average(array):
    myset=set()
    sum=0
    l=0
    for element in array:
        myset.add(element)
    for h in myset:
        sum+=h
    result=sum/len(myset)
    return result

# No Idea!
n, m = map(int, input().split())
arr = list(map(int, input().split()))
A = set(map(int, input().split()))
B = set(map(int, input().split()))
happiness = 0
for i in arr:
    if i in A:
        happiness += 1
    elif i in B:
        happiness -= 1
print(happiness)

# Symmetric Difference
n = int(input())
list_a = list(map(int, input().split()))
m = int(input())
list_b = list(map(int, input().split()))
set_a=set()
set_b=set()
for el in list_a:
    set_a.add(el)
for elem in list_b:
    set_b.add(elem)
intersezione=set_a.intersection(set_b)
unione=set_a.union(set_b)
sorted_union=sorted(unione)
for result in sorted_union:
    if result not in intersezione:
        print(result)

# Set .add()
n = int(input())
distinct_stamps = set()
for _ in range(n):
    stamp = input()
    distinct_stamps.add(stamp)
print(len(distinct_stamps))

# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    command = input().split()
    if command[0] == "pop":
        s.pop()
    elif command[0] == "remove":
        s.remove(int(command[1]))
    elif command[0] == "discard":
        s.discard(int(command[1]))
print(sum(s))

# Set .union() Operation
n = int(input())
english = set(map(int, input().split()))
b = int(input())
french = set(map(int, input().split()))
print(len(english.union(french)))
# Set .intersection() operation
n = int(input())
english = set(map(int, input().split()))
b = int(input())
french = set(map(int, input().split()))
print(len(english.intersection(french)))
# Set .difference() operation
n = int(input())
english = set(map(int, input().split()))
b = int(input())
french = set(map(int, input().split()))
print(len(english.difference(french)))

# Set .symmetric_difference() operation
n = int(input())
english = set(map(int, input().split()))
b = int(input())
french = set(map(int, input().split()))
print(len(english.symmetric_difference(french)))

# Set Mutations
n = int(input())
A = set(map(int, input().split()))
B = list(map(int, input().split()))
N = int(input())
for _ in range(N):
    command = input().split()
    if command[0] == "intersection_update":
        A.intersection_update(set(B))
    elif command[0] == "update":
        A.update(set(B))
    elif command[0] == "symmetric_difference_update":
        A.symmetric_difference_update(set(B))
    elif command[0] == "difference_update":
        A.difference_update(set(B))
print(sum(A))

# The Captain's Room
k = int(input())
rooms = list(map(int, input().split()))
rooms.sort()
for i in range(len(rooms)):
    if i == len(rooms) - 1:
        print(rooms[i])
        break
    if rooms[i] != rooms[i + 1]:
        print(rooms[i])
        break

# Check Subset
T=int(input())
for _ in range(T):
    x=int(input())
    A=set(map(int, input().split()))
    y=int(input())
    B=set(map(int, input().split()))
    if A.issubset(B):
        print(True)
    else:
        print(False)

# Check Strict Superset
set_a = set(map(int, input().split()))
n = int(input())
is_strict_superset = True  
for i in range(n):
    other_set = set(map(int, input().split()))
    if not set_a.issuperset(other_set):
        is_strict_superset = False
        break  # No need to check other
print(is_strict_superset)

# Collections.Counter()
from collections import Counter
X = int(input())
shoe_sizes = Counter(map(int, input().split()))
N = int(input())
total = 0
for _ in range(N):
    size, price = map(int, input().split())
    if shoe_sizes[size]:
        total += price
        shoe_sizes[size] -= 1
print(total)

# DefaultDict Tutorial
from collections import defaultdict
n, m = map(int, input().split())
d = defaultdict(list)
for i in range(n):
    d[input()].append(i + 1)
for i in range(m):
    print(*d[input()] or [-1])

# Collections.namedtuple()
from collections import namedtuple
N = int(input())
columns = input().split()
Student = namedtuple('Student', columns)
sum_of_marks = 0
total_students = 0
for _ in range(N):
    data = input().split()
    student = Student(*data)
    sum_of_marks += int(student.MARKS)
    total_students += 1
average_marks = sum_of_marks / total_students
print(f"{average_marks:.2f}")

# Collections.OrderedDict()
from collections import OrderedDict
N = int(input())
ordered_dictionary = OrderedDict()
for _ in range(N):
    item, space, price = input().rpartition(" ")
    ordered_dictionary[item] = ordered_dictionary.get(item, 0) + int(price)
for item, price in ordered_dictionary.items():
    print(item, price)


# Word Order
n = int(input())
word_count = {}
word_order = []
for i in range(n):
    word = input().strip()
    if word in word_count:
        word_count[word] += 1
    else:
        word_count[word] = 1
        word_order.append(word)
print(len(word_order))
print(*[word_count[word] for word in word_order])

# Collections.deque()
from collections import deque
d = deque()
N = int(input())
for _ in range(N):
    command = input().split()
    if command[0] == "append":
        d.append(command[1])
    elif command[0] == "appendleft":
        d.appendleft(command[1])
    elif command[0] == "pop":
        d.pop()
    elif command[0] == "popleft":
        d.popleft()
print(*d)

# Company Logo
from collections import Counter
if __name__ == '__main__':
    s = input().strip()
    char_count = Counter(s)
    sorted_chars = sorted(char_count.items(), key=lambda item: (-item[1], item[0]))
    for char, count in sorted_chars[:3]:
        print(char, count)

# Piling Up!
from collections import deque
T = int(input())
for _ in range(T):
    n = int(input())
    side_lengths = deque(map(int, input().split()))
    top_cube = float('inf')
    possible = True
    while side_lengths:
        if side_lengths[0] >= side_lengths[-1]:
            cube = side_lengths.popleft()
        else:
            cube = side_lengths.pop()
        if cube > top_cube:
            possible = False
            break
        top_cube = cube
    print("Yes" if possible else "No")

# Calendar Module
import datetime
date_input = input().strip()
date = datetime.datetime.strptime(date_input, '%m %d %Y')
day_of_week = date.strftime('%A').upper()
print(day_of_week)

# Exceptions
T=int(input())
for i in range(T):
    a, b =input().split()
    try:
        print(int(a)//int(b))
    except ZeroDivisionError as e:
        print("Error Code:", e)
    except ValueError as v:
        print("Error Code:", v)

# Zipped!
N, X = map(int, input().split())
scores = []
for _ in range(X):
    scores.append(map(float, input().split()))
for student in zip(*scores):
    print(sum(student) / X)
    # Array Manipulation
    n, m = map(int, input().split())
    array = [0] * (n + 1)
    for _ in range(m):
        a, b, k = map(int, input().split())
        array[a - 1] += k
        array[b] -= k
    max_value = 0
    running_count = 0
    for i in array:
        running_count += i
        max_value = max(max_value, running_count)
    print(max_value)

# Incorrect Regex
import re
def is_valid_regex(s):
    try:
        re.compile(s)
        return True
    except re.error:
        return False

# Map and Lambda Function
cube = lambda x: x ** 3
def fibonacci(n):
    fib = [0, 1]
    for i in range(2, n):
        fib.append(fib[i - 1] + fib[i - 2])
    return fib[:n]

# Detect Floating Point Number
import re
def is_valid_float(s):
    pattern = r'^[+-]?\d*\.\d+$'
    return bool(re.match(pattern, s))

# Re.split()
regex_pattern = r"[.,]+"
string = "Hello World!"
import re
print("\n".join(re.split(regex_pattern, string)))

# Group(), Groups() & Groupdict()  
import re
S=input()
m=r'([a-zA-Z0-9])\1+'
matches=re.search(m,S)
if matches:
    print(matches.group(1))
else:
    print(-1)

# Re.findall() & Re.finditer()
import re
S=input()
pattern=r'(?<=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])[AEIOUaeiou]{2,}(?=[QWRTYPSDFGHJKLZXCVBNMqwrtypsdfghjklzxcvbnm])'
matches=re.findall(pattern,S)
if matches:
    for match in matches:   
        print(match)
else:
    print(-1)

# Re.start() & Re.end()
import re
S=input()
k=input()
pattern=re.compile(k)
r=pattern.search(S)
if not r:
    print("(-1, -1)")
while r:
    print("({0}, {1})".format(r.start(), r.end() - 1))
    r=pattern.search(S, r.start() + 1)

# Regex Substitution
import re
def sub(match):
    symbol = match.group(0)
    if symbol == '&&':
        return 'and'
    elif symbol == '||':
        return 'or'
N = int(input())
for i in range(N):
    line = input()
    res = re.sub(r'(?<= )&&(?= )|(?<= )\|\|(?= )', sub, line)
    print(res)

# Validating Roman Numerals
regex_pattern = r"M{,3}(CM|CD|D?C{,3})(XC|XL|L?X{,3})(IX|IV|V?I{,3})$"
import re
print(str(bool(re.match(regex_pattern, input()))))

# Validating phone numbers
import re
N=int(input())
for i in range(N):
    number=input()
    if re.match(r'[789]\d{9}$',number):
        print("YES")
    else:
        print("NO")

# Validating and Parsing Email Addresses
import re
N=int(input())
for i in range(N):
    name,email=input().split()
    if re.match(r'<[a-zA-Z][\w\-.]*@[a-zA-Z]+\.[a-zA-Z]{1,3}>$',email):
        print(name,email)

# Hex Color Code
import re
N=int(input())
for i in range(N):
    line=input()
    matches=re.findall(r'(?<!^)(#(?:[\da-fA-F]{3}){1,2})',line)
    if matches:
        for match in matches:
            print(match)

# HTML Parser - Part 1
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print("Start :", tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])
    def handle_endtag(self, tag):
        print("End   :", tag)
    def handle_startendtag(self, tag, attrs):
        print("Empty :", tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])
parser = MyHTMLParser()
N = int(input())
for _ in range(N):
    parser.feed(input())

# HTML Parser - Part 2
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if "\n" in data:
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data)
    def handle_data(self, data):
        if data != "\n":
            print(">>> Data")
            print(data)
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
# You are given an HTML code snippet of N lines. Your task is to detect and print all the HTML tags, attributes and attribute values.
from html.parser import HTMLParser
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr in attrs:
            print("->", attr[0], ">", attr[1])
html = ""
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Validating UID
import re
N=int(input())
for i in range(N):
    uid=input()
    if re.match(r'(?!.*(.).*\1)(?=(?:.*[A-Z]){2,})(?=(?:.*\d){3,})[a-zA-Z0-9]{10}$',uid):
        print("Valid")
    else:
        print("Invalid")

# Validating Credit Card Numbers
import re
N=int(input())
for i in range(N):
    number=input()
    if re.match(r'^[456]\d{3}(-?\d{4}){3}$',number) and not re.search(r'(\d)\1{3,}',number.replace('-','')):
        print("Valid")
    else:
        print("Invalid")

# Validating Postal Codes
regex_integer_in_range = r"^[1-9][\d]{5}$"
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"
import re
P = input()
print(bool(re.match(regex_integer_in_range, P)) and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)
print (bool(re.match(regex_integer_in_range, P))
         and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)

# Matrix Script
import re
N, M = map(int, input().split())
matrix = []
for _ in range(N):
    matrix_item = input()
    matrix.append(matrix_item)
encoded_message = ""
for column in range(M):
    for row in range(N):
        encoded_message += matrix[row][column]
print(re.sub(r"(?<=\w)([^\w]+)(?=\w)", " ", encoded_message))

# XML 1 - Find the Score
import sys
import xml.etree.ElementTree as etree
def get_attr_number(node):
    count = 0
    for child in node.iter():
        count += len(child.attrib)
    return count
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML2 - Find the Maximum Depth
import xml.etree.ElementTree as etree
maxdepth = 0
def depth(elem, level):
    global maxdepth
    level += 1
    if level >= maxdepth:
        maxdepth = level
    for child in elem:
        depth(child, level)
    return maxdepth
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml = xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(depth(root, -1))

# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        f(["+91 " + c[-10:-5] + " " + c[-5:] for c in l])
    return fun
@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')
if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)

# Decorators 2 - Name Directory
import operator
def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner
@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]
people = [input().split() for i in range(int(input()))]
print(*name_format(people), sep='\n')

# Arrays
import numpy
def arrays(arr):
    return numpy.array(arr[::-1], float)
arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape and Reshape
import numpy
arr = numpy.array(input().split(), int)
print(numpy.reshape(arr, (3, 3)))

# Transpose and Flatten
import numpy
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
matrix = numpy.array(matrix)
print(numpy.transpose(matrix))
print(matrix.flatten())

# Concatenate 
import numpy
N, M, P = map(int, input().split())
matrix_1 = []
matrix_2 = []
for i in range(N):
    matrix_1.append(list(map(int, input().split())))
for i in range(M):
    matrix_2.append(list(map(int, input().split())))
matrix_1 = numpy.array(matrix_1)
matrix_2 = numpy.array(matrix_2)
print(numpy.concatenate((matrix_1, matrix_2), axis=0))

# Zeros and Ones
import numpy
N = tuple(map(int, input().split()))
print(numpy.zeros(N, dtype=numpy.int))
print(numpy.ones(N, dtype=numpy.int))

# Eye and Identity
import numpy
N, M = map(int, input().split())
numpy.set_printoptions(legacy='1.13')
print(numpy.eye(N, M))

# Array Mathematics
import numpy
N, M = map(int, input().split())
A = []
B = []
for i in range(N):
    A.append(list(map(int, input().split())))
for i in range(N):
    B.append(list(map(int, input().split())))
A = numpy.array(A)
B = numpy.array(B)
print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)

# Floor, Ceil and Rint
import numpy
numpy.set_printoptions(legacy='1.13')
A = numpy.array(input().split(), float)
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Sum and Prod
import numpy
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
matrix = numpy.array(matrix)
print(numpy.prod(numpy.sum(matrix, axis=0)))

# Min and Max
import numpy
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
matrix = numpy.array(matrix)
print(numpy.max(numpy.min(matrix, axis=1)))

# Mean, Var, and Std
# You are given a 2-D array of size M X N.
# Your task is to find:
# The mean along axis 1
# The var along axis 0
# The std along axis None

import numpy
N, M = map(int, input().split())
matrix = []
for i in range(N):
    matrix.append(list(map(int, input().split())))
matrix = numpy.array(matrix)
print(numpy.mean(matrix, axis=1))
print(numpy.var(matrix, axis=0))
print(round(numpy.std(matrix, axis=None), 11))

# Dot and Cross
import numpy
N = int(input())
A = []
B = []
for i in range(N):
    A.append(list(map(int, input().split())))
for i in range(N):
    B.append(list(map(int, input().split())))
A = numpy.array(A)
B = numpy.array(B)
print(numpy.dot(A, B))

# Inner and Outer
import numpy
A = numpy.array(input().split(), int)
B = numpy.array(input().split(), int)
print(numpy.inner(A, B))
print(numpy.outer(A, B))

# Polynomials
import numpy
P = list(map(float, input().split()))
x = float(input())
print(numpy.polyval(P, x))

# Linear Algebra
import numpy
N = int(input())
A = []
for i in range(N):
    A.append(list(map(float, input().split())))
A = numpy.array(A)
print(round(numpy.linalg.det(A), 2))






























