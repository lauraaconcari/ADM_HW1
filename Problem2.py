# Birthday Cake Candles
# https://www.hackerrank.com/challenges/birthday-cake-candles/problem
def birthdayCakeCandles(candles):
    max_height = max(candles)
    count = candles.count(max_height)
    return count
if __name__ == '__main__':
    ar_count = int(input())
    ar = list(map(int, input().rstrip().split()))
    result = birthdayCakeCandles(ar)
    print(result)

# Number Line Jumps
def kangaroo(x1, v1, x2, v2):
    if v1 > v2 and (x2 - x1) % (v1 - v2) == 0:
        return 'YES'
    else:
        return 'NO'
if __name__ == '__main__':
    x1V1X2V2 = input().split()
    x1 = int(x1V1X2V2[0])
    v1 = int(x1V1X2V2[1])
    x2 = int(x1V1X2V2[2])
    v2 = int(x1V1X2V2[3])
    result = kangaroo(x1, v1, x2, v2)
    print(result)

# Viral Advertising
def viralAdvertising(n):
    list=[1,5,2,2]
    if n==1:
        return list[3]
    else:
        for day in range(2, n+1):
            list[0]=day
            list[1]=(list[1]//2)*3
            list[2]=list[1]//2
            list[3]+=list[2]
        return list[3]
if __name__ == '__main__':
    n = int(input())
    result = viralAdvertising(n)
    print(result)

# Recursive Digit Sum
def superDigit(n, k):
    if len(n) == 1:
        return n
    else:
        sum = 0
        for i in n:
            sum += int(i)
        return superDigit(str(sum * k), 1)
if __name__ == '__main__':
    nk = input().split()
    n = nk[0]
    k = int(nk[1])
    result = superDigit(n, k)
    print(result)

# Insertion Sort - Part 1
def insertionSort1(n, arr):
    num = arr[-1]
    for i in range(n-2, -1, -1):
        if arr[i] > num:
            arr[i+1] = arr[i]
            print(*arr)
        else:
            arr[i+1] = num
            print(*arr)
            break
    if arr[0] > num:
        arr[0] = num
        print(*arr)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort1(n, arr)

# Insertion Sort - Part 2
def insertionSort2(n, arr):
    for i in range(1, n):
        num = arr[i]
        for j in range(i-1, -1, -1):
            if arr[j] > num:
                arr[j+1] = arr[j]
            else:
                arr[j+1] = num
                break
        if arr[0] > num:
            arr[0] = num
        print(*arr)
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().rstrip().split()))
    insertionSort2(n, arr)