import math
#소수 판별 함수
def is_prime(number):
    if number <= 1:
        return False
    if number == 2:
        return True
    if number % 2 == 0:
        return False
    for div in range(3, int(math.sqrt(number) + 1), 2):
        if number % div == 0:
            return False
    return True

#iterator에 대해서 next() 함수가 호출될 때까지 yield 실행 다음부터 재개
def get_primes(number):
    while True:
        if is_prime(number):
            yield number
        number += 1

prime_iterator = get_primes(1)