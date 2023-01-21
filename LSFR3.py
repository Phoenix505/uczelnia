import numpy as np
import random 

class LSFR_Generator():
    def __init__(self, initial_state, taps, free_word = None):
        self.register = initial_state
        self.initial_state = initial_state
        self.taps = taps
        self.free_word = free_word
        print("Initial state: ", self.register)
        if free_word is not None:
            print("Free word: ", free_word)

    def generate_bit(self):
        next_bit = np.bitwise_xor.reduce(self.register[self.taps])
        if self.free_word is not None:
            next_bit ^= self.free_word.pop(0)
            if len(self.free_word) == 0:
                self.free_word = None
        self.register = np.roll(self.register, -1)
        self.register[-1] = next_bit
        return next_bit
    def generate_sequence(self, length):
        sequence = np.zeros(length, dtype=int)
        for i in range(length):
            sequence[i] = self.generate_bit()
            ##sprawdzenie czy wartość początkowa resjestru została osiągnięta
            if all(self.register == self.initial_state):
                print(f"Initial state reached at iteration {i}")
        return sequence
    # Long_runss_test dla generatora LSFR 
    def long_runs_test(self, sequence):
        for i in range(len(sequence) - 26):
            #Warunek sprawdzajacy czy wystepuje ciaglosc wartosci 0 lub 1 przez 26 bitów
            if all(val == 0 for val in sequence[i:i+26]) or all(val == 1 for val in sequence[i:i+26]):
                return False
        return True

class Geffe_Generator():
    def __init__(self, l1, l2, l3):
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3

    def generate_sequence(self, length):
        sequence = np.zeros(length, dtype=int)
        for i in range(length):
            l1_bit = self.l1.generate_bit()
            l2_bit = self.l2.generate_bit()
            l3_bit = self.l3.generate_bit()
            sequence[i] = (l1_bit * l2_bit) ^ (l1_bit * l3_bit) ^ (l2_bit * l3_bit)
        return sequence
    #funkcja long_runs_test dla Stop and go
    def long_runs_test(self, sequence, bit_value=0):
        run_length = 0
        for i in range(len(sequence)):
            if sequence[i] == bit_value:
                run_length += 1
                if run_length == 26:
                    print(f"Test failed at iteration {i}")
                    return False
            else:
                run_length = 0
        return True

class Stop_and_Go_Generator():
    def __init__(self, l1, l2, initial_state):
        self.l1 = l1
        self.l2 = l2
        self.register = initial_state
    def generate_sequence(self, length):
        sequence = np.zeros(length, dtype=int)
        for i in range(length):
            l1_bit = self.l1.generate_bit()
            l2_bit = self.l2.generate_bit()
            next_bit = (l1_bit * self.register[0]) ^ (l2_bit * self.register[1])
            self.register = np.roll(self.register, -1)
            self.register[-1] = next_bit
            sequence[i] = next_bit
        return sequence
    #funkcja long_runs_test dla Stop and go
    def long_runs_test(self, sequence, bit_value=0):
        run_length = 0
        for i in range(len(sequence)):
            if sequence[i] == bit_value:
                run_length += 1
                if run_length == 26:
                    print(f"Test failed at iteration {i}")
                    return False
            else:
                run_length = 0
        return True



## Test 1: Geffe generator with different LSFR generators
#l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
#l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
#l3 = LSFR_Generator(np.array([1, 1, 0, 0], dtype=int), np.array([0, 2, 3], dtype=int),[0,1,0,1])
#
#geffe_gen = Geffe_Generator(l1, l2, l3)
#sequence = geffe_gen.generate_sequence(10)
#print("Test 1: Geffe generator with different LSFR generators")
#print('wyraz wolny')
#print("Generated sequence:", sequence)
#print("Long Runs Test result: ", geffe_gen.long_runs_test(sequence))
#print()
#
## Test 2: Geffe generator with different sequence length
#l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
#l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
#l3 = LSFR_Generator(np.array([1, 1, 0, 0], dtype=int), np.array([0, 2, 3], dtype=int),[0,1,0,1])
#geffe_gen = Geffe_Generator(l1, l2, l3)
#sequence = geffe_gen.generate_sequence(50)
#print("Test 2: Geffe generator with different sequence length")
#print("Generated sequence:", sequence)
#print("Long Runs Test result: ", geffe_gen.long_runs_test(sequence))
#print()
#
#
## Test 1: Stop and Go generator with different LSFR generators
#l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
#l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
#stop_and_go_gen = Stop_and_Go_Generator(l1, l2, np.array([1, 0, 1], dtype=int))
#sequence = stop_and_go_gen.generate_sequence(10)
#print("Test 1: Stop and Go generator with different LSFR generators")
#print("Generated sequence:", sequence)
#print("Long Runs Test result: ", stop_and_go_gen.long_runs_test(sequence))
#print()
#
## Test 2: Stop and Go generator with different sequence length
#l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
#l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
#stop_and_go_gen = Stop_and_Go_Generator(l1, l2, np.array([1, 0, 1], dtype=int))
#sequence = stop_and_go_gen.generate_sequence(50)
#print("Test 2: Stop and Go generator with different sequence length")
#print("Generated sequence:", sequence)
#print("Long Runs Test result: ", stop_and_go_gen.long_runs_test(sequence))
#print()


l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
sequence1 = l1.generate_sequence(10)
print('Test 1: LSFR generator with initial state [1, 0, 1, 0], taps [0, 1, 3] and free word [0, 1, 0, 1]')
print("Generated sequence: ", sequence1)
print("Long runs test: ", l1.long_runs_test(sequence1))

l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
sequence2 = l2.generate_sequence(50)
print('Test 2: LSFR generator with initial state [0, 1, 1, 0], taps [1, 2, 3] and free word [0, 1, 0, 1]')
print("Generated sequence: ", sequence2)
print("Long runs test: ", l2.long_runs_test(sequence2))

l3 = LSFR_Generator(np.array([1, 1, 0, 0], dtype=int), np.array([0, 2, 3], dtype=int))
sequence3 = l3.generate_sequence(100)
print('Test 3: LSFR generator with initial state [1, 1, 0, 0], taps [0, 2, 3] and no free word')
print("Generated sequence: ", sequence3)
print("Long runs test: ", l3.long_runs_test(sequence3))

l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
l3 = LSFR_Generator(np.array([1, 1, 0, 0], dtype=int), np.array([0, 2, 3], dtype=int),[0,1,0,1])
geffe_gen = Geffe_Generator(l1, l2, l3)
sequence4 = geffe_gen.generate_sequence(1000)
print('Test 4: Geffe generator with different LSFR generators')
print("Geffe generator sequence: ", sequence4)
print("Long runs test: ", geffe_gen.long_runs_test(sequence4))

l1 = LSFR_Generator(np.array([1, 0, 1, 0], dtype=int), np.array([0, 1, 3], dtype=int),[0,1,0,1])
l2 = LSFR_Generator(np.array([0, 1, 1, 0], dtype=int), np.array([1, 2, 3], dtype=int),[0,1,0,1])
stop_and_go_gen = Stop_and_Go_Generator(l1, l2, np.array([1, 1, 0, 0], dtype=int))
sequence5 = stop_and_go_gen.generate_sequence(1000)
print('Test 5: Stop-and-Go generator with different LSFR generators')
print("Stop-and-Go generator sequence: ", sequence5)
print("Long runs test: ", stop_and_go_gen.long_runs_test(sequence5))

random_gen = random.Random()
sequence6 = [random_gen.randint(0,1) for _ in range(1000)]
print('Test 6: Pythons built-in random module')
print("Python's built-in random module sequence: ", sequence6)
print("Long runs test: ", l1.long_runs_test(sequence6))

np_random_gen = np.random.RandomState()
sequence7 = np_random_gen.randint(0, 2, 1000)
print('Test 7: NumPys random module')
print("NumPy's random module sequence: ", sequence7)
print("Long runs test: ", l1.long_runs_test(sequence7))