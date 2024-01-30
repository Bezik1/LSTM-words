import re

SPECIAL_SIGNS = re.escape(r"!@#%^*()_=+[]{}|;\",.<>?/~`")

base_len = 30
def convert_to_same_len(sent, same_len):
    sent += "0" * (same_len - len(sent))
    return sent

def convert_data(path, size):
    train_x = []
    train_y = []

    file = open(path, 'r')
    lines = file.readlines()
    file.close()
    
    i = 0
    for line in lines:
        if(i >= size):
            break
        else:
            line = line.lower()
            line = re.sub(f'[{SPECIAL_SIGNS}]', r' ', line)

            parts = line.strip().split('\t')
            train_x.append(convert_to_same_len([t for t in parts[0].split(" ") if t != ""], base_len))
            train_y.append(convert_to_same_len([t for t in parts[1].split(" ") if t != ""], base_len))
            i += 1
    
    text = ""
    data_str = ""
    for x in lines[:size]:
        for i in range(len(x)-1, -1, -1):
            if x[i] in SPECIAL_SIGNS:
                x = x.replace(x[i], f' {x[i]}')
        data_str += x
    text = data_str.lower().split()

    return train_x, train_y, text

DATA_PATH = "data/dialogs.txt"
DATA_SIZE = 1000
train_x, train_y, text = convert_data(DATA_PATH, DATA_SIZE)

# x1 = convert_to_same_len("Jak się nazywasz?", base_len)
# y1 = convert_to_same_len("Cześć jestem Bezik", base_len)

# x2 = convert_to_same_len("Ile masz lat?", base_len)
# y2 = convert_to_same_len("Mam 18 lat", base_len)

text.append("0")
text.append(" ")
chars = set(text)
data_size, char_size = len(text), len(chars)
test_x, test_y = train_x, train_y