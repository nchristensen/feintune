# Adapted from
# https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python
# CC-BY-SA 3.0

def matching_brackets(string, idx):
    if idx < len(string) and string[idx] == "[":
        opening = [i for i, c in enumerate(string[idx + 1 :]) if c == "["]
        closing = [i for i, c in enumerate(string[idx + 1 :]) if c == "]"]
        for i, j in enumerate(closing):
            if i >= len(opening) or j < opening[i]:
                return j + idx + 1
    return -1
